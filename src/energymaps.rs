//! Energy-first pipeline with projection-aware Dirichlet computation.
//! Changes from previous version:
//! - Replaces normalize_len/rayleigh_dirichlet tiling with ProjectedEnergy trait
//! - Uses ArrowSpace.projection_matrix for consistent feature-space operations
//! - Falls back to spectral signals (F×F) when available, else bounded L2

use log::{debug, info, trace, warn};
use std::cmp::Ordering;
use std::sync::Arc;

use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;
use rayon::prelude::*;
use dashmap::DashMap;

use crate::builder::ArrowSpaceBuilder;
use crate::core::ArrowSpace;
use crate::eigenmaps::{ClusteredOutput, EigenMaps};
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::reduction::ImplicitProjection;

/// Parameters for the energy-only pipeline.
///
/// Controls all stages of energy-aware graph construction: optical compression,
/// diffusion, sub-centroid splitting, and energy-distance kNN computation.
#[derive(Clone, Debug)]
pub struct EnergyParams {
    /// Target number of centroids after optical compression. `None` disables compression.
    pub optical_tokens: Option<usize>,
    /// Fraction of high-norm items to trim per spatial bin during compression (0..1).
    pub trim_quantile: f64,
    /// Diffusion step size for heat-flow smoothing over L₀.
    pub eta: f64,
    /// Number of diffusion iterations.
    pub steps: usize,
    /// Quantile threshold for splitting high-dispersion centroids (0..1).
    pub split_quantile: f64,
    /// Neighborhood size for dispersion computation and local statistics.
    pub neighbor_k: usize,
    /// Magnitude of offset when splitting centroids along local gradient.
    pub split_tau: f64,
    /// Weight for lambda proximity term in energy distance.
    pub w_lambda: f64,
    /// Weight for dispersion difference term in energy distance.
    pub w_disp: f64,
    /// Weight for Rayleigh-Dirichlet term in energy distance.
    pub w_dirichlet: f64,
    /// Number of candidate neighbors to evaluate before selecting k nearest (M ≥ k).
    pub candidate_m: usize,
}

impl Default for EnergyParams {
    /// Creates default EnergyParams with balanced weights and moderate compression.
    fn default() -> Self {
        debug!("Creating default EnergyParams");
        Self {
            optical_tokens: None,
            trim_quantile: 0.1,
            eta: 0.1,
            steps: 4,
            split_quantile: 0.9,
            neighbor_k: 8,
            split_tau: 0.15,
            w_lambda: 1.0,
            w_disp: 0.5,
            w_dirichlet: 0.25,
            candidate_m: 32,
        }
    }
}

/// Trait providing energy-only methods for ArrowSpace construction and search.
///
/// All methods remove cosine similarity dependence, using only energy (Rayleigh quotient),
/// dispersion (local edge concentration), and Dirichlet (spectral roughness) features.
pub trait EnergyMaps {
    /// Compress centroids to a target token budget using 2D spatial binning and low-activation pooling.
    ///
    /// # Arguments
    /// * `centroids` - Input centroid matrix (X × F)
    /// * `token_budget` - Target number of output centroids
    /// * `trim_quantile` - Fraction of high-norm items to remove per bin before pooling
    ///
    /// # Returns
    /// Compressed centroid matrix (≤ token_budget rows)
    fn optical_compress_centroids(
        centroids: &DenseMatrix<f64>,
        token_budget: usize,
        trim_quantile: f64,
    ) -> DenseMatrix<f64>;

    /// Build bootstrap Laplacian L₀ in centroid space using Euclidean kNN (no cosine).
    ///
    /// # Arguments
    /// * `centroids` - Centroid matrix (X × F) where rows are graph nodes
    /// * `k` - Number of nearest neighbors per node
    /// * `normalise` - Whether to use symmetric normalized Laplacian
    /// * `sparsity_check` - Whether to verify sparsity and log statistics
    ///
    /// # Returns
    /// GraphLaplacian with shape (X × X) in centroid space
    fn bootstrap_centroid_laplacian(
        centroids: &DenseMatrix<f64>,
        k: usize,
        normalise: bool,
        sparsity_check: bool,
    ) -> GraphLaplacian;

    /// Apply diffusion smoothing over L₀ and generate sub-centroids by splitting high-dispersion nodes.
    ///
    /// # Arguments
    /// * `centroids` - Input centroid matrix
    /// * `l0` - Bootstrap Laplacian for diffusion
    /// * `p` - EnergyParams controlling diffusion and splitting
    ///
    /// # Returns
    /// Augmented centroid matrix with original + split centroids
    fn diffuse_and_split_subcentroids(
        centroids: &DenseMatrix<f64>,
        l0: &GraphLaplacian,
        p: &EnergyParams,
    ) -> DenseMatrix<f64>;

    /// Perform energy-only nearest-neighbor search (no cosine).
    ///
    /// Ranks items by weighted sum of:
    /// - Lambda proximity: |λ_query - λ_item|
    /// - Rayleigh-Dirichlet: spectral roughness of feature difference
    ///
    /// # Arguments
    /// * `query` - Query vector in original feature space
    /// * `gl_energy` - Energy-based Laplacian for query lambda computation
    /// * `k` - Number of results to return
    /// * `w_lambda` - Weight for lambda proximity term
    /// * `w_dirichlet` - Weight for Rayleigh-Dirichlet term
    ///
    /// # Returns
    /// Vector of (index, score) sorted descending by score
    fn search_energy(
        &self,
        query: &[f64],
        gl_energy: &GraphLaplacian,
        k: usize,
        w_lambda: f64,
        w_dirichlet: f64,
    ) -> Vec<(usize, f64)>;
}

impl EnergyMaps for ArrowSpace {
    fn optical_compress_centroids(
        centroids: &DenseMatrix<f64>,
        token_budget: usize,
        trim_quantile: f64,
    ) -> DenseMatrix<f64> {
        info!(
            "EnergyMaps::optical_compress_centroids: target={} tokens, trim_q={:.2}",
            token_budget, trim_quantile
        );
        let (x, f) = centroids.shape();
        debug!("Input centroids: {} × {} (X centroids, F features)", x, f);

        if token_budget == 0 || token_budget >= x {
            info!("Optical compression skipped: budget {} >= centroids {}", token_budget, x);
            return centroids.clone();
        }

        trace!("Creating implicit projection F={} → 2D for spatial binning", f);
        let proj = ImplicitProjection::new(f, 2);
        let mut xy = Vec::with_capacity(x * 2);
        for i in 0..x {
            let row = (0..f).map(|c| *centroids.get((i, c))).collect::<Vec<_>>();
            let p2 = proj.project(&row);
            xy.extend([p2[0], p2[1]]);
        }
        debug!("Projected {} centroids to 2D space", x);

        let g = (token_budget as f64).sqrt().ceil() as usize;
        let (minx, maxx, miny, maxy) = minmax2d(&xy);
        debug!("Grid size: {}×{}, bounds: x=[{:.3}, {:.3}], y=[{:.3}, {:.3}]", g, g, minx, maxx, miny, maxy);

        let mut bins: Vec<Vec<usize>> = vec![Vec::new(); g * g];
        for i in 0..x {
            let px = (xy[2 * i] - minx) / (maxx - minx + 1e-9);
            let py = (xy[2 * i + 1] - miny) / (maxy - miny + 1e-9);
            let bx = (px * g as f64).floor().clamp(0.0, (g - 1) as f64) as usize;
            let by = (py * g as f64).floor().clamp(0.0, (g - 1) as f64) as usize;
            bins[by * g + bx].push(i);
        }

        let non_empty = bins.iter().filter(|b| !b.is_empty()).count();
        debug!("Binned centroids: {} non-empty bins out of {}", non_empty, g * g);

        let mut out: Vec<f64> = Vec::new();
        let mut pooled_count = 0;
        for (bin_idx, bin) in bins.into_iter().enumerate() {
            if bin.is_empty() {
                continue;
            }
            let mut members = bin;
            let orig_size = members.len();
            if members.len() > 4 {
                members = trim_high_norm(centroids, &members, trim_quantile);
                trace!("Bin {}: trimmed {} → {} members", bin_idx, orig_size, members.len());
            }
            let pooled = mean_rows(centroids, &members);
            out.extend(pooled);
            pooled_count += 1;
            if out.len() / f >= token_budget {
                debug!("Reached token budget after {} pooled centroids", pooled_count);
                break;
            }
        }

        if out.len() / f < token_budget {
            let deficit = token_budget - (out.len() / f);
            debug!("Underfilled by {} tokens, topping up with low-norm centroids", deficit);
            let mut norms: Vec<(usize, f64)> = (0..x)
                .map(|i| {
                    let n = (0..f).map(|c| { let v = *centroids.get((i, c)); v * v }).sum::<f64>().sqrt();
                    (i, n)
                })
                .collect();
            norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let mut added = 0;
            for (i, norm) in norms {
                if out.len() / f >= token_budget {
                    break;
                }
                out.extend((0..f).map(|c| *centroids.get((i, c))));
                added += 1;
                trace!("Added centroid {} with norm {:.6}", i, norm);
            }
            debug!("Top-up complete: added {} centroids", added);
        }

        let rows = out.len() / f;
        info!("Optical compression complete: {} → {} centroids ({:.1}% compression)", x, rows, 100.0 * (1.0 - rows as f64 / x as f64));
        DenseMatrix::<f64>::from_iterator(out.iter().copied(),rows, f, 1)
    }

    fn bootstrap_centroid_laplacian(
        centroids: &DenseMatrix<f64>,
        k: usize,
        normalise: bool,
        sparsity_check: bool,
    ) -> GraphLaplacian {
        info!("EnergyMaps::bootstrap_centroid_laplacian: k={}, normalise={}", k, normalise);
        let (x, f) = centroids.shape();
        debug!("Building bootstrap L₀ on {} centroids (nodes) × {} features", x, f);

        let params = GraphParams {
            eps: 1e-3,
            k: k.min(x - 1),  // cap k at x-1 to avoid issues with small centroid counts
            topk: k.min(4).min(x - 1),
            p: 2.0,
            sigma: None,
            normalise,
            sparsity_check: false,  // disable for small matrices
        };
        trace!("GraphParams: eps={}, k={}, topk={}, p={}", params.eps, params.k, params.topk, params.p);

        // Build Laplacian where nodes = centroids (rows), edges based on centroid similarity
        // This produces an x×x Laplacian operating in centroid space
        let gl = build_laplacian_matrix(centroids.clone(), &params, Some(x));
        
        if sparsity_check == true {
            let sparsity = GraphLaplacian::sparsity(&gl.matrix);
            info!("Bootstrap L₀ complete: {}×{} (centroid space), {} non-zeros, {:.2}% sparse", 
                gl.shape().0, gl.shape().1, gl.nnz(), sparsity * 100.0);
        }
        
        assert_eq!(gl.nnodes, x, "L₀ must be in centroid space ({}×{})", x, x);
        gl
    }


    fn diffuse_and_split_subcentroids(
        centroids: &DenseMatrix<f64>,
        l0: &GraphLaplacian,
        p: &EnergyParams,
    ) -> DenseMatrix<f64> {
        info!("EnergyMaps::diffuse_and_split_subcentroids: eta={:.3}, steps={}, split_q={:.2}", 
              p.eta, p.steps, p.split_quantile);
        let (x, f) = centroids.shape();
        debug!("Diffusing {} centroids over {} steps", x, p.steps);
        let mut work = centroids.clone();

        for step in 0..p.steps {
            trace!("Diffusion step {}/{}", step + 1, p.steps);
            for col in 0..f {
                let mut col_vec = vec![0.0; x];
                for i in 0..x {
                    col_vec[i] = *work.get((i, col));
                }
                let l_col = l0.multiply_vector(&col_vec);
                for i in 0..x {
                    let v = *work.get((i, col)) - p.eta * l_col[i];
                    work.set((i, col), v);
                }
            }
        }
        debug!("Diffusion complete after {} steps", p.steps);

        trace!("Computing node energy and dispersion with neighbor_k={}", p.neighbor_k);
        let (lambda, gini) = node_energy_and_dispersion(&work, l0, p.neighbor_k);
        let lambda_stats = (
            lambda.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            lambda.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            lambda.iter().sum::<f64>() / lambda.len() as f64
        );
        let gini_stats = (
            gini.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            gini.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            gini.iter().sum::<f64>() / gini.len() as f64
        );
        debug!("Energy: λ ∈ [{:.6}, {:.6}], mean={:.6}", lambda_stats.0, lambda_stats.1, lambda_stats.2);
        debug!("Dispersion: G ∈ [{:.6}, {:.6}], mean={:.6}", gini_stats.0, gini_stats.1, gini_stats.2);

        let mut g_sorted = gini.clone();
        g_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let q_idx = ((g_sorted.len() as f64 - 1.0) * p.split_quantile).round() as usize;
        let thresh = g_sorted[q_idx];
        debug!("Split threshold (quantile {:.2}): G ≥ {:.6}", p.split_quantile, thresh);

        let mut data: Vec<f64> = Vec::with_capacity(x * f * 2);
        for i in 0..x {
            for c in 0..f {
                data.push(*work.get((i, c)));
            }
        }

        let mut split_count = 0;
        for i in 0..x {
            if gini[i] >= thresh {
                let nbrs = topk_by_l2(&work, i, p.neighbor_k);
                let mean = mean_rows(&work, &nbrs);
                let dir = unit_diff(work.get_row(i).iterator(0).copied().collect(), &mean);
                let std_loc = local_std(work.get_row(i).iterator(0).copied().collect(), &mean);
                let tau = p.split_tau * std_loc.max(1e-6);

                let c = &work.get_row(i);
                let c1 = add_scaled(&c.iterator(0).copied().collect(), &dir, tau);
                let c2 = add_scaled(&c.iterator(0).copied().collect(), &dir, -tau);
                data.extend(c1);
                data.extend(c2);
                split_count += 1;
                trace!("Split centroid {}: G={:.6}, τ={:.6}", i, gini[i], tau);
            }
        }

        let final_rows = data.len() / f;
        info!("Sub-centroid generation: {} → {} centroids ({} splits)", x, final_rows, split_count);
        DenseMatrix::<f64>::from_iterator(data.iter().copied(), final_rows, f, 1)
    }

    fn search_energy(
        &self,
        query: &[f64],
        gl_energy: &GraphLaplacian,
        k: usize,
        w_lambda: f64,
        w_dirichlet: f64,
    ) -> Vec<(usize, f64)> {
        info!("EnergyMaps::search_energy: k={}, w_λ={:.2}, w_D={:.2}", k, w_lambda, w_dirichlet);
        debug!("Query dimension: {}, index items: {}", query.len(), self.nitems);

        let params = ProjectedEnergyParams {
            w_lambda,
            w_dirichlet,
            eps_norm: 1e-9,
        };

        trace!("Computing projection-aware energy scores for {} items", self.nitems);
        let mut scored: Vec<(usize, f64)> = (0..self.nitems)
            .map(|i| {
                // Use ProjectedEnergy trait for projection-consistent scoring
                let energy_dist = self.score(gl_energy, query, i, params);
                (i, -energy_dist)  // negate for descending sort
            })
            .collect();

        trace!("Sorting and truncating to top-{}", k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);

        if !scored.is_empty() {
            debug!("Search complete: {} results, top_score={:.6}, bottom_score={:.6}", 
                   scored.len(), scored[0].1, scored[scored.len() - 1].1);
        } else {
            warn!("Search returned no results for k={}", k);
        }
        scored
    }
}

// ------- helpers with logging -------

/// Compute 2D bounding box for projected points.
fn minmax2d(xy: &Vec<f64>) -> (f64, f64, f64, f64) {
    trace!("Computing 2D bounds over {} points", xy.len() / 2);
    let mut minx = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    for i in (0..xy.len()).step_by(2) {
        let x = xy[i];
        let y = xy[i + 1];
        minx = minx.min(x);
        maxx = maxx.max(x);
        miny = miny.min(y);
        maxy = maxy.max(y);
    }
    (minx, maxx, miny, maxy)
}

/// Remove high-norm items from a set using quantile-based trimming.
fn trim_high_norm(dm: &DenseMatrix<f64>, idx: &Vec<usize>, q: f64) -> Vec<usize> {
    trace!("Trimming high-norm items: {} candidates, quantile={:.2}", idx.len(), q);
    let f = dm.shape().1;
    let mut pairs: Vec<(usize, f64)> = idx
        .iter()
        .map(|&i| {
            let n = (0..f).map(|c| { let v = *dm.get((i, c)); v * v }).sum::<f64>().sqrt();
            (i, n)
        })
        .collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let cut = (pairs.len() as f64 * (1.0 - q)).round().clamp(1.0, pairs.len() as f64) as usize;
    let result = pairs.into_iter().take(cut).map(|(i, _)| i).collect::<Vec<_>>();
    trace!("Trimmed to {} items", result.len());
    result
}

/// Compute element-wise mean of selected matrix rows.
fn mean_rows(dm: &DenseMatrix<f64>, idx: &Vec<usize>) -> Vec<f64> {
    let f = dm.shape().1;
    if idx.is_empty() {
        trace!("mean_rows: empty index, returning zero vector");
        return vec![0.0; f];
    }
    trace!("Computing mean of {} rows", idx.len());
    let mut acc = vec![0.0; f];
    for &i in idx {
        for c in 0..f {
            acc[c] += *dm.get((i, c));
        }
    }
    for c in 0..f {
        acc[c] /= idx.len() as f64;
    }
    acc
}

// /// Extract a single row from a DenseMatrix as a vector.
// fn row(dm: &DenseMatrix<f64>, r: usize) -> Vec<f64> {
//     (0..dm.shape().1).map(|c| *dm.get((r, c))).collect()
// }

/// Compute unit direction vector from a to b.
fn unit_diff(a: Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let n = (d.iter().map(|v| v * v).sum::<f64>()).sqrt().max(1e-9);
    for v in d.iter_mut() {
        *v /= n;
    }
    d
}

/// Compute local standard deviation between two vectors.
fn local_std(a: Vec<f64>, b: &Vec<f64>) -> f64 {
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len().max(1) as f64;
    let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / diffs.len().max(1) as f64;
    var.sqrt()
}

/// Add a scaled direction vector to a base vector.
fn add_scaled(a: &Vec<f64>, dir: &Vec<f64>, t: f64) -> Vec<f64> {
    a.iter().zip(dir.iter()).map(|(x, d)| x + t * d).collect()
}

/// Compute element-wise difference between two vectors.
fn vec_diff(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Find k nearest neighbors by Euclidean distance in a dense matrix.
fn topk_by_l2(dm: &DenseMatrix<f64>, i: usize, k: usize) -> Vec<usize> {
    let target = dm.get_row(i);
    let mut scored: Vec<(usize, f64)> = (0..dm.shape().0)
        .filter(|&j| j != i)
        .map(|j| {
            let v = dm.get_row(j);
            let d = target.iterator(0).zip(v.iterator(0)).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
            (j, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(j, _)| j).collect()
}

/// Alias for topk_by_l2 (for candidate selection).
fn topm_by_l2(dm: &DenseMatrix<f64>, i: usize, m: usize) -> Vec<usize> {
    topk_by_l2(dm, i, m)
}

/// Compute robust scale estimate using Median Absolute Deviation (MAD).
fn robust_scale(x: &Vec<f64>) -> f64 {
    if x.is_empty() {
        trace!("robust_scale: empty vector, returning 1.0");
        return 1.0;
    }
    let mut v = x.clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = v[v.len() / 2];
    let mut devs: Vec<f64> = v.iter().map(|t| (t - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));
    let mad = devs[devs.len() / 2];
    let scale = (1.4826 * mad).max(1e-9);
    trace!("robust_scale: median={:.6}, MAD={:.6}, scale={:.6}", median, mad, scale);
    scale
}

/// Compute node energy (Rayleigh quotient) and dispersion (edge concentration) for all nodes.
///
/// # Arguments
/// * `x` - Node feature matrix (N × F)
/// * `l` - Graph Laplacian (N × N)
/// * `k` - Neighborhood size for dispersion computation
///
/// # Returns
/// Tuple of (lambda vector, gini/dispersion vector)
fn node_energy_and_dispersion(
    x: &DenseMatrix<f64>,
    l: &GraphLaplacian,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (n, f) = x.shape();
    trace!("Computing node energy and dispersion: {} nodes, {} features, k={}", n, f, k);

    let mut lx = vec![0.0; n * f];
    for col in 0..f {
        let col_vec: Vec<f64> = (0..n).map(|i| *x.get((i, col))).collect();
        let l_col = l.multiply_vector(&col_vec);
        for i in 0..n {
            lx[i * f + col] = l_col[i];
        }
    }
    trace!("L·X precomputed");

    let mut lambda = vec![0.0; n];
    let mut gini = vec![0.0; n];

    for i in 0..n {
        let xi = x.get_row(i);
        let lxi = (0..f).map(|c| lx[i * f + c]).collect::<Vec<_>>();
        let denom = xi.iterator(0).map(|v| v * v).sum::<f64>().max(1e-9);
        lambda[i] = xi.iterator(0).zip(lxi.iter()).map(|(a, b)| a * b).sum::<f64>() / denom;

        let nbrs = topk_by_l2(x, i, k);
        let mut parts: Vec<f64> = Vec::with_capacity(nbrs.len());
        for &j in nbrs.iter() {
            let w = -l.matrix.get(i, j).copied().unwrap_or(0.0).max(0.0);
            let d = {
                let cj = x.get_row(j);
                xi.iterator(0).zip(cj.iterator(0)).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
            };
            parts.push((w * d).max(0.0));
        }
        let sum = parts.iter().sum::<f64>();
        gini[i] = if sum > 0.0 {
            parts.iter().map(|e| (e / sum).powi(2)).sum::<f64>()
        } else {
            0.0
        };
    }

    debug!("Energy and dispersion computed for {} nodes", n);
    (lambda, gini)
}

/// Compute feature-wise difference between two matrix rows.
fn pair_diff(dm: &DenseMatrix<f64>, i: usize, j: usize) -> Vec<f64> {
    let f = dm.shape().1;
    let mut out = Vec::with_capacity(f);
    for c in 0..f {
        out.push(*dm.get((i, c)) - *dm.get((j, c)));
    }
    out
}

/// Builder trait for constructing energy-only ArrowSpace indices.
///
/// Extends ArrowSpaceBuilder with methods to build energy-aware Laplacian graphs
/// that remove cosine similarity dependence from both construction and search.
pub trait EnergyMapsBuilder {
    /// Build ArrowSpace using energy-only pipeline (no cosine).
    ///
    /// # Arguments
    /// * `rows` - Input dataset (N × F)
    /// * `energy_params` - Parameters controlling energy pipeline stages
    ///
    /// # Returns
    /// Tuple of (ArrowSpace with energy-computed lambdas, energy-only GraphLaplacian)
    fn build_energy(
        &mut self,
        rows: Vec<Vec<f64>>,
        energy_params: EnergyParams
    ) -> (ArrowSpace, GraphLaplacian);

    /// Build energy-distance kNN Laplacian with parallel symmetrization.
    ///
    /// Constructs graph where edges are weighted by energy distance:
    /// d = w_λ·|Δλ| + w_G·|ΔG| + w_D·Dirichlet(Δfeatures)
    ///
    /// # Arguments
    /// * `sub_centroids` - Augmented centroid matrix (after diffusion/splits)
    /// * `p` - EnergyParams with distance weights
    ///
    /// # Returns
    /// Tuple of (symmetric GraphLaplacian, lambda vector, dispersion vector)
    fn build_energy_laplacian(
        &self,
        sub_centroids: &DenseMatrix<f64>,
        p: &EnergyParams,
    ) -> (GraphLaplacian, Vec<f64>, Vec<f64>);
}

impl EnergyMapsBuilder for ArrowSpaceBuilder {
    fn build_energy(&mut self, rows: Vec<Vec<f64>>, energy_params: EnergyParams) -> (ArrowSpace, GraphLaplacian) {
        let ClusteredOutput {
            mut aspace,
            mut centroids,
            ..
        } = ArrowSpace::start_clustering(self, rows);

        if let Some(tokens) = energy_params.optical_tokens {
            centroids = ArrowSpace::optical_compress_centroids(&centroids, tokens, energy_params.trim_quantile);
        }

        let l0 = ArrowSpace::bootstrap_centroid_laplacian(
            &centroids, energy_params.neighbor_k.max(self.lambda_k), self.normalise, self.sparsity_check);
        let sub_centroids = ArrowSpace::diffuse_and_split_subcentroids(&centroids, &l0, &energy_params);
        let (gl_energy, _, _) = self.build_energy_laplacian(&sub_centroids, &energy_params);

        aspace.compute_taumode(&gl_energy);

        (aspace, gl_energy)
    }

    fn build_energy_laplacian(
        &self,
        sub_centroids: &DenseMatrix<f64>,
        energy_params: &EnergyParams,
    ) -> (GraphLaplacian, Vec<f64>, Vec<f64>) {
        info!("EnergyMaps::build_energy_laplacian: k={}, w_λ={:.2}, w_G={:.2}, w_D={:.2}", 
            self.lambda_k, energy_params.w_lambda, energy_params.w_disp, energy_params.w_dirichlet);
        let (x, f) = sub_centroids.shape();
        debug!("Building energy Laplacian on {} sub-centroids × {} features", x, f);

        trace!("Bootstrapping L' for energy feature computation");
        let l_boot = ArrowSpace::bootstrap_centroid_laplacian(
            sub_centroids, 
            energy_params.neighbor_k.max(self.lambda_k), 
            self.normalise, 
            self.sparsity_check
        );

        trace!("Computing energy and dispersion features");
        let (lambda, gini) = node_energy_and_dispersion(
            sub_centroids, 
            &l_boot, 
            energy_params.neighbor_k.max(self.lambda_k)
        );
        let s_l = robust_scale(&lambda).max(1e-9);
        let s_g = robust_scale(&gini).max(1e-9);
        debug!("Robust scales: λ={:.6}, G={:.6}", s_l, s_g);

        debug!("Building energy-distance kNN with candidate pruning (M={}) [parallel]", energy_params.candidate_m);
        
        let adjacency = Arc::new(DashMap::with_capacity(x * self.lambda_k));
        let l_boot_arc = Arc::new(l_boot);
        
        (0..x).into_par_iter().for_each(|i| {
            let cand = topm_by_l2(sub_centroids, i, energy_params.candidate_m.max(self.lambda_k));
            
            let mut scored: Vec<(usize, f64)> = cand
                .into_iter()
                .filter(|&j| j != i)
                .map(|j| {
                    let d_lambda = (lambda[i] - lambda[j]).abs() / s_l;
                    let d_gini = (gini[i] - gini[j]).abs() / s_g;
                    let diff = pair_diff(sub_centroids, i, j);
                    
                    // Use projection-aware Dirichlet: bounded L2 in feature space
                    // (no tiling since we're already in sub-centroid space)
                    let r_pair = bounded_l2_energy(&diff);
                    
                    let dist = energy_params.w_lambda * d_lambda 
                            + energy_params.w_disp * d_gini 
                            + energy_params.w_dirichlet * r_pair;
                    (j, dist)
                })
                .collect();
            
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            scored.truncate(self.lambda_k);

            for (j, d) in scored {
                let w = (-d).exp();
                adjacency.insert((i, j), w);
            }
        });

        // [Rest of symmetrization and Laplacian building unchanged]
        
        debug!("Symmetrizing adjacency: {} directed edges [parallel]", adjacency.len());
        let sym_adjacency = Arc::new(DashMap::with_capacity(adjacency.len() * 2));
        let processed = Arc::new(DashMap::with_capacity(adjacency.len()));
        
        adjacency.iter().par_bridge().for_each(|entry| {
            let &(i, j) = entry.key();
            let &w_ij = entry.value();
            let (u, v) = if i < j { (i, j) } else { (j, i) };
            
            if processed.insert((u, v), true).is_none() {
                let w_ji = adjacency.get(&(j, i)).map(|e| *e.value()).unwrap_or(0.0);
                let w_sym = w_ij.max(w_ji);
                sym_adjacency.insert((i, j), w_sym);
                sym_adjacency.insert((j, i), w_sym);
            }
        });
        
        let mut tri = sprs::TriMat::<f64>::new((x, x));
        let degrees: Vec<f64> = (0..x).into_par_iter().map(|i| {
            sym_adjacency.iter()
                .filter(|e| { let &(u, v) = e.key(); u == i && i != v })
                .map(|e| *e.value())
                .sum()
        }).collect();
        
        for entry in sym_adjacency.iter() {
            let &(i, j) = entry.key();
            let &w = entry.value();
            if i != j { tri.add_triplet(i, j, -w); }
        }
        for i in 0..x { tri.add_triplet(i, i, degrees[i]); }
        
        let csr = tri.to_csr();
        let gl = GraphLaplacian {
            init_data: sub_centroids.clone(),
            matrix: csr,
            nnodes: x,
            graph_params: GraphParams {
                eps: self.lambda_eps, k: self.lambda_k, topk: self.lambda_topk,
                p: 2.0, sigma: None, normalise: self.normalise, sparsity_check: self.sparsity_check,
            },
        };

        info!("Energy Laplacian built: {}×{}, {} nnz, {:.2}% sparse", 
            gl.shape().0, gl.shape().1, gl.nnz(), GraphLaplacian::sparsity(&gl.matrix) * 100.0);
        (gl, lambda, gini)
    }
}


/// ============================================================================
// ProjectedEnergy: Projection-aware energy scoring (replaces tiling approach)
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct ProjectedEnergyParams {
    pub w_lambda: f64,
    pub w_dirichlet: f64,
    pub eps_norm: f64,
}

impl Default for ProjectedEnergyParams {
    fn default() -> Self {
        Self { w_lambda: 1.0, w_dirichlet: 0.5, eps_norm: 1e-9 }
    }
}

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Bounded L2 energy for feature-space differences (no tiling).
fn bounded_l2_energy(diff: &[f64]) -> f64 {
    let num = l2_norm(diff);
    (num / (1.0 + num)).min(1.0)
}

trait ProjectedEnergy {
    fn project_vec(&self, v: &[f64]) -> Vec<f64>;
    fn projected_dirichlet(&self, diff_proj: &[f64]) -> f64;
    fn score(&self, gl: &GraphLaplacian, query: &[f64], item_index: usize, p: ProjectedEnergyParams) -> f64;
}

impl ProjectedEnergy for ArrowSpace {
    #[inline]
    fn project_vec(&self, v: &[f64]) -> Vec<f64> {
        if let Some(proj) = &self.projection_matrix {
            proj.project(v)
        } else {
            v.to_vec()
        }
    }

    fn projected_dirichlet(&self, diff_proj: &[f64]) -> f64 {
        // Use spectral signals (F×F) if available and dimension matches
        if self.signals.rows() > 0 && self.signals.cols() == diff_proj.len() {
            let mut y = vec![0.0; self.signals.rows()];
            for (row_idx, row) in self.signals.outer_iterator().enumerate() {
                let mut sum = 0.0f64;
                for (col_idx, &val) in row.iter() {
                    sum += val * diff_proj[col_idx];
                }
                y[row_idx] = sum;
            }
            let num = l2_norm(&y);
            return (num / (1.0 + num)).min(1.0);
        }
        // Fallback: bounded L2
        bounded_l2_energy(diff_proj)
    }

    fn score(&self, gl: &GraphLaplacian, query: &[f64], item_index: usize, p: ProjectedEnergyParams) -> f64 {
        let lambda_q = self.prepare_query_item(query, gl);
        let lambda_i = self.get_item(item_index).lambda;
        let d_lambda = (lambda_q - lambda_i).abs();

        let q_proj = self.project_vec(query);
        let it = self.get_item(item_index);
        let i_proj = self.project_vec(&it.item);
        let diff = vec_diff(&q_proj, &i_proj);

        let d_dir = self.projected_dirichlet(&diff);
        p.w_lambda * d_lambda + p.w_dirichlet * d_dir
    }
}