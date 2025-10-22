//! Cosine-free, energy-first pipeline with optional optical compression (inspired by DeepSeek-OCR):
//! 1) Optional 2D "contexts optical compression" over centroids to a target token budget
//! 2) Bootstrap L₀ on (compressed) centroids using neutral kNN
//! 3) Brief diffusion over L₀ and dispersion-driven sub-centroids
//! 4) Build energy-only kNN Laplacian (λ, dispersion, local Dirichlet)
//! 5) Compute taumode on original items; energy-only search (no cosine)

use log::{debug, info, trace, warn};
use std::cmp::Ordering;

use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::core::{ArrowItem, ArrowSpace};
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::reduction::ImplicitProjection;
use crate::taumode::TauMode;

#[derive(Clone, Debug)]
pub struct EnergyParams {
    pub optical_tokens: Option<usize>,
    pub trim_quantile: f64,
    pub eta: f64,
    pub steps: usize,
    pub split_quantile: f64,
    pub neighbor_k: usize,
    pub split_tau: f64,
    pub w_lambda: f64,
    pub w_disp: f64,
    pub w_dirichlet: f64,
    pub candidate_m: usize,
    pub k: usize,
    pub normalise: bool,
    pub sparsity_check: bool,
}

impl Default for EnergyParams {
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
            k: 6,
            normalise: true,
            sparsity_check: false,
        }
    }
}

pub trait EnergyMaps {
    fn optical_compress_centroids(
        centroids: &DenseMatrix<f64>,
        token_budget: usize,
        trim_quantile: f64,
    ) -> DenseMatrix<f64>;

    fn bootstrap_centroid_laplacian(
        centroids: &DenseMatrix<f64>,
        k: usize,
        normalise: bool,
        sparsity_check: bool,
    ) -> GraphLaplacian;

    fn diffuse_and_split_subcentroids(
        centroids: &DenseMatrix<f64>,
        l0: &GraphLaplacian,
        p: &EnergyParams,
    ) -> DenseMatrix<f64>;

    fn build_energy_laplacian(
        sub_centroids: &DenseMatrix<f64>,
        p: &EnergyParams,
    ) -> (GraphLaplacian, Vec<f64>, Vec<f64>);

    fn search_energy_only(
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
        DenseMatrix::new(rows, f, out, false).unwrap()
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
                let dir = unit_diff(row(&work, i), &mean);
                let std_loc = local_std(row(&work, i), &mean);
                let tau = p.split_tau * std_loc.max(1e-6);

                let c = row(&work, i);
                let c1 = add_scaled(&c, &dir, tau);
                let c2 = add_scaled(&c, &dir, -tau);
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

    fn build_energy_laplacian(
        sub_centroids: &DenseMatrix<f64>,
        p: &EnergyParams,
    ) -> (GraphLaplacian, Vec<f64>, Vec<f64>) {
        info!("EnergyMaps::build_energy_laplacian: k={}, w_λ={:.2}, w_G={:.2}, w_D={:.2}", 
            p.k, p.w_lambda, p.w_disp, p.w_dirichlet);
        let (x, f) = sub_centroids.shape();
        debug!("Building energy Laplacian on {} sub-centroids × {} features", x, f);

        trace!("Bootstrapping L' for energy feature computation");
        let l_boot = Self::bootstrap_centroid_laplacian(
            sub_centroids,
            p.neighbor_k.max(p.k),
            p.normalise,
            p.sparsity_check,
        );

        trace!("Computing energy and dispersion features");
        let (lambda, gini) = node_energy_and_dispersion(sub_centroids, &l_boot, p.neighbor_k);
        let s_l = robust_scale(&lambda).max(1e-9);
        let s_g = robust_scale(&gini).max(1e-9);
        debug!("Robust scales: λ={:.6}, G={:.6}", s_l, s_g);

        debug!("Building energy-distance kNN with candidate pruning (M={})", p.candidate_m);
        
        // Build adjacency (W) as HashMap for easy symmetrization
        let mut adjacency: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
        
        for i in 0..x {
            let cand = topm_by_l2(sub_centroids, i, p.candidate_m.max(p.k));
            trace!("Node {}: evaluating {} candidates", i, cand.len());

            let mut scored: Vec<(usize, f64)> = cand
                .into_iter()
                .filter(|&j| j != i)
                .map(|j| {
                    let d_lambda = (lambda[i] - lambda[j]).abs() / s_l;
                    let d_gini = (gini[i] - gini[j]).abs() / s_g;
                    let diff = pair_diff(sub_centroids, i, j);
                    let r_pair = rayleigh_dirichlet(&l_boot, &diff);
                    let dist = p.w_lambda * d_lambda + p.w_disp * d_gini + p.w_dirichlet * r_pair;
                    (j, dist)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            scored.truncate(p.k);

            for (j, d) in scored {
                let w = (-d).exp();
                // Store directed edge i→j
                adjacency.insert((i, j), w);
            }
        }

        debug!("Symmetrizing adjacency: {} directed edges before symmetrization", adjacency.len());
        
        // Symmetrize: for each edge (i,j), ensure (j,i) exists with max(w_ij, w_ji)
        let mut sym_adjacency: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
        let mut processed = std::collections::HashSet::new();
        
        for (&(i, j), &w_ij) in adjacency.iter() {
            if processed.contains(&(i, j)) {
                continue;
            }
            
            let w_ji = adjacency.get(&(j, i)).copied().unwrap_or(0.0);
            // Use max for symmetrization (alternative: average)
            let w_sym = w_ij.max(w_ji);
            
            sym_adjacency.insert((i, j), w_sym);
            sym_adjacency.insert((j, i), w_sym);
            processed.insert((i, j));
            processed.insert((j, i));
        }
        
        debug!("Symmetrization complete: {} symmetric edges", sym_adjacency.len());
        
        // Build Laplacian from symmetrized adjacency
        let mut tri = sprs::TriMat::<f64>::new((x, x));
        let mut degrees = vec![0.0; x];
        
        for (&(i, j), &w) in sym_adjacency.iter() {
            if i != j {
                tri.add_triplet(i, j, -w);
                degrees[i] += w;
            }
        }
        
        // Set diagonal to degree
        for i in 0..x {
            tri.add_triplet(i, i, degrees[i]);
        }
        
        let csr = tri.to_csr();
        let gl = GraphLaplacian {
            init_data: sub_centroids.clone(),
            matrix: csr,
            nnodes: x,
            graph_params: GraphParams {
                eps: 0.0,
                k: p.k,
                topk: p.k.min(4),
                p: 2.0,
                sigma: None,
                normalise: p.normalise,
                sparsity_check: p.sparsity_check,
            },
        };

        let sparsity = GraphLaplacian::sparsity(&gl.matrix);
        info!("Energy Laplacian built: {}×{}, {} non-zeros, {:.2}% sparse", 
            gl.shape().0, gl.shape().1, gl.nnz(), sparsity * 100.0);
        (gl, lambda, gini)
    }


    fn search_energy_only(
        &self,
        query: &[f64],
        gl_energy: &GraphLaplacian,
        k: usize,
        w_lambda: f64,
        w_dirichlet: f64,
    ) -> Vec<(usize, f64)> {
        info!("EnergyMaps::search_energy_only: k={}, w_λ={:.2}, w_D={:.2}", k, w_lambda, w_dirichlet);
        debug!("Query dimension: {}, index items: {}", query.len(), self.nitems);

        trace!("Preparing query λ with Laplacian and projection");
        let q_lambda = self.prepare_query_item(query, gl_energy);
        let q = self.project_query(query);
        debug!("Query λ={:.6}, projected dimension: {}", q_lambda, q.len());

        trace!("Computing energy distances for {} items", self.nitems);
        let mut scored: Vec<(usize, f64)> = (0..self.nitems)
            .map(|i| {
                let item = self.get_item(i);
                let d_lambda = (q_lambda - item.lambda).abs();
                let diff = vec_diff(&q, &item.item);
                let r_pair = rayleigh_dirichlet(gl_energy, &diff);
                let energy_dist = w_lambda * d_lambda + w_dirichlet * r_pair;
                (i, -energy_dist)
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

fn row(dm: &DenseMatrix<f64>, r: usize) -> Vec<f64> {
    (0..dm.shape().1).map(|c| *dm.get((r, c))).collect()
}

fn unit_diff(a: Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let n = (d.iter().map(|v| v * v).sum::<f64>()).sqrt().max(1e-9);
    for v in d.iter_mut() {
        *v /= n;
    }
    d
}

fn local_std(a: Vec<f64>, b: &Vec<f64>) -> f64 {
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len().max(1) as f64;
    let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / diffs.len().max(1) as f64;
    var.sqrt()
}

fn add_scaled(a: &Vec<f64>, dir: &Vec<f64>, t: f64) -> Vec<f64> {
    a.iter().zip(dir.iter()).map(|(x, d)| x + t * d).collect()
}

fn vec_diff(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn topk_by_l2(dm: &DenseMatrix<f64>, i: usize, k: usize) -> Vec<usize> {
    let target = row(dm, i);
    let mut scored: Vec<(usize, f64)> = (0..dm.shape().0)
        .filter(|&j| j != i)
        .map(|j| {
            let v = row(dm, j);
            let d = target.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
            (j, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(j, _)| j).collect()
}

fn topm_by_l2(dm: &DenseMatrix<f64>, i: usize, m: usize) -> Vec<usize> {
    topk_by_l2(dm, i, m)
}

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
        let xi = row(x, i);
        let lxi = (0..f).map(|c| lx[i * f + c]).collect::<Vec<_>>();
        let denom = xi.iter().map(|v| v * v).sum::<f64>().max(1e-9);
        lambda[i] = xi.iter().zip(lxi.iter()).map(|(a, b)| a * b).sum::<f64>() / denom;

        let nbrs = topk_by_l2(x, i, k);
        let mut parts: Vec<f64> = Vec::with_capacity(nbrs.len());
        for &j in nbrs.iter() {
            let w = -l.matrix.get(i, j).copied().unwrap_or(0.0).max(0.0);
            let d = {
                let cj = row(x, j);
                xi.iter().zip(cj.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
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

fn rayleigh_dirichlet(l: &GraphLaplacian, x: &Vec<f64>) -> f64 {
    trace!("Computing Rayleigh-Dirichlet surrogate for pair difference");
    let z = normalize_len(l.nnodes, x);
    let lz = l.multiply_vector(&z);
    let num = lz.iter().map(|v| v * v).sum::<f64>().sqrt();
    let result = (num / (1.0 + num)).min(1.0);
    trace!("Rayleigh-Dirichlet: ||Lz||={:.6}, result={:.6}", num, result);
    result
}

fn normalize_len(n: usize, v: &Vec<f64>) -> Vec<f64> {
    if v.is_empty() {
        return vec![0.0; n];
    }
    let mut out = vec![0.0; n];
    for i in 0..n {
        out[i] = v[i % v.len()];
    }
    out
}

fn pair_diff(dm: &DenseMatrix<f64>, i: usize, j: usize) -> Vec<f64> {
    let f = dm.shape().1;
    let mut out = Vec::with_capacity(f);
    for c in 0..f {
        out.push(*dm.get((i, c)) - *dm.get((j, c)));
    }
    out
}
