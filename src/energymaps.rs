//! Energy-first pipeline with projection-aware Dirichlet computation.
//! Changes from previous version:
//! - Replaces normalize_len/rayleigh_dirichlet tiling with ProjectedEnergy trait
//! - Uses ArrowSpace.projection_matrix for consistent feature-space operations
//! - Falls back to spectral signals (F×F) when available, else bounded L2

use log::{debug, info, trace, warn};
use std::cmp::Ordering;
use std::sync::Arc;

use dashmap::DashMap;
use rayon::prelude::*;
use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::builder::ArrowSpaceBuilder;
use crate::core::ArrowSpace;
use crate::eigenmaps::{ClusteredOutput, EigenMaps};
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::reduction::ImplicitProjection;
use crate::taumode::TauMode;

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
            optical_tokens: Some(50),
            trim_quantile: 0.1,
            eta: 0.1,
            steps: 4,
            split_quantile: 0.9,
            neighbor_k: 20,
            split_tau: 0.15,
            w_lambda: 1.0,
            w_disp: 0.5,
            w_dirichlet: 0.25,
            candidate_m: 32,
        }
    }
}

impl EnergyParams {
    /// Create adaptive EnergyParams from ArrowSpaceBuilder configuration.
    ///
    /// If the builder has `expected_nitems` set, uses dataset-size-aware adaptive tokens (2√N).
    /// Otherwise falls back to dimensionality-based compression heuristics.
    ///
    /// # Arguments
    /// * `builder` - Reference to ArrowSpaceBuilder with configuration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Dataset-aware (recommended for large datasets)
    /// let builder = ArrowSpaceBuilder::new()
    ///     .with_expected_items(313841)
    ///     .with_lambda_graph(1.31, 25, 15, 2.0, Some(0.535));
    /// let params = EnergyParams::new(&builder);
    /// // → optical_tokens = 1119 (from 2√313841)
    ///
    /// // Dimensionality-based (legacy behavior)
    /// let builder = ArrowSpaceBuilder::new()
    ///     .with_lambda_graph(1.31, 25, 15, 2.0, Some(0.535));
    /// let params = EnergyParams::new(&builder);
    /// // → optical_tokens based on dim_reduction_ratio
    /// ```
    pub fn new(builder: &ArrowSpaceBuilder) -> Self {
        info!("Creating adaptive EnergyParams from ArrowSpaceBuilder");

        // Extract builder parameters
        let base_k = builder.lambda_k;
        let dim_reduction_ratio = builder.rp_eps;

        // Adaptive neighbor_k: scale with graph connectivity
        // Rule: neighbor_k should be 2-3x lambda_k for dense energy graph
        let neighbor_k = (base_k * 2).max(15).min(50);

        // Adaptive candidate_m: larger pool for better neighbor selection
        // Rule: candidate_m ≈ 2-3x neighbor_k
        let candidate_m = (neighbor_k * 3).max(30).min(128);

        // Adaptive optical compression
        // Priority 1: Use dataset size if available (2√N rule)
        // Priority 2: Use dimensionality heuristic (legacy)
        let optical_tokens = if builder.n_items_original != 0 {
            // Dataset-size-aware adaptive tokens (preferred)
            let tokens = Self::compute_adaptive_tokens(builder.n_items_original);
            info!(
                "Using dataset-aware optical_tokens={} for {} items (2√N rule)",
                tokens, builder.n_items_original
            );
            Some(tokens)
        } else if builder.use_dims_reduction {
            // Dimensionality-based heuristic (fallback)
            let tokens = (80.0 / dim_reduction_ratio).ceil() as usize;
            let tokens = tokens.max(40).min(200);
            warn!(
                "Using dim-reduction heuristic: optical_tokens={} (consider setting expected_nitems for better scaling)",
                tokens
            );
            Some(tokens)
        } else {
            // Moderate compression when no context available
            warn!("No dataset size or dim reduction info; using default optical_tokens=60");
            Some(60)
        };

        debug!(
            "Adaptive params: neighbor_k={}, candidate_m={}, optical_tokens={:?}",
            neighbor_k, candidate_m, optical_tokens
        );

        Self {
            optical_tokens,
            trim_quantile: 0.1,
            eta: 0.1,
            steps: 4,
            split_quantile: 0.9,
            neighbor_k,
            split_tau: 0.15,

            // Balanced energy weights (default)
            w_lambda: 1.0,
            w_disp: 0.5,
            w_dirichlet: 0.25,

            candidate_m,
        }
    }

    /// Compute adaptive optical token budget based on dataset size.
    ///
    /// Uses the rule of thumb: 2√N tokens, clamped to [100, 2000].
    /// This provides good resolution while keeping compression effective.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let tokens = compute_adaptive_tokens(1000);    // ~63 tokens
    /// let tokens = compute_adaptive_tokens(10000);   // ~200 tokens
    /// let tokens = compute_adaptive_tokens(100000);  // ~632 tokens
    /// let tokens = compute_adaptive_tokens(313841);  // ~1119 tokens
    /// let tokens = compute_adaptive_tokens(1000000); // ~2000 tokens (clamped)
    /// ```
    pub fn compute_adaptive_tokens(nitems: usize) -> usize {
        let sqrt_n = (nitems as f64).sqrt();
        let tokens = (sqrt_n * 2.0).round() as usize;
        tokens.max(100).min(2000)
    }

    /// Creates EnergyParams with minimal compression for maximum resolution.
    ///
    /// Use when:
    /// - Self-retrieval accuracy is critical
    /// - Dataset is small (<1000 items)
    /// - Memory is not a constraint
    pub fn high_resolution(builder: &ArrowSpaceBuilder) -> Self {
        info!("Creating high-resolution EnergyParams");

        Self {
            optical_tokens: None, // No compression on sub_centroids
            neighbor_k: (builder.lambda_k * 3).max(25),
            candidate_m: (builder.lambda_k * 5).max(50),

            // Higher split threshold to create more sub_centroids
            split_quantile: 0.85,
            steps: 5,

            ..Self::new(builder)
        }
    }

    /// Creates EnergyParams optimized for large datasets (>10K items).
    ///
    /// Use when:
    /// - Dataset is large
    /// - Memory efficiency is important
    /// - Slight accuracy trade-off is acceptable
    pub fn large_dataset(builder: &ArrowSpaceBuilder) -> Self {
        info!("Creating large-dataset EnergyParams");

        Self {
            optical_tokens: Some(100), // Aggressive compression
            neighbor_k: builder.lambda_k.max(15).min(30),
            candidate_m: (builder.lambda_k * 2).max(30).min(80),

            // Fewer diffusion steps for speed
            steps: 3,
            split_quantile: 0.92,

            ..Self::new(builder)
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

    /// Compute adaptive w_lambda from normalized lambda range
    ///
    /// Uses existing `range_lambdas` field (already normalized to [0,1]).
    ///
    /// # Weight Mapping
    ///
    /// | Range    | Quality      | Weight |
    /// |----------|--------------|--------|
    /// | 0.0-0.05 | Degenerate   | 0.5    |
    /// | 0.2-0.5  | Moderate     | 1.0    |
    /// | 0.5-1.0  | Good         | 1.5+   |
    fn adaptive_w_lambda(&self) -> f64;

    /// Compute balanced (w_lambda, w_dirichlet) pair
    fn adaptive_energy_weights(&self) -> (f64, f64);

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
    ///
    /// # Returns
    /// Vector of (index, score) sorted descending by score
    fn search_energy(
        &self,
        query: &[f64],
        gl_energy: &GraphLaplacian,
        k: usize,
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
            info!(
                "Optical compression skipped: budget {} >= centroids {}",
                token_budget, x
            );
            return centroids.clone();
        }

        trace!(
            "Creating implicit projection F={} → 2D for spatial binning",
            f
        );
        let proj = Arc::new(ImplicitProjection::new(f, 2)); // [PARALLEL] wrap for sharing

        // [PARALLEL] Project all centroids in parallel
        let xy: Vec<f64> = (0..x)
            .into_par_iter()
            .flat_map(|i| {
                let row = (0..f).map(|c| *centroids.get((i, c))).collect::<Vec<_>>();
                let p2 = proj.project(&row);
                vec![p2[0], p2[1]]
            })
            .collect();

        debug!("Projected {} centroids to 2D space [parallel]", x);

        let g = (token_budget as f64).sqrt().ceil() as usize;
        let (minx, maxx, miny, maxy) = minmax2d(&xy);
        debug!(
            "Grid size: {}×{}, bounds: x=[{:.3}, {:.3}], y=[{:.3}, {:.3}]",
            g, g, minx, maxx, miny, maxy
        );

        let mut bins: Vec<Vec<usize>> = vec![Vec::new(); g * g];
        for i in 0..x {
            let px = (xy[2 * i] - minx) / (maxx - minx + 1e-9);
            let py = (xy[2 * i + 1] - miny) / (maxy - miny + 1e-9);
            let bx = (px * g as f64).floor().clamp(0.0, (g - 1) as f64) as usize;
            let by = (py * g as f64).floor().clamp(0.0, (g - 1) as f64) as usize;
            bins[by * g + bx].push(i);
        }

        let non_empty = bins.iter().filter(|b| !b.is_empty()).count();
        debug!(
            "Binned centroids: {} non-empty bins out of {}",
            non_empty,
            g * g
        );

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
                trace!(
                    "Bin {}: trimmed {} → {} members",
                    bin_idx,
                    orig_size,
                    members.len()
                );
            }
            let pooled = mean_rows(centroids, &members);
            out.extend(pooled);
            pooled_count += 1;
            if out.len() / f >= token_budget {
                debug!(
                    "Reached token budget after {} pooled centroids",
                    pooled_count
                );
                break;
            }
        }

        if out.len() / f < token_budget {
            let deficit = token_budget - (out.len() / f);
            debug!(
                "Underfilled by {} tokens, topping up with low-norm centroids [parallel]",
                deficit
            );

            // [PARALLEL] Compute norms in parallel
            let mut norms: Vec<(usize, f64)> = (0..x)
                .into_par_iter()
                .map(|i| {
                    let n = (0..f)
                        .map(|c| {
                            let v = *centroids.get((i, c));
                            v * v
                        })
                        .sum::<f64>()
                        .sqrt();
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
        info!(
            "Optical compression complete: {} → {} centroids ({:.1}% compression)",
            x,
            rows,
            100.0 * (1.0 - rows as f64 / x as f64)
        );
        DenseMatrix::<f64>::from_iterator(out.iter().copied(), rows, f, 1)
    }

    fn bootstrap_centroid_laplacian(
        centroids: &DenseMatrix<f64>,
        k: usize,
        normalise: bool,
        sparsity_check: bool,
    ) -> GraphLaplacian {
        info!(
            "EnergyMaps::bootstrap_centroid_laplacian: k={}, normalise={}",
            k, normalise
        );
        let (x, f) = centroids.shape();
        debug!(
            "Building bootstrap L₀ on {} centroids (nodes) × {} features",
            x, f
        );

        let params = GraphParams {
            eps: 1e-3,
            k: k.min(x - 1), // cap k at x-1 to avoid issues with small centroid counts
            topk: k.min(4).min(x - 1),
            p: 2.0,
            sigma: None,
            normalise,
            sparsity_check: false, // disable for small matrices
        };
        trace!(
            "GraphParams: eps={}, k={}, topk={}, p={}",
            params.eps,
            params.k,
            params.topk,
            params.p
        );

        // Build Laplacian where nodes = centroids (rows), edges based on centroid similarity
        // This produces an x×x Laplacian operating in centroid space
        let gl = build_laplacian_matrix(centroids.clone(), &params, Some(x));

        if sparsity_check == true {
            let sparsity = GraphLaplacian::sparsity(&gl.matrix);
            info!(
                "Bootstrap L₀ complete: {}×{} (centroid space), {} non-zeros, {:.2}% sparse",
                gl.shape().0,
                gl.shape().1,
                gl.nnz(),
                sparsity * 100.0
            );
        }

        assert_eq!(gl.nnodes, x, "L₀ must be in centroid space ({}×{})", x, x);
        gl
    }

    fn diffuse_and_split_subcentroids(
        centroids: &DenseMatrix<f64>,
        l0: &GraphLaplacian,
        p: &EnergyParams,
    ) -> DenseMatrix<f64> {
        info!(
            "EnergyMaps::diffuse_and_split_subcentroids: eta={:.3}, steps={}, split_q={:.2}",
            p.eta, p.steps, p.split_quantile
        );
        let (x, f) = centroids.shape();
        debug!("Diffusing {} centroids over {} steps", x, p.steps);
        let mut work = centroids.clone();

        for step in 0..p.steps {
            trace!("Diffusion step {}/{} [parallel]", step + 1, p.steps);

            // [PARALLEL] Process all columns in parallel
            let updated_cols: Vec<Vec<f64>> = (0..f)
                .into_par_iter()
                .map(|col| {
                    let col_vec: Vec<f64> = (0..x).map(|i| *work.get((i, col))).collect();
                    let l_col = l0.multiply_vector(&col_vec);
                    (0..x)
                        .map(|i| *work.get((i, col)) - p.eta * l_col[i])
                        .collect()
                })
                .collect();

            // Write back (sequential due to DenseMatrix::set not being thread-safe)
            for (col, values) in updated_cols.iter().enumerate() {
                for (i, &val) in values.iter().enumerate() {
                    work.set((i, col), val);
                }
            }
        }
        debug!("Diffusion complete after {} steps", p.steps);

        trace!(
            "Computing node energy and dispersion with neighbor_k={}",
            p.neighbor_k
        );
        let (lambda, gini) = node_energy_and_dispersion(&work, l0, p.neighbor_k);
        let lambda_stats = (
            lambda.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            lambda.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            lambda.iter().sum::<f64>() / lambda.len() as f64,
        );
        let gini_stats = (
            gini.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            gini.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            gini.iter().sum::<f64>() / gini.len() as f64,
        );
        debug!(
            "Energy: λ ∈ [{:.6}, {:.6}], mean={:.6}",
            lambda_stats.0, lambda_stats.1, lambda_stats.2
        );
        debug!(
            "Dispersion: G ∈ [{:.6}, {:.6}], mean={:.6}",
            gini_stats.0, gini_stats.1, gini_stats.2
        );

        let mut g_sorted = gini.clone();
        g_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let q_idx = ((g_sorted.len() as f64 - 1.0) * p.split_quantile).round() as usize;
        let thresh = g_sorted[q_idx];
        debug!(
            "Split threshold (quantile {:.2}): G ≥ {:.6}",
            p.split_quantile, thresh
        );

        let mut data: Vec<f64> = work.iterator(0).copied().collect();

        let split_data: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..x)
            .into_par_iter()
            .filter(|&i| gini[i] >= thresh)
            .map(|i| {
                let nbrs = topk_by_l2(&work, i, p.neighbor_k);
                let mean = mean_rows(&work, &nbrs);
                let dir = unit_diff(work.get_row(i).iterator(0).copied().collect(), &mean);
                let std_loc = local_std(work.get_row(i).iterator(0).copied().collect(), &mean);
                let tau = p.split_tau * std_loc.max(1e-6);

                let c = work.get_row(i).iterator(0).copied().collect::<Vec<_>>();
                let c1 = add_scaled(&c, &dir, tau);
                let c2 = add_scaled(&c, &dir, -tau);

                (i, c1, c2)
            })
            .collect();

        let split_count = split_data.len();
        debug!("Computed {} splits [parallel]", split_count);

        // Extend data sequentially
        for (i, c1, c2) in split_data {
            data.extend(c1);
            data.extend(c2);
            trace!("Split centroid {}: G={:.6}", i, gini[i]);
        }

        let final_rows = data.len() / f;
        info!(
            "Sub-centroid generation: {} → {} centroids ({} splits)",
            x, final_rows, split_count
        );
        DenseMatrix::<f64>::from_iterator(data.iter().copied(), final_rows, f, 1)
    }

    /// Look-up query item against lambdas computed at build time
    /// Pure lambda-only search - ultra fast linear scan
    ///
    /// Query lambda is computed once via subcentroid mapping,
    /// then search is O(N) comparison of pre-normalized lambdas.
    fn search_energy(
        &self,
        query: &[f64],
        gl_energy: &GraphLaplacian,
        k: usize,
    ) -> Vec<(usize, f64)> {
        let query_lambda = self.prepare_query_item(query, gl_energy);

        // Pre-compute query norm (once)
        let query_norm = query.iter().map(|x| x * x).sum::<f64>().sqrt();

        let mut scored: Vec<(usize, f64)> = self
            .lambdas
            .iter()
            .enumerate()
            .map(|(i, &lambda)| {
                // Lambda distance (primary ranking)
                let lambda_dist = (query_lambda - lambda).abs();

                // Tie-breaking: cosine similarity (only when lambdas match)
                let tie_breaker = if lambda_dist < 1e-9 {
                    let item = self.get_item(i);

                    // Dot product (must compute)
                    let dot: f64 = query.iter().zip(item.item.iter()).map(|(a, b)| a * b).sum();

                    // Item norm (pre-computed!)
                    let item_norm = self
                        .item_norms
                        .as_ref()
                        .map(|norms| norms[i])
                        .unwrap_or_else(|| item.item.iter().map(|x| x * x).sum::<f64>().sqrt());

                    let cosine = dot / (query_norm * item_norm + 1e-9);
                    (1.0 - cosine) * 1e-12 // Distance (smaller = better)
                } else {
                    0.0 // Different lambdas: no tie-breaking
                };

                (i, lambda_dist + tie_breaker)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Compute adaptive w_lambda from normalized lambda range
    #[inline]
    fn adaptive_w_lambda(&self) -> f64 {
        if self.range_lambdas < 1e-9 {
            return 0.5; // Degenerate case
        }

        // Linear mapping: range [0,1] → weight [0.5, 2.0]
        0.5 + 1.5 * self.range_lambdas
    }

    /// Compute balanced (w_lambda, w_dirichlet) pair
    #[inline]
    fn adaptive_energy_weights(&self) -> (f64, f64) {
        let w_lambda = self.adaptive_w_lambda();
        let w_dirichlet = 2.5 - w_lambda; // Complementary
        (w_lambda, w_dirichlet)
    }

    // /// Energy-only search with automatic adaptive weighting
    // ///
    // /// Weights are computed automatically from lambda statistics:
    // /// - w_lambda: Based on range_lambdas (discriminative power)
    // /// - w_dirichlet: Complementary to balance contribution
    // ///
    // /// No manual parameter tuning required!
    // pub fn search_energy(
    //     &self,
    //     query: &[f64],
    //     gl_energy: &GraphLaplacian,
    //     k: usize,
    // ) -> Vec<(usize, f64)> {
    //     // Auto-compute weights from lambda statistics
    //     let (w_lambda, w_dirichlet) = self.adaptive_energy_weights();

    //     debug!(
    //         "Energy search: w_λ={:.3}, w_D={:.3} (range={:.4})",
    //         w_lambda, w_dirichlet, self.range_lambdas
    //     );

    //     // Warn if lambdas are degenerate
    //     if self.range_lambdas < 0.05 {
    //         warn!(
    //             "Lambda range is small ({:.4}), results may be feature-dominated",
    //             self.range_lambdas
    //         );
    //     }

    //     let query_lambda = self.prepare_query_item(query, gl_energy);

    //     // Precompute query norm for feature distance normalization
    //     let query_norm = query.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);

    //     let mut scored: Vec<(usize, f64)> = self
    //         .lambdas
    //         .iter()
    //         .enumerate()
    //         .map(|(i, &lambda)| {
    //             let item = self.get_item(i);

    //             // Lambda distance (normalized via range_lambdas)
    //             let lambda_dist = (query_lambda - lambda).abs();

    //             // Feature distance (L2, normalized by query norm)
    //             let feat_dist: f64 = query
    //                 .iter()
    //                 .zip(item.item.iter())
    //                 .map(|(a, b)| (a - b).powi(2))
    //                 .sum::<f64>()
    //                 .sqrt() / query_norm;

    //             // Combined energy distance with adaptive weights
    //             let score = w_lambda * lambda_dist + w_dirichlet * feat_dist;

    //             (i, score)
    //         })
    //         .collect();

    //     // Sort by score (ascending = best matches first)
    //     scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    //     scored.truncate(k);

    //     scored
    // }
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
    trace!(
        "Trimming high-norm items: {} candidates, quantile={:.2} [parallel]",
        idx.len(),
        q
    );
    let f = dm.shape().1;

    // [PARALLEL] Compute norms in parallel
    let mut pairs: Vec<(usize, f64)> = idx
        .par_iter()
        .map(|&i| {
            let n = (0..f)
                .map(|c| {
                    let v = *dm.get((i, c));
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            (i, n)
        })
        .collect();

    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let cut = (pairs.len() as f64 * (1.0 - q))
        .round()
        .clamp(1.0, pairs.len() as f64) as usize;
    let result = pairs
        .into_iter()
        .take(cut)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    trace!("Trimmed to {} items [parallel]", result.len());
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
    let var =
        diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / diffs.len().max(1) as f64;
    var.sqrt()
}

/// Add a scaled direction vector to a base vector.
fn add_scaled(a: &Vec<f64>, dir: &Vec<f64>, t: f64) -> Vec<f64> {
    a.iter().zip(dir.iter()).map(|(x, d)| x + t * d).collect()
}

/// Compute element-wise difference between two vectors.
#[allow(dead_code)]
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
            let d = target
                .iterator(0)
                .zip(v.iterator(0))
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>();
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
    trace!(
        "robust_scale: median={:.6}, MAD={:.6}, scale={:.6}",
        median,
        mad,
        scale
    );
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
    trace!(
        "Computing node energy and dispersion: {} nodes, {} features, k={} [parallel]",
        n,
        f,
        k
    );

    // Compute L·X column-wise in parallel
    let lx: Vec<f64> = (0..f)
        .into_par_iter()
        .flat_map(|col| {
            let col_vec: Vec<f64> = (0..n).map(|i| *x.get((i, col))).collect();
            l.multiply_vector(&col_vec)
        })
        .collect();

    trace!("L·X precomputed [parallel]");

    // Compute lambda and gini in parallel
    let results: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = x.get_row(i);
            let lxi = (0..f).map(|c| lx[i * f + c]).collect::<Vec<_>>();
            let denom = xi.iterator(0).map(|v| v * v).sum::<f64>().max(1e-9);
            let lambda_i = xi
                .iterator(0)
                .zip(lxi.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / denom;

            let nbrs = topk_by_l2(x, i, k);
            let mut parts: Vec<f64> = Vec::with_capacity(nbrs.len());
            for &j in nbrs.iter() {
                let w = -l.matrix.get(i, j).copied().unwrap_or(0.0).max(0.0);
                let d = {
                    let cj = x.get_row(j);
                    xi.iterator(0)
                        .zip(cj.iterator(0))
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                };
                parts.push((w * d).max(0.0));
            }
            let sum = parts.iter().sum::<f64>();
            let gini_i = if sum > 0.0 {
                parts.iter().map(|e| (e / sum).powi(2)).sum::<f64>()
            } else {
                0.0
            };

            (lambda_i, gini_i)
        })
        .collect();

    let (lambda, gini): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    debug!("Energy and dispersion computed for {} nodes [parallel]", n);
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
        energy_params: EnergyParams,
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
    /// Build an ArrowSpace index using the energy-only pipeline (no cosine similarity).
    ///
    /// This method constructs a graph-based index where edges are weighted purely by energy features:
    /// node lambda (Rayleigh quotient), dispersion (edge concentration), and Dirichlet smoothness.
    /// The pipeline completely removes cosine similarity dependence from both construction and search.
    ///
    /// # Pipeline Stages
    ///
    /// 1. **Clustering & Projection**: Runs incremental clustering with optional JL dimensionality
    ///    reduction to produce a compact centroid representation.
    ///
    /// 2. **Optical Compression** (optional): If `energy_params.optical_tokens` is set, applies
    ///    2D spatial binning with low-activation pooling inspired by DeepSeek-OCR to further
    ///    compress centroids while preserving structural information.
    ///
    /// 3. **Bootstrap Laplacian L₀**: Builds an initial Euclidean kNN Laplacian over centroids
    ///    in the (possibly projected) feature space using neutral distance metrics.
    ///
    /// 4. **Diffusion & Sub-Centroid Generation**: Applies heat-flow diffusion over L₀ to smooth
    ///    the centroid manifold, then splits high-dispersion nodes along local gradients to
    ///    generate sub-centroids that better capture local geometry.
    ///
    /// 5. **Energy Laplacian Construction**: Builds the final graph where edge weights are computed
    ///    from energy distances: `d = w_λ·|Δλ| + w_G·|ΔG| + w_D·Dirichlet(Δfeatures)`, using
    ///    parallel candidate pruning and symmetric kNN with DashMap for efficiency.
    ///
    /// 6. **Taumode Lambda Computation**: Computes per-item Rayleigh quotients (lambdas) over the
    ///    energy graph using the selected synthesis mode (Mean/Median/Max), enabling energy-aware
    ///    ranking during search.
    /// 7. ...
    /// 8. ...
    ///
    /// 2x/3x slower than `build(...)`
    fn build_energy(
        &mut self,
        rows: Vec<Vec<f64>>,
        energy_params: EnergyParams,
    ) -> (ArrowSpace, GraphLaplacian) {
        assert!(
            self.use_dims_reduction == true,
            "When using build_energy, dim reduction is needed"
        );
        if self.prebuilt_spectral == true {
            panic!("Spectral mode not compatible with build_energy, please do not enable for energy search");
        }
        self.n_items_original = rows.len();

        // Step 1: Build base ArrowSpace with clustering
        let ClusteredOutput {
            mut aspace,
            mut centroids,
            ..
        } = ArrowSpace::start_clustering(self, rows);

        // Step 2: Optional optical compression on centroids
        if let Some(tokens) = energy_params.optical_tokens {
            centroids = ArrowSpace::optical_compress_centroids(
                &centroids,
                tokens,
                energy_params.trim_quantile,
            );
        }

        // Step 3: Bootstrap Laplacian on centroids
        let l0 = ArrowSpace::bootstrap_centroid_laplacian(
            &centroids,
            energy_params.neighbor_k.max(self.lambda_k),
            self.normalise,
            self.sparsity_check,
        );

        // Step 4: Diffuse and split to create sub_centroids
        let mut sub_centroids =
            ArrowSpace::diffuse_and_split_subcentroids(&centroids, &l0, &energy_params);

    // Step 5: Optional optical compression on sub_centroids
    if let Some(tokens) = energy_params.optical_tokens {
        sub_centroids = ArrowSpace::optical_compress_centroids(
            &sub_centroids,
            tokens,
            energy_params.trim_quantile,
        );
    }

    // Step 5.5: APPLY JL DIMENSION REDUCTION (matches build() pipeline)
    let (sub_centroids_reduced, reduced_dim) = if self.use_dims_reduction {
        let (n_subcentroids, current_features) = sub_centroids.shape();
        if current_features > 64 {
            use crate::reduction::{compute_jl_dimension, project_matrix, ImplicitProjection};
            
            let jl_dim = compute_jl_dimension(n_subcentroids, self.rp_eps);
            let target_dim = jl_dim.min(current_features / 2);
            
            if target_dim < current_features {
                info!(
                    "Applying JL projection to sub_centroids: {} features → {} dimensions (ε={:.2})",
                    current_features, target_dim, self.rp_eps
                );
                
                let implicit_proj = ImplicitProjection::new(current_features, target_dim);
                let projected = project_matrix(&sub_centroids, &implicit_proj);
                
                info!(
                    "Sub_centroids projection complete: {:.1}x compression",
                    current_features as f64 / target_dim as f64
                );
                
                (projected, target_dim)
            } else {
                debug!(
                    "JL target dimension {} >= current {}, skipping projection",
                    target_dim, current_features
                );
                (sub_centroids, current_features)
            }
        } else {
            debug!("Sub_centroids dimension {} too small for JL projection", current_features);
            (sub_centroids, current_features)
        }
    } else {
        panic!("build_energy requires dimensionality reduction")
    };

    info!(
        "Energy graph: {:?} centroids → {:?} sub_centroids (reduced_dim={})",
        centroids.shape(),
        sub_centroids_reduced.shape(),
        reduced_dim
    );

        // Step 6: Build energy Laplacian on sub_centroids
        let (gl_energy, _, _) = self.build_energy_laplacian(&sub_centroids_reduced, &energy_params);

        // Step 7: Compute lambdas on sub_centroids ONLY
        // Store sub_centroids for query mapping
        aspace.sub_centroids = Some(sub_centroids_reduced.clone());
        let sub_centroids_shape = sub_centroids_reduced.shape();
        
        let mut subcentroid_space = ArrowSpace::subcentroids_from_dense_matrix(sub_centroids_reduced.clone());
        subcentroid_space.taumode = self.synthesis;
        assert_eq!(
            subcentroid_space.nitems, gl_energy.shape().0,
            "Subcentroid count must match energy graph dimensions"
        );

        info!(
            "Computing lambdas on {} sub_centroids...",
            subcentroid_space.nitems
        );
        // safeguard to clear signals
        subcentroid_space.signals = sprs::CsMat::empty(sprs::CSR, 0);
        // finally compute taumode on the subcentroids
        TauMode::compute_taumode_lambdas_parallel(
            &mut subcentroid_space,
            &gl_energy,
            self.synthesis,
        );

        aspace.subcentroid_lambdas = Some(subcentroid_space.lambdas.clone());
        info!(
            "Sub_centroid λ: min={:.6}, max={:.6}, mean={:.6}",
            subcentroid_space
                .lambdas
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)),
            subcentroid_space
                .lambdas
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b)),
            subcentroid_space.lambdas.iter().sum::<f64>() / subcentroid_space.nitems as f64
        );

        // Step 8: Assign lambdas + compute norms (single parallel loop)
        info!(
            "Mapping {} items to {:?} sub_centroids and computing norms...",
            aspace.nitems,
            sub_centroids_shape
        );

        let results: Vec<(usize, f64, f64)> = (0..aspace.nitems)
            .into_par_iter()
            .map(|i| {
                let item = aspace.get_item(i);

                // Find nearest sub_centroid using L2 distance
                let mut best_idx = 0;
                let mut best_dist = f64::INFINITY;

                for sc_idx in 0..sub_centroids_reduced.shape().0 {
                    let dist: f64 = item
                        .item
                        .iter()
                        .zip(sub_centroids_reduced.get_row(sc_idx).iterator(0))
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = sc_idx;
                    }
                }

                // Compute item L2 norm for cosine tie-breaking
                let norm: f64 = item.item.iter().map(|x| x * x).sum::<f64>().sqrt();

                // Return (centroid_idx, lambda, norm)
                (best_idx, subcentroid_space.lambdas[best_idx], norm)
            })
            .collect();

        // Unzip results into separate vectors
        let (centroid_map, item_lambdas, item_norms): (Vec<_>, Vec<_>, Vec<_>) = {
            let mut cmap = Vec::with_capacity(results.len());
            let mut lambdas = Vec::with_capacity(results.len());
            let mut norms = Vec::with_capacity(results.len());

            for (cidx, lambda, norm) in results {
                cmap.push(cidx);
                lambdas.push(lambda);
                norms.push(norm);
            }

            (cmap, lambdas, norms)
        };

        // Store in aspace
        aspace.centroid_map = Some(centroid_map);
        aspace.lambdas = item_lambdas;
        aspace.item_norms = Some(item_norms);

        info!(
            "Item λ assigned: min={:.6}, max={:.6}, mean={:.6}",
            aspace.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            aspace.lambdas.iter().fold(0.0_f64, |a, &b| a.max(b)),
            aspace.lambdas.iter().sum::<f64>() / aspace.nitems as f64
        );

        info!(
            "Item norms computed: min={:.6}, max={:.6}, mean={:.6}",
            aspace
                .item_norms
                .as_ref()
                .unwrap()
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)),
            aspace
                .item_norms
                .as_ref()
                .unwrap()
                .iter()
                .fold(0.0_f64, |a, &b| a.max(b)),
            aspace.item_norms.as_ref().unwrap().iter().sum::<f64>() / aspace.nitems as f64
        );

        (aspace, gl_energy)
    }

    fn build_energy_laplacian(
        &self,
        sub_centroids: &DenseMatrix<f64>,
        energy_params: &EnergyParams,
    ) -> (GraphLaplacian, Vec<f64>, Vec<f64>) {
        info!(
            "EnergyMaps::build_energy_laplacian: k={}, w_λ={:.2}, w_G={:.2}, w_D={:.2}",
            self.lambda_k, energy_params.w_lambda, energy_params.w_disp, energy_params.w_dirichlet
        );
        let (x, f) = sub_centroids.shape();
        debug!(
            "Building energy Laplacian on {} sub-centroids × {} features",
            x, f
        );

        trace!("Bootstrapping L' for energy feature computation");
        let l_boot = ArrowSpace::bootstrap_centroid_laplacian(
            sub_centroids,
            energy_params.neighbor_k.max(self.lambda_k),
            self.normalise,
            self.sparsity_check,
        );

        trace!("Computing energy and dispersion features");
        let (lambda, gini) = node_energy_and_dispersion(
            sub_centroids,
            &l_boot,
            energy_params.neighbor_k.max(self.lambda_k),
        );
        let s_l = robust_scale(&lambda).max(1e-9);
        let s_g = robust_scale(&gini).max(1e-9);
        debug!("Robust scales: λ={:.6}, G={:.6}", s_l, s_g);

        debug!(
            "Building energy-distance kNN with candidate pruning (M={}) [parallel]",
            energy_params.candidate_m
        );

        let adjacency = Arc::new(DashMap::with_capacity(x * self.lambda_k));

        (0..x).into_par_iter().for_each(|i| {
            let cand = topm_by_l2(
                sub_centroids,
                i,
                energy_params.candidate_m.max(self.lambda_k),
            );

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

        debug!(
            "Symmetrizing adjacency: {} directed edges [parallel]",
            adjacency.len()
        );
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
        let degrees: Vec<f64> = (0..x)
            .into_par_iter()
            .map(|i| {
                sym_adjacency
                    .iter()
                    .filter(|e| {
                        let &(u, v) = e.key();
                        u == i && i != v
                    })
                    .map(|e| *e.value())
                    .sum()
            })
            .collect();

        for entry in sym_adjacency.iter() {
            let &(i, j) = entry.key();
            let &w = entry.value();
            if i != j {
                tri.add_triplet(i, j, -w);
            }
        }
        for i in 0..x {
            tri.add_triplet(i, i, degrees[i]);
        }

        let csr = tri.to_csr();
        let gl = GraphLaplacian {
            init_data: sub_centroids.clone(),
            matrix: csr,
            nnodes: x,
            graph_params: GraphParams {
                eps: self.lambda_eps,
                k: self.lambda_k,
                topk: self.lambda_topk,
                p: 2.0,
                sigma: None,
                normalise: self.normalise,
                sparsity_check: self.sparsity_check,
            },
        };

        info!(
            "Energy Laplacian built: {}×{}, {} nnz, {:.2}% sparse",
            gl.shape().0,
            gl.shape().1,
            gl.nnz(),
            GraphLaplacian::sparsity(&gl.matrix) * 100.0
        );
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
        Self {
            w_lambda: 1.0,
            w_dirichlet: 0.5,
            eps_norm: 1e-9,
        }
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
