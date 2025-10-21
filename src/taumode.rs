//! Encapsulates tau selection policies for the synthetic index transform.
//
//! API:
//! - `TauMode``: policy enum
//! - `select_tau(&[f64], TauMode) -> f64``: returns a strictly positive tau (> 0)
//!
//! Notes:
//! - Energies may contain NaN/inf; this utility filters non-finite values out.
//! - If the filtered set is empty or produces a non-positive tau, a small floor is used.
//!
//! TauMode policy and selection utility for synthetic lambda transform.
//! Default is Median, use `with_synthesis` to change the default to one of the other
//!
//! The computation should run once and the graph should not be stored, only the syntethic index should be stored
//!
//! A single-pass, stored synthetic index can be designed as a per-row scalar that blends the global Rayleigh energy with a localized Laplacian-weighted roughness summary, so downstream search can use “semantic” scores together with “spectral roughness” without retaining the graph. Concretely, compute once per row r a scalar S_r that merges the Rayleigh quotient $E_r = \frac{x_r^\top L x_r}{x_r^\top x_r}$ with a Laplacian-locality moment derived from the same per-edge Dirichlet energy, normalize it, and store only S_r alongside the row; discard L afterward. This preserves the smoothness signal that Rayleigh encodes while incorporating a stable, intensity-aware normalization and local variability component, enabling alpha–beta blending at query time with no graph in memory.
//! ## Key idea
//! - Use the Rayleigh energy $E_r$ as the primary smoothness measure; it is non‑negative and scale‑invariant in x, reflecting how “wiggly” a row is over the item graph built once for the dataset.[^2][^1]
//! - Augment $E_r$ with a local Dirichlet statistic that captures how energy distributes across edges, e.g., an energy-weighted Gini or variance of per-edge contributions, to distinguish uniformly rough rows from rows whose roughness is concentrated; summarize this as $V_r$ and combine with $E_r$ into a single scalar index $S_r$ via a bounded transform that is comparable across rows.[^3][^1]
//! ## Per-row synthetic index
//! For each row vector x in R^N and Laplacian L computed once:
//! - Rayleigh energy: $E_r = \frac{x^\top L x}{x^\top x}$ with the convention $E_r = 0$ if $x^\top x = 0$.[^1][^2]
//! - Edgewise energy distribution: expand the Dirichlet energy as a sum over edges using the standard identity $x^\top L x = \sum_{(i,j)\in E} w_{ij} (x_i - x_j)^2$; define per-edge share $e_{ij} = \frac{w_{ij} (x_i - x_j)^2}{\sum w_{uv} (x_u - x_v)^2}$ if the denominator is nonzero, else 0.[^3][^1]
//! - Locality moment: compute a dispersion summary $V_r$ over {e_{ij}} such as:
//!     - Gini-like concentration: $G_r = \sum e_{ij}^2$ (higher means energy concentrated on fewer edges), or
//!     - Variance: $\operatorname{Var}_r = \sum (e_{ij} - \frac{1}{|E_x|})^2$ over edges with nonzero contribution [^3][^1].
//! - Normalization: map $E_r$ to a bounded score $E_r' = \frac{E_r}{E_r + \tau}$ with a small positive scale $\tau$ (e.g., median E across rows) to stabilize tails and keep 0–1 range; keep $G_r$ already in  by construction.`ArrowSpace`[^2][^1]
//! - Synthetic index: $S_r = \alpha \, E_r' + (1-\alpha) \, G_r$, with \$\alpha \in \$ (e.g., 0.7). Store S_r per row and discard L and all edge stats.`ArrowSpace`[^2][^1]
//! ## Why this works
//! - $E_r$ is the classical Dirichlet (Rayleigh) energy; lower means smoother over the item graph, higher means high-frequency variation; it is standard in spectral methods and tightly coupled to Laplacian eigen-analysis.[^2][^1]
//! - $G_r$ distinguishes equal-energy rows by how the energy is distributed across edges: a row with many small, diffuse irregularities differs from one with a few sharp discontinuities; a single scalar keeps this nuance without storing the graph.[^1][^3]
//! ## One-time computation plan
//! - Build the item graph Laplacian L once (any construction policy, e.g., your λτ-graph or KNN), compute S_r for all rows, then drop L.[^3][^1]
//! - Complexity: computing $x^\top L x$ is O(nnz(L)) per row using CSR; computing e_{ij} shares during the same pass has the same complexity; no extra matrices are needed; storage is O(nrows) for S_r.[^2][^3]
//! ## Practical details
//! - Zero/constant rows: if $x^\top x = 0$, set $E_r=0$ and $G_r=0$, hence $S_r=0$, which is consistent with smoothness on connected graphs.[^1][^2]
//! - Scale selection: pick $\tau$ as median or a small quantile of {E_r} to get a stable logistic-like compression; this makes S_r robust and comparable across datasets.[^2][^1]
//! - Optional calibration: z-score or rank-normalize E_r across rows before the bounded map if heavy tails; keep G_r as-is in.`ArrowSpace`[^3][^1]
//! ## Using the index
//! - At search time, blend semantic similarity with S_r proximity using the same alpha–beta logic already used for λ, replacing lambda proximity with an S_r proximity term, e.g., $1/(1+|S_q - S_i|)$; this preserves a spectral bias without reconstructing or storing any graph [^2][^1].
//! - The single scalar S_r per row can also be used for band filters, clustering seeds, or diversity penalties biased by spectral roughness, again without graphs.[^3][^1]
//! <span style="display:none">[^10][^11][^5][^6][^7][^8][^9]</span>
//! <div style="text-align: center">⁂</div>
//! [^1]: https://www.cs.yale.edu/homes/spielman/462/2007/lect7-07.pdf
//! [^2]: https://www.microsoft.com/en-us/research/wp-content/uploads/2013/07/Krishnan-SG13.pdf
//! [^3]: https://www.math.uni-potsdam.de/fileadmin/user_upload/Prof-GraphTh/Keller/KellerLenzWojciechowski_GraphsAndDiscreteDirichletSpaces_personal.pdf
//! [^5]: https://arxiv.org/html/2501.11024v1
//! [^6]: https://arxiv.org/html/2508.06123v1
//! [^7]: https://d-nb.info/1162140003/34
//! [^8]: https://www.ins.uni-bonn.de/media/public/publication-media/MA_Schwartz_2016.pdf?pk=703
//! [^9]: https://proceedings.neurips.cc/paper_files/paper/2024/file/f57de20ab7bb1540bcac55266ebb5401-Paper-Conference.pdf
//! [^10]: https://arxiv.org/pdf/2309.11251.pdf
//! [^11]: https://academic.oup.com/mnras/article-pdf/370/4/1713/3388125/mnras0370-1713.pdf
//!
//! Taumode has been tested for:
//! 1. **Non-negativity**: Fundamental property of Rayleigh quotients
//! 2. **Reasonable bounds**: Ensures values aren't pathologically large
//! 3. **Computational consistency**: Recomputation gives identical results
//! 4. **Data sensitivity**: Different inputs produce different outputs (discriminative power)
//! 5. **Numerical stability**: Values are finite (no NaN/infinity)
//! 6. **Statistical properties**: Basic variance analysis to understand feature discrimination
//!
use std::fmt;

use crate::core::ArrowSpace;
use crate::graph::GraphLaplacian;

use serde::{Deserialize, Serialize};
use sprs::CsMat;

use rayon::prelude::*;

use log::{info, trace};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum TauMode {
    Fixed(f64),
    #[default]
    Median,
    Mean,
    Percentile(f64),
}

pub const TAU_FLOOR: f64 = 1e-10;

impl TauMode {
    pub fn select_tau(energies: &[f64], mode: TauMode) -> f64 {
        match mode {
            TauMode::Fixed(t) => {
                if t.is_finite() && t > 0.0 {
                    t
                } else {
                    TAU_FLOOR
                }
            }
            TauMode::Mean => {
                let mut sum = 0.0;
                let mut cnt = 0usize;
                for &e in energies {
                    if e.is_finite() {
                        sum += e;
                        cnt += 1;
                    }
                }
                let m = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
                m.max(TAU_FLOOR)
            }
            TauMode::Median | TauMode::Percentile(_) => {
                let mut v: Vec<f64> = energies.iter().copied().filter(|x| x.is_finite()).collect();
                if v.is_empty() {
                    return TAU_FLOOR;
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if let TauMode::Percentile(p) = mode {
                    let pp = p.clamp(0.0, 1.0);
                    let idx = ((v.len() - 1) as f64 * pp).round() as usize;
                    return v[idx].max(TAU_FLOOR);
                }
                if v.len() % 2 == 1 {
                    v[v.len() / 2].max(TAU_FLOOR)
                } else {
                    let mid = 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]);
                    mid.max(TAU_FLOOR)
                }
            }
        }
    }

    /// Compute synthetic lambdas in parallel using adaptive optimization
    ///
    /// This function computes synthetic lambda values for all items in the ArrowSpace
    /// using a parallel, cache-optimized implementation with adaptive algorithm selection.
    ///
    /// # Algorithm Overview
    ///
    /// For each item vector, the synthetic lambda is computed as:
    /// ```ignore
    /// λ_synthetic = τ · E_bounded + (1-τ) · G_clamped
    /// ```
    /// where:
    /// - `E_bounded = E_raw / (E_raw + τ)` is the bounded Rayleigh quotient energy
    /// - `E_raw = (x^T · L · x) / (x^T · x)` is the raw Rayleigh quotient
    /// - `G_clamped` is the dispersion measure clamped to [0, 1]
    /// - `τ` is selected according to the `TauMode` strategy
    ///
    /// # Implementation Details
    ///
    /// 1. **Parallel Processing**: Uses Rayon to compute lambdas across all items in parallel
    /// 2. **Adaptive Selection**: Automatically chooses between sequential and parallel
    ///    computation per-item based on graph size:
    ///    - Sequential for small graphs (< 1000 nodes or < 10,000 edges)
    ///    - Parallel chunked for large graphs
    /// 3. **Graph Selection**: Uses precomputed signals if available, otherwise falls
    ///    back to the graph Laplacian
    /// 4. **Memory Efficient**: Processes items in batches with optimal chunk sizing
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(n · nnz) where n is number of items and nnz is
    ///   average non-zeros per row
    /// - **Space Complexity**: O(n) for result storage
    /// - **Parallelism**: Scales near-linearly with available CPU cores
    /// - **Cache Efficiency**: Chunked processing improves cache locality
    ///
    /// # Arguments
    ///
    /// * `aspace` - Mutable reference to ArrowSpace to update with computed lambdas
    /// * `gl` - Reference to GraphLaplacian containing the spectral information
    /// * `taumode` - Strategy for computing tau parameter:
    ///   - `TauMode::Fixed(τ)`: Use constant tau value
    ///   - `TauMode::Median`: Compute tau as median of item vector
    ///   - `TauMode::Mean`: Compute tau as mean of item vector
    ///   - `TauMode::Percentile(p)`: Use p-th percentile of item vector
    pub fn compute_taumode_lambdas_parallel(
        aspace: &mut ArrowSpace,
        gl: &GraphLaplacian,
        taumode: TauMode,
    ) {
        let n_items = aspace.nitems;
        let n_features = aspace.nfeatures;
        let num_threads = rayon::current_num_threads();
        let start_total = std::time::Instant::now();

        // Log configuration
        info!("╔═════════════════════════════════════════════════════════════╗");
        info!("║          Parallel TauMode Lambda Computation                ║");
        info!("╠═════════════════════════════════════════════════════════════╣");
        info!("║ Configuration:                                              ║");
        info!("║   Items:           {:<40} ║", n_items);
        info!("║   Features:        {:<40} ║", n_features);
        info!("║   Threads:         {:<40} ║", num_threads);
        info!("║   TauMode:         {:<40} ║", format!("{:?}", taumode));

        // Determine graph source
        let using_signals = aspace.signals.shape() != (0, 0);
        let graph = if using_signals {
            &aspace.signals
        } else {
            &gl.matrix
        };
        let (graph_rows, graph_cols) = graph.shape();
        let graph_nnz = graph.nnz();
        let sparsity = (graph_nnz as f64) / ((graph_rows * graph_cols) as f64);

        info!(
            "║   Graph Source:    {:<40} ║",
            if using_signals {
                "Precomputed Signals"
            } else {
                "Laplacian Matrix"
            }
        );
        info!("║   Graph Shape:     {}×{:<36} ║", graph_rows, graph_cols);
        info!("║   Graph NNZ:       {:<40} ║", graph_nnz);
        info!("║   Graph Sparsity:  {:<40.6} ║", sparsity);
        info!("╚═════════════════════════════════════════════════════════════╝");

        // Threshold for adaptive algorithm selection
        const PARALLEL_THRESHOLD: usize = 1000;

        // Counters for algorithm selection statistics
        use std::sync::atomic::{AtomicUsize, Ordering};
        let sequential_count = AtomicUsize::new(0);
        let parallel_count = AtomicUsize::new(0);

        info!("Starting parallel lambda computation...");
        let start_compute = std::time::Instant::now();

        // Parallel computation with adaptive algorithm selection
        let synthetic_lambdas: Vec<f64> = (0..n_items)
            .into_par_iter()
            .map(|item_idx| {
                let item = aspace.get_item(item_idx);
                let tau = Self::select_tau(&item.item, taumode);

                let n = graph.rows();
                let nnz = graph.nnz();

                // Adaptive selection: sequential for small, parallel for large
                let lambda = if n < PARALLEL_THRESHOLD || nnz < PARALLEL_THRESHOLD * 10 {
                    sequential_count.fetch_add(1, Ordering::Relaxed);
                    Self::compute_synthetic_lambda_csr(&item.item, graph, tau)
                } else {
                    parallel_count.fetch_add(1, Ordering::Relaxed);
                    Self::compute_synthetic_lambda_parallel(&item.item, graph, tau)
                };

                // Log progress for large datasets
                if n_items > 10000 && item_idx % (n_items / 10) == 0 {
                    let progress = (item_idx as f64 / n_items as f64) * 100.0;
                    info!(
                        "  Progress: {:.1}% ({}/{} items)",
                        progress, item_idx, n_items
                    );
                }

                lambda
            })
            .collect();

        let compute_time = start_compute.elapsed();

        // Log algorithm selection statistics
        let seq_count = sequential_count.load(Ordering::Relaxed);
        let par_count = parallel_count.load(Ordering::Relaxed);

        info!("╔═════════════════════════════════════════════════════════════╗");
        info!("║          Computation Statistics                             ║");
        info!("╠═════════════════════════════════════════════════════════════╣");
        info!("║   Sequential Items: {:<39} ║", seq_count);
        info!("║   Parallel Items:   {:<39} ║", par_count);
        info!("║   Compute Time:     {:<39.3?} ║", compute_time);

        // Update ArrowSpace
        let start_update = std::time::Instant::now();
        aspace.update_lambdas(synthetic_lambdas);
        let update_time = start_update.elapsed();

        let total_time = start_total.elapsed();
        let items_per_sec = n_items as f64 / total_time.as_secs_f64();

        info!("║   Update Time:      {:<39.3?} ║", update_time);
        info!("║   Total Time:       {:<39.3?} ║", total_time);
        info!("║   Throughput:       {:<39.0} items/sec ║", items_per_sec);

        // Compute lambda statistics
        #[cfg(test)]
        if !aspace.lambdas.is_empty() {
            let lambdas = &aspace.lambdas;
            let min_lambda = lambdas.iter().copied().fold(f64::INFINITY, f64::min);
            let max_lambda = lambdas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean_lambda = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            let variance = lambdas
                .iter()
                .map(|&x| (x - mean_lambda).powi(2))
                .sum::<f64>()
                / lambdas.len() as f64;
            let std_lambda = variance.sqrt();

            info!("╠═════════════════════════════════════════════════════════════╣");
            info!("║          Lambda Statistics                                  ║");
            info!("╠═════════════════════════════════════════════════════════════╣");
            info!("║   Min:              {:<39.6} ║", min_lambda);
            info!("║   Max:              {:<39.6} ║", max_lambda);
            info!("║   Mean:             {:<39.6} ║", mean_lambda);
            info!("║   Std Dev:          {:<39.6} ║", std_lambda);
            info!("║   Range:            {:<39.6} ║", max_lambda - min_lambda);
        }

        info!("╚═════════════════════════════════════════════════════════════╝");
        info!("✓ Parallel taumode lambda computation completed successfully");
    }

    /// Compute synthetic lambda for a single item using parallel processing
    ///
    /// This function computes the synthetic lambda index for a single item vector
    /// by parallelizing the computation of Rayleigh quotient energy (E) and
    /// dispersion measure (G) across graph rows.
    ///
    /// # Algorithm
    ///
    /// The synthetic lambda is computed as:
    /// ```ignore
    /// λ = τ · E_bounded + (1-τ) · G_clamped
    /// ```
    ///
    /// Where:
    /// - **E_bounded** = E_raw / (E_raw + τ) is the bounded Rayleigh energy
    /// - **E_raw** = (x^T · L · x) / (x^T · x) is the Rayleigh quotient
    /// - **G_clamped** is the dispersion measure, clamped to [0, 1]
    /// - **τ** is the tau parameter controlling the energy-dispersion tradeoff
    ///
    /// ## Computation Stages
    ///
    /// 1. **First Pass (Parallel)**:
    ///    - Compute Rayleigh quotient numerator: Σᵢⱼ xᵢ·Lᵢⱼ·xⱼ
    ///    - Compute edge energy sum: Σᵢⱼ wᵢⱼ·(xᵢ - xⱼ)²
    ///    - Both computed in single pass for efficiency
    ///
    /// 2. **Denominator (Parallel)**:
    ///    - Compute x^T·x = Σᵢ xᵢ²
    ///    - Calculate E_raw = numerator / denominator
    ///
    /// 3. **Second Pass (Parallel)**:
    ///    - Compute dispersion: G = Σᵢⱼ (wᵢⱼ·(xᵢ-xⱼ)² / edge_energy_sum)²
    ///    - Only executed if edge_energy_sum > 0
    ///
    /// 4. **Final Combination**:
    ///    - Apply bounded transformation to E_raw
    ///    - Clamp G to [0, 1]
    ///    - Blend with tau parameter
    ///
    /// # Arguments
    ///
    /// * `item_vector` - The item/feature vector (length = graph dimension)
    /// * `graph` - Sparse graph Laplacian or signal matrix (CSR format preferred)
    /// * `tau` - Tau parameter in [0, 1] controlling energy vs dispersion weight
    ///
    /// # Returns
    ///
    /// The synthetic lambda value in range [0, 1]
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(nnz) where nnz is number of non-zero entries
    /// - **Space Complexity**: O(1) per-thread temporary storage
    /// - **Parallelism**: Scales linearly with number of CPU cores
    /// - **Cache Efficiency**: Row-wise iteration maximizes cache hits
    ///
    /// # Notes
    ///
    /// - Best performance when graph is in CSR format
    /// - For small graphs (< 1000 nodes), sequential version may be faster
    /// - Logs detailed trace information when RUST_LOG=trace is set
    /// - Uses Rayon for work-stealing parallelism
    ///
    /// # See Also
    ///
    /// - `compute_synthetic_lambda_csr` - Sequential version for small graphs
    /// - `compute_taumode_lambdas_parallel` - Batch computation for all items
    pub fn compute_synthetic_lambda_parallel(
        item_vector: &[f64],
        graph: &CsMat<f64>,
        tau: f64,
    ) -> f64 {
        let n = graph.rows();
        let nnz = graph.nnz();

        trace!("=== Starting parallel synthetic lambda computation ===");
        trace!("  Graph dimensions: {}×{}", graph.rows(), graph.cols());
        trace!(
            "  Graph NNZ: {} (sparsity: {:.4}%)",
            nnz,
            (nnz as f64 / ((n * n) as f64)) * 100.0
        );
        trace!("  Tau parameter: {:.6}", tau);
        trace!(
            "  Vector norm: {:.6}",
            item_vector.iter().map(|&x| x * x).sum::<f64>().sqrt()
        );

        let start_first_pass = std::time::Instant::now();

        // Parallel first pass: compute numerator and edge_energy_sum
        let (numerator, edge_energy_sum): (f64, f64) = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = graph.outer_view(i).unwrap();
                let xi = item_vector[i];
                let mut local_numerator = 0.0;
                let mut local_edge_energy = 0.0;

                for (j, &lij) in row.iter() {
                    // Rayleigh quotient numerator
                    local_numerator += xi * lij * item_vector[j];

                    // Dispersion computation (only off-diagonal)
                    if i != j {
                        let w = (-lij).max(0.0);
                        if w > 0.0 {
                            let d = xi - item_vector[j];
                            local_edge_energy += w * d * d;
                        }
                    }
                }

                (local_numerator, local_edge_energy)
            })
            .reduce(|| (0.0, 0.0), |(n1, e1), (n2, e2)| (n1 + n2, e1 + e2));

        trace!("  First pass completed in {:?}", start_first_pass.elapsed());
        trace!("    Rayleigh numerator: {:.8}", numerator);
        trace!("    Edge energy sum: {:.8}", edge_energy_sum);

        // Parallel computation of denominator
        let start_denominator = std::time::Instant::now();
        let denominator: f64 = item_vector.par_iter().map(|&x| x * x).sum();
        let e_raw = if denominator > 1e-12 {
            numerator / denominator
        } else {
            trace!(
                "    WARNING: Near-zero denominator ({:.2e}), setting E_raw = 0",
                denominator
            );
            0.0
        };

        trace!(
            "  Denominator computed in {:?}",
            start_denominator.elapsed()
        );
        trace!("    Denominator: {:.8}", denominator);
        trace!("    E_raw (Rayleigh quotient): {:.8}", e_raw);

        // Second pass for G (parallel)
        let start_second_pass = std::time::Instant::now();
        let g_sq_sum_parts = if edge_energy_sum > 0.0 {
            trace!("  Computing dispersion (G) - second pass...");

            let result = (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = graph.outer_view(i).unwrap();
                    let xi = item_vector[i];
                    let mut local_g_sq_sum = 0.0;

                    for (j, &lij) in row.iter() {
                        if i != j {
                            let w = (-lij).max(0.0);
                            if w > 0.0 {
                                let d = xi - item_vector[j];
                                let contrib = w * d * d;
                                let share = contrib / edge_energy_sum;
                                local_g_sq_sum += share * share;
                            }
                        }
                    }

                    local_g_sq_sum
                })
                .sum();

            trace!(
                "  Second pass completed in {:?}",
                start_second_pass.elapsed()
            );
            result
        } else {
            trace!("  Skipping dispersion computation (edge_energy_sum = 0)");
            0.0
        };

        let g_raw = g_sq_sum_parts.clamp(0.0, 1.0);
        trace!("    G_raw (dispersion): {:.8}", g_sq_sum_parts);
        trace!("    G_clamped: {:.8}", g_raw);

        // Apply bounded transformation
        let e_bounded = e_raw / (e_raw + tau);
        let g_clamped = g_raw.clamp(0.0, 1.0);

        trace!("  Applying bounded transformation:");
        trace!("    E_bounded = E_raw / (E_raw + τ) = {:.8}", e_bounded);
        trace!("    G_clamped = {:.8}", g_clamped);

        let synthetic_lambda = tau * e_bounded + (1.0 - tau) * g_clamped;

        trace!("  Final synthetic lambda: {:.8}", synthetic_lambda);
        trace!(
            "    Energy contribution:     τ·E_bounded = {:.8}",
            tau * e_bounded
        );
        trace!(
            "    Dispersion contribution: (1-τ)·G = {:.8}",
            (1.0 - tau) * g_clamped
        );
        trace!("=== Parallel synthetic lambda computation complete ===");

        synthetic_lambda
    }

    /// Compute synthetic lambda using CSR-optimized parallel processing
    ///
    /// This function computes the synthetic lambda for a single item vector using
    /// parallel row iteration optimized for CSR (Compressed Sparse Row) format.
    /// It combines Rayleigh quotient energy and dispersion measures into a single
    /// synthetic index.
    ///
    /// # Algorithm
    ///
    /// Computes: λ = τ · E_bounded + (1-τ) · G_clamped
    ///
    /// Where:
    /// - E_bounded = E_raw / (E_raw + τ)
    /// - E_raw = (x^T · L · x) / (x^T · x) (Rayleigh quotient)
    /// - G_clamped = dispersion measure clamped to [0, 1]
    ///
    /// # Arguments
    ///
    /// * `item_vector` - Item/feature vector
    /// * `graph` - Sparse graph matrix (CSR or CSC format)
    /// * `tau` - Tau parameter in [0, 1]
    ///
    /// # Returns
    ///
    /// Synthetic lambda value in [0, 1]
    ///
    /// # Performance
    ///
    /// - Uses `outer_iterator()` which works efficiently for both CSR and CSC
    /// - Parallel row-wise iteration with `par_bridge()`
    /// - Two-pass algorithm with fused first pass for E and G computation
    pub fn compute_synthetic_lambda_csr(item_vector: &[f64], graph: &CsMat<f64>, tau: f64) -> f64 {
        let n = graph.rows();
        let nnz = graph.nnz();

        trace!("=== CSR synthetic lambda computation ===");
        trace!("  Graph: {}×{}, NNZ: {}", n, graph.cols(), nnz);
        trace!("  Tau: {:.6}", tau);
        trace!(
            "  Vector L2 norm: {:.6}",
            item_vector.iter().map(|&x| x * x).sum::<f64>().sqrt()
        );

        // Directly use the matrix - outer_iterator works for both CSR and CSC
        let (numerator, edge_energy_sum): (f64, f64) = graph
            .outer_iterator()
            .enumerate()
            .par_bridge()
            .map(|(i, row)| {
                let xi = item_vector[i];
                let mut local_num = 0.0;
                let mut local_edge = 0.0;

                for (j, &lij) in row.iter() {
                    local_num += xi * lij * item_vector[j];

                    if i != j {
                        let w = (-lij).max(0.0);
                        if w > 0.0 {
                            let d = xi - item_vector[j];
                            local_edge += w * d * d;
                        }
                    }
                }

                (local_num, local_edge)
            })
            .reduce(|| (0.0, 0.0), |(n1, e1), (n2, e2)| (n1 + n2, e1 + e2));

        trace!(
            "  First pass: numerator={:.8}, edge_energy={:.8}",
            numerator,
            edge_energy_sum
        );

        let denominator: f64 = item_vector.par_iter().map(|&x| x * x).sum();
        let e_raw = if denominator > 1e-12 {
            numerator / denominator
        } else {
            trace!("  WARNING: Near-zero denominator ({:.2e})", denominator);
            0.0
        };

        trace!(
            "  E_raw: {:.8} (num={:.8}, denom={:.8})",
            e_raw,
            numerator,
            denominator
        );

        let g_sq_sum = if edge_energy_sum > 0.0 {
            trace!("  Computing dispersion (G)...");
            graph
                .outer_iterator()
                .enumerate()
                .par_bridge()
                .map(|(i, row)| {
                    let xi = item_vector[i];
                    let mut local_g = 0.0;

                    for (j, &lij) in row.iter() {
                        if i != j {
                            let w = (-lij).max(0.0);
                            if w > 0.0 {
                                let d = xi - item_vector[j];
                                let contrib = w * d * d;
                                let share = contrib / edge_energy_sum;
                                local_g += share * share;
                            }
                        }
                    }

                    local_g
                })
                .sum()
        } else {
            trace!("  Skipping G computation (zero edge energy)");
            0.0
        };

        let g_raw = g_sq_sum.clamp(0.0, 1.0);
        let e_bounded = e_raw / (e_raw + tau);

        trace!("  G_raw: {:.8} (clamped from {:.8})", g_raw, g_sq_sum);
        trace!("  E_bounded: {:.8}", e_bounded);

        let result = tau * e_bounded + (1.0 - tau) * g_raw;

        trace!(
            "  Result: {:.8} = {:.8}·{:.8} + {:.8}·{:.8}",
            result,
            tau,
            e_bounded,
            1.0 - tau,
            g_raw
        );
        trace!("=== CSR computation complete ===");

        result
    }
}

impl fmt::Display for TauMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TauMode::Fixed(value) => write!(f, "Fixed({})", value),
            TauMode::Median => write!(f, "Median"),
            TauMode::Mean => write!(f, "Mean"),
            TauMode::Percentile(p) => write!(f, "Percentile({})", p),
        }
    }
}
