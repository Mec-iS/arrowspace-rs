//! Taumode synthetic lambda computation
//!
//! Computes per-item synthetic lambdas by measuring spectral roughness in FEATURE SPACE.
//! The graph must be F×F (feature-to-feature Laplacian), not N×N (item-to-item).
//!
//! Key formula: S_r = τ·E_bounded + (1-τ)·G_clamped
//! where E is Rayleigh quotient energy and G is edge dispersion measure.

use crate::core::ArrowSpace;
use crate::graph::GraphLaplacian;
use log::{info, trace};
use rayon::prelude::*;
use sprs::CsMat;
use std::fmt;

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum TauMode {
    Fixed(f64),
    #[default]
    Median,
    Mean,
    Percentile(f64),
}

pub const TAU_FLOOR: f64 = 1e-10;

impl TauMode {
    /// Select tau parameter from energy distribution
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
                let (sum, cnt) = energies
                    .iter()
                    .filter(|e| e.is_finite())
                    .fold((0.0, 0), |(s, c), &e| (s + e, c + 1));
                if cnt > 0 {
                    (sum / cnt as f64).max(TAU_FLOOR)
                } else {
                    TAU_FLOOR
                }
            }
            TauMode::Median | TauMode::Percentile(_) => {
                let mut v: Vec<f64> = energies.iter().copied().filter(|x| x.is_finite()).collect();
                if v.is_empty() {
                    return TAU_FLOOR;
                }
                v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                if let TauMode::Percentile(p) = mode {
                    let pp = p.clamp(0.0, 1.0);
                    let idx = ((v.len() - 1) as f64 * pp).round() as usize;
                    return v[idx].max(TAU_FLOOR);
                }
                
                let mid = if v.len() % 2 == 1 {
                    v[v.len() / 2]
                } else {
                    0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
                };
                mid.max(TAU_FLOOR)
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

        // Determine graph source, cannot use signals in the subcentroid space or when signals are off
        let using_signals = aspace.signals.shape() != (0, 0)
            && aspace.signals.rows() == gl.matrix.shape().0
            && aspace.centroid_map.is_none(); 
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

                // Adaptive selection: sequential for small, parallel for large
                let lambda = Self::compute_synthetic_lambda(&item.item, graph, tau);

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

    /// Compute synthetic lambda for a single item (ORIGINAL LOGIC)
    ///
    /// # Arguments
    /// * `item_vector` - F-dimensional feature vector
    /// * `graph` - F×F feature-space Laplacian (sparse CSR)
    /// * `tau` - Normalization parameter
    ///
    /// # Returns
    /// Synthetic lambda S = τ·E/(E+τ) + (1-τ)·G
    pub fn compute_synthetic_lambda(
        item_vector: &[f64],
        graph: &CsMat<f64>,
        tau: f64,
    ) -> f64 {
        // Check for zero/constant vector
        if item_vector
            .iter()
            .all(|&v| approx::relative_eq!(v, 0.0, epsilon = 1e-10))
        {
            trace!("Zero vector detected, returning λ=0");
            return 0.0;
        }

        // Compute Rayleigh quotient E_raw
        let e_raw = Self::compute_rayleigh_quotient_from_matrix(graph, item_vector);

        // Compute dispersion G_raw
        let g_raw = Self::compute_item_dispersion(item_vector, graph);

        // Bounded transformation
        let e_bounded = e_raw / (e_raw + tau);
        let g_clamped = g_raw.clamp(0.0, 1.0);

        // Synthetic index
        let synthetic_lambda = tau * e_bounded + (1.0 - tau) * g_clamped;

        trace!(
            "Synthetic λ: E_raw={:.6}, G_raw={:.6}, τ={:.6}, S={:.6}",
            e_raw, g_raw, tau, synthetic_lambda
        );

        synthetic_lambda
    }

    /// Compute Rayleigh quotient: R(L,x) = x^T L x / x^T x
    ///
    /// This operates in FEATURE SPACE:
    /// - graph is F×F Laplacian (features × features)
    /// - item_vector is F-dimensional
    /// - i,j indices reference FEATURES, not items
    pub fn compute_rayleigh_quotient_from_matrix(
        matrix: &CsMat<f64>,
        vector: &[f64],
    ) -> f64 {
        let n = matrix.shape().0;

        // Compute x^T M x (numerator)
        let mut numerator = 0.0;
        for i in 0..n {
            let xi = vector[i];
            for j in 0..n {
                if let Some(&mij) = matrix.get(i, j) {
                    numerator += xi * mij * vector[j];
                }
            }
        }

        // Compute x^T x (denominator)
        let denominator: f64 = vector.iter().map(|&x| x * x).sum();

        if denominator > 1e-12 {
            (numerator / denominator).max(0.0)
        } else {
            0.0
        }
    }

    /// Compute dispersion G using edge-wise energy distribution
    ///
    /// G = Σ(e_ij)² where e_ij = w_ij(x_i - x_j)² / total_edge_energy
    fn compute_item_dispersion(item_vector: &[f64], spectrum: &CsMat<f64>) -> f64 {
        let n_features = item_vector.len();

        // Step 1: Compute total edge energy sum
        let mut edge_energy_sum = 0.0;
        for i in 0..n_features {
            let xi = item_vector[i];
            for (j, &item_j) in item_vector.iter().enumerate() {
                if i != j {
                    let lij = spectrum.get(i, j).copied().unwrap_or(0.0);
                    let w = (-lij).max(0.0); // Off-diagonal weight
                    if w > 0.0 {
                        let d = xi - item_j;
                        edge_energy_sum += w * d * d;
                    }
                }
            }
        }

        if edge_energy_sum <= 1e-12 {
            return 0.0;
        }

        // Step 2: Compute G as sum of squared normalized edge shares
        let mut g_sq_sum = 0.0;
        for i in 0..n_features {
            let xi = item_vector[i];
            for (j, &item_j) in item_vector.iter().enumerate() {
                if i != j {
                    let lij = spectrum.get(i, j).copied().unwrap_or(0.0);
                    let w = (-lij).max(0.0);
                    if w > 0.0 {
                        let d = xi - item_j;
                        let contrib = w * d * d;
                        let share = contrib / edge_energy_sum;
                        g_sq_sum += share * share;
                    }
                }
            }
        }

        g_sq_sum.clamp(0.0, 1.0)
    }

    /// Legacy compatibility: non-parallel version
    pub fn compute_taumode_lambdas(
        aspace: &mut ArrowSpace,
        gl: &GraphLaplacian,
        taumode: TauMode,
    ) {
        Self::compute_taumode_lambdas_parallel(aspace, gl, taumode);
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
