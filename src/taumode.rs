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

use log::{debug, trace};

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

    /// Compute synthetic lambdas using Laplacian as passed by the `GraphLaplacian`
    /// Use this to compute the Rayleigh quotient space-wide.
    ///
    /// This function implements the synthetic index computation described in the design:
    /// 1) Compute per-feature Rayleigh energy E_f and dispersion G_f using DenseMatrix operations
    /// 2) Aggregate feature contributions to per-item raw energies weighted by feature magnitudes  
    /// 3) Apply bounded transformation with tau normalization
    /// 4) Blend energy and dispersion components with alpha parameter
    /// 5) Store resulting synthetic lambdas in ArrowSpace
    ///
    /// Compute **taumode synthetic lambdas** using a formula involving:
    /// - Feature energies ($E_f$)
    /// - G values
    /// - Tau parameter (0.9)
    /// - Tau computation using median selection
    /// - Final synthetic indices calculation
    // ///
    /// # Arguments
    /// * `aspace` - Mutable ArrowSpace to update with synthetic lambdas
    /// * `gl` - GraphLaplacian with DenseMatrix representation
    /// * `alpha` - Blending parameter for energy vs dispersion (typically 0.7)
    /// * `tau_mode` - Tau selection policy for bounded transformation
    pub fn compute_taumode_lambdas(
        aspace: &mut ArrowSpace,
        gl: &GraphLaplacian,
        taumode: TauMode, // used if different taumode from ArrowSpace is needed
    ) {
        let n_items = aspace.nitems;

        // Parallel processing of all features
        let synthetic_lambdas: Vec<f64> = (0..n_items)
            .into_par_iter()
            .map(|f| {
                let item = aspace.get_item(f);
                let tau = Self::select_tau(&item.item, taumode);

                // compute synthetic index from:
                // * signals if computed in `ArrowSpace`
                // * standard L(FxN) in `GraphLaplacian`
                Self::compute_item_vector_synthetic_lambda(
                    &item.item,
                    match aspace.signals.shape() {
                        (0, 0) => &gl.matrix,
                        _ => &aspace.signals,
                    },
                    tau,
                )
            })
            .collect();

        aspace.update_lambdas(synthetic_lambdas);
    }

    /// Compute synthetic lambda for a single external query vector
    /// This follows the same algorithm as compute_taumode_lambdas but for one vector.
    /// compute synthetic index from:
    /// * aspace.signals: if computed in `ArrowSpace`
    /// * graph.matrix: standard L(FxN) in `GraphLaplacian`
    pub fn compute_item_vector_synthetic_lambda(
        item_vector: &[f64], // F-dimensional external query vector
        graph: &CsMat<f64>,  // Existing ArrowSpace with computed spectrum
        tau: f64,
    ) -> f64 {
        // assert_eq!(
        //     item_vector.len(),
        //     graph.shape().0,
        //     "Item vector length {} must match ArrowSpace features {}",
        //     item_vector.len(),
        //     graph.shape().0,
        // );

        if item_vector
            .iter()
            .all(|&v| approx::relative_eq!(v, 0.0, epsilon = 1e-10))
        {
            panic!(r#"This vector {:?} is a constant zero vector"#, item_vector)
        }

        #[cfg(debug_assertions)]
        {
            debug!("=== COMPUTING SYNTHETIC LAMBDA FOR ITEM VECTOR ===");
            debug!("Item vector dimensions: {}", item_vector.len());
        }

        // Step 1: Compute Rayleigh energy E_q = query^T * spectrum * query / (query^T * query)
        let e_raw = Self::compute_rayleigh_quotient_from_matrix(graph, item_vector);

        // Step 2: Compute dispersion G_q using edge-wise energy distribution
        let g_raw = Self::compute_item_dispersion(item_vector, graph);

        // Step 3: Apply bounded transformation
        let e_bounded = e_raw / (e_raw + tau);
        let g_clamped = g_raw.clamp(0.0, 1.0);

        // Step 4: Compute synthetic index S_q = α * E_bounded + (1-α) * G_clamped
        let synthetic_lambda = tau * e_bounded + (1.0 - tau) * g_clamped;

        #[cfg(debug_assertions)]
        trace!(
            "Query synthetic lambda: E_raw={:.8}, G_raw={:.8}, tau={:.8}, final={:.8}",
            e_raw, g_raw, tau, synthetic_lambda
        );

        synthetic_lambda
    }

    /// Compute dispersion G_q for the query vector using edge-wise energy distribution
    fn compute_item_dispersion(item_vector: &[f64], spectrum: &CsMat<f64>) -> f64 {
        let n_features = item_vector.len();

        // Compute total edge energy: sum over all edges w_ij * (x_i - x_j)^2
        let mut edge_energy_sum = 0.0;
        for i in 0..n_features {
            let xi = item_vector[i];
            for (j, item) in item_vector.iter().enumerate() {
                if i != j {
                    let lij = match spectrum.get(i, j) {
                        Some(v) => *v,
                        _ => 0.0,
                    };
                    // For Laplacian, off-diagonal entries are -w_ij, so w_ij = -lij
                    let w = (-lij).max(0.0);
                    if w > 0.0 {
                        let d = xi - item;
                        edge_energy_sum += w * d * d;
                    }
                }
            }
        }

        // Compute G_q as sum of squared normalized edge shares
        let mut g_sq_sum = 0.0;
        if edge_energy_sum > 0.0 {
            for i in 0..n_features {
                let xi = item_vector[i];
                for (j, item) in item_vector.iter().enumerate() {
                    if i != j {
                        let lij = match spectrum.get(i, j) {
                            Some(v) => *v,
                            _ => 0.0,
                        };
                        let w = (-lij).max(0.0);
                        if w > 0.0 {
                            let d = xi - item;
                            let contrib = w * d * d;
                            let share = contrib / edge_energy_sum;
                            g_sq_sum += share * share;
                        }
                    }
                }
            }
        }

        g_sq_sum.clamp(0.0, 1.0)
    }

    /// Compute Rayleigh quotient for any vector against any matrix
    ///
    /// The Rayleigh quotient is defined as: R(M, x) = (x^T M x) / (x^T x)
    /// where M is a symmetric matrix and x is a non-zero vector.
    ///
    /// Mathematical Properties:
    /// - For symmetric positive semi-definite matrices: R(M,x) ≥ 0
    /// - Scale invariant: R(M, cx) = R(M, x) for any non-zero scalar c
    /// - Bounds eigenvalues: λ_min ≤ R(M,x) ≤ λ_max for any x
    /// - Maximized by largest eigenvector, minimized by smallest eigenvector
    ///
    /// # Arguments
    /// * `matrix` - A symmetric matrix (typically a Laplacian or similarity matrix)
    /// * `vector` - The vector for which to compute the quotient
    ///
    /// # Returns
    /// The Rayleigh quotient value
    ///
    /// # Panics
    /// Panics if vector length doesn't match matrix dimensions
    pub fn compute_rayleigh_quotient_from_matrix(matrix: &CsMat<f64>, vector: &[f64]) -> f64 {
        let n = matrix.shape().0;
        // assert_eq!(
        //     vector.len(),
        //     n,
        //     "Vector length {} must match matrix size {}",
        //     vector.len(),
        //     n
        // );
        assert_eq!(
            matrix.shape().0,
            matrix.shape().1,
            "Matrix must be square for Rayleigh quotient computation"
        );

        // Compute x^T M x (numerator)
        let mut numerator = 0.0;
        for i in 0..n {
            for j in 0..n {
                match matrix.get(i, j) {
                    Some(value) => numerator += vector[i] * value * vector[j],
                    _ => numerator += 0.0,
                };
            }
        }

        // Compute x^T x (denominator)
        let denominator: f64 = vector.iter().map(|&x| x * x).sum();

        // Return quotient or 0 for zero vector
        if denominator > 1e-12 {
            numerator / denominator
        } else {
            0.0 // Return 0 for zero vector (convention)
        }
    }

    /// Batch computation for multiple vectors (efficient for multiple queries)
    pub fn compute_rayleigh_quotients_batch(matrix: &CsMat<f64>, vectors: &[Vec<f64>]) -> Vec<f64> {
        vectors
            .iter()
            .map(|v| Self::compute_rayleigh_quotient_from_matrix(matrix, v))
            .collect()
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
