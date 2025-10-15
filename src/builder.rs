use std::sync::{Arc, Mutex};
// Add logging
use log::{debug, info, trace, warn};

use rayon::prelude::*;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::clustering::ClusteringHeuristic;
use crate::core::{ArrowSpace, TAUDEFAULT};
use crate::graph::{GraphFactory, GraphLaplacian};
use crate::reduction::{compute_jl_dimension, ImplicitProjection};
use crate::sampling::{InlineSampler, SamplerType};
use crate::taumode::TauMode;

#[derive(Clone, Debug)]
pub enum PairingStrategy {
    FastPair,            // 1-NN union via Smartcore FastPair
    Default,             // O(n^2) path
    CoverTreeKNN(usize), // k for k-NN build
}

pub struct ArrowSpaceBuilder {
    // Data
    //arrows: ArrowSpace,
    pub prebuilt_spectral: bool, // true if spectral laplacian has been computed

    // Lambda-graph parameters (the canonical path)
    // A good starting point is to choose parameters that keep the λ-graph broadly connected but sparse,
    // and set the kernel to behave nearly linearly for small gaps so it doesn't overpower cosine;
    // a practical default is: lambda_eps ≈ 1e-3, lambda_k ≈ 3–10, lambda_p = 2.0,
    // lambda_sigma = None (which defaults σ to eps)
    lambda_eps: f64,
    lambda_k: usize,
    lambda_topk: usize,
    lambda_p: f64,
    lambda_sigma: Option<f64>,
    normalise: bool,   // using normalisation is not relevant for taumode, do not use if are not sure
    sparsity_check: bool,

    // activate sampling, default false
    pub sampling: Option<SamplerType>,

    // Synthetic index configuration (used `with_synthesis`)
    synthesis: TauMode, // (tau_mode)

    /// Max clusters X (default: nfeatures; cap on centroids)
    cluster_max_clusters: Option<usize>,
    /// Squared L2 threshold for new cluster creation (default 1.0)
    cluster_radius: f64,
    clustering_seed: Option<u64>,
    deterministic_clustering: bool,

    // dimensionality reduction with random projection (dafault false)
    use_dims_reduction: bool,
    rp_eps: f64,
}

impl Default for ArrowSpaceBuilder {
    fn default() -> Self {
        debug!("Creating ArrowSpaceBuilder with default parameters");
        Self {
            // arrows: ArrowSpace::default(),
            prebuilt_spectral: false,

            // enable synthetic λ with α=0.7 and Median τ by default
            synthesis: TAUDEFAULT,

            // λ-graph parameters
            lambda_eps: 1e-3,
            lambda_k: 6,
            lambda_topk: 3,
            lambda_p: 2.0,
            lambda_sigma: None, // means σ := eps inside the builder
            normalise: false,
            sparsity_check: false,
            // sampling default
            sampling: Some(SamplerType::Simple(0.6)),
            // Clustering defaults
            cluster_max_clusters: None, // will be set to nfeatures at build time
            cluster_radius: 1.0,
            clustering_seed: None,
            deterministic_clustering: false,
            // dim reduction
            use_dims_reduction: false,
            rp_eps: 0.3,
        }
    }
}

impl ClusteringHeuristic for ArrowSpaceBuilder {}

impl ArrowSpaceBuilder {
    pub fn new() -> Self {
        info!("Initializing new ArrowSpaceBuilder");
        Self::default()
    }

    // -------------------- Lambda-graph configuration --------------------

    /// Use this to pass λτ-graph parameters. If not called, use defaults
    /// Configure the base λτ-graph to be built from the provided data matrix:
    /// - eps: threshold for |Δλ| on items
    /// - k: optional cap on neighbors per item
    /// - p: weight kernel exponent
    /// - sigma_override: optional scale σ for the kernel (default = eps)
    pub fn with_lambda_graph(
        mut self,
        eps: f64,
        k: usize,
        topk: usize,
        p: f64,
        sigma_override: Option<f64>,
    ) -> Self {
        info!(
            "Configuring lambda graph: eps={}, k={}, p={}, sigma={:?}",
            eps, k, p, sigma_override
        );
        debug!(
            "Lambda graph will use {} for normalization",
            if self.normalise {
                "normalized items"
            } else {
                "raw item magnitudes"
            }
        );

        self.lambda_eps = eps;
        self.lambda_k = k;
        self.lambda_topk = topk;
        self.lambda_p = p;
        self.lambda_sigma = sigma_override;

        self
    }

    // -------------------- Synthetic index --------------------

    /// Optional: override the default tau policy or tau for synthetic index.
    pub fn with_synthesis(mut self, tau_mode: TauMode) -> Self {
        info!("Configuring synthesis with tau mode: {:?}", tau_mode);
        self.synthesis = tau_mode;
        self
    }

    pub fn with_normalisation(mut self, normalise: bool) -> Self {
        info!("Setting normalization: {}", normalise);
        self.normalise = normalise;
        self
    }

    /// Optional define if building spectral matrix at building time
    /// This is expensive as requires twice laplacian computation
    /// use only on limited dataset for analysis, exploration and data QA
    pub fn with_spectral(mut self, compute_spectral: bool) -> Self {
        info!("Setting compute spectral: {}", compute_spectral);
        self.prebuilt_spectral = compute_spectral;
        self
    }

    pub fn with_sparsity_check(mut self, sparsity_check: bool) -> Self {
        info!("Setting sparsity check falg: {}", sparsity_check);
        self.sparsity_check = sparsity_check;
        self
    }

    pub fn with_inline_sampling(mut self, sampling: Option<SamplerType>) -> Self {
        info!("Configuring inline sampling: {}", sampling.as_ref().unwrap());
        self.sampling = sampling;
        self
    }

    pub fn with_dims_reduction(mut self, enable: bool, eps: Option<f64>) -> Self {
        self.use_dims_reduction = enable;
        self.rp_eps = eps.unwrap_or(0.5); // default JL tolerance
        self
    }

    /// Set a custom seed for deterministic clustering.
    /// Enable sequential (deterministic) clustering.
    /// This ensures reproducible results at the cost of parallelization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        info!("Setting custom clustering seed: {}", seed);
        self.clustering_seed = Some(seed);
        self.deterministic_clustering = true;
        self
    }

    /// Define the results number of k-neighbours from the
    ///  max number of neighbours connections (`GraphParams::k` -> result_k)
    /// Check if the passed cap_k is reasonable and define an euristics to
    ///  select a proper value.
    fn define_result_k(&mut self) {
        // normalise values for small values,
        // leave to the user for higher values
        if self.lambda_k <= 5 {
            self.lambda_topk = 3;
        } else if self.lambda_k < 10 {
            self.lambda_topk = 4;
        };
    }

    // -------------------- Build --------------------

    /// Build the ArrowSpace and the selected Laplacian (if any).
    ///
    /// Priority order for graph selection:
    ///   1) prebuilt Laplacian (if provided)
    ///   2) hypergraph clique/normalized (if provided)
    ///   3) fallback: λτ-graph-from-data (with_lambda_graph config or defaults)
    ///
    /// Behavior:
    /// - If fallback (#3) is selected, synthetic lambdas are always computed using TauMode::Median
    ///   unless with_synthesis was called, in which case the provided tau_mode and alpha are used.
    /// - If prebuilt or hypergraph graph is selected, standard Rayleigh lambdas are computed unless
    ///   with_synthesis was called, in which case synthetic lambdas are computed on that graph.
    pub fn build(mut self, rows: Vec<Vec<f64>>) -> (ArrowSpace, GraphLaplacian) {
        let n_items = rows.len();
        let n_features = rows.first().map(|r| r.len()).unwrap_or(0);

        // set baseline for topk
        self.define_result_k();

        info!(
            "Building ArrowSpace from {} items with {} features",
            n_items, n_features
        );
        debug!(
            "Build configuration: eps={}, k={}, p={}, sigma={:?}, normalise={}, synthesis={:?}",
            self.lambda_eps,
            self.lambda_k,
            self.lambda_p,
            self.lambda_sigma,
            self.normalise,
            self.synthesis
        );

        // 1) Create starting `ArrowSpace`
        trace!("Creating ArrowSpace from items");
        let mut aspace = ArrowSpace::new(rows.clone(), self.synthesis);
        debug!(
            "ArrowSpace created with {} items and {} features",
            n_items, n_features
        );

        // Sampler switch
        let sampler: Arc<Mutex<dyn InlineSampler>> = match self.sampling {
            Some(SamplerType::Simple(r)) => Arc::new(Mutex::new(SamplerType::new_simple(r))),
            Some(SamplerType::DensityAdaptive(r)) => Arc::new(Mutex::new(SamplerType::new_density_adaptive(r))),
            None => Arc::new(Mutex::new(SamplerType::new_simple(0.6))),
        };

        // ---- Compute optimal K automatically ----
        let (_, _, intrinsic_dim) = {
            info!("Auto-computing optimal clustering parameters");
            let params = self.compute_optimal_k(&rows, n_items, n_features, self.clustering_seed);
            info!(
                "Auto K={}, radius={:.6}, intrinsic_dim={}",
                params.0, params.1, params.2
            );
            self.cluster_max_clusters = Some(params.0);
            self.cluster_radius = params.1;
            params
        };

        info!(
            "Clustering: {} centroids, radius= {}, intrinsic_dim ≈ {}",
            self.cluster_max_clusters.unwrap(),
            self.cluster_radius,
            intrinsic_dim
        );

        // Run incremental clustering
        // include inline sampling if flag is on
        let (clustered_dm, assignments, sizes) = self.run_incremental_clustering_with_sampling(
            &rows,
            n_features,
            self.cluster_max_clusters.unwrap(),
            self.cluster_radius,
            sampler,
        );

        // Store clustering results in ArrowSpace
        aspace.n_clusters = clustered_dm.shape().0;
        aspace.cluster_assignments = assignments;
        aspace.cluster_sizes = sizes;
        aspace.cluster_radius = self.cluster_radius;

        info!(
            "Clustering complete: {} centroids, {} items assigned",
            aspace.cluster_sizes.len(),
            aspace
                .cluster_assignments
                .iter()
                .filter(|x| x.is_some())
                .count()
        );

        let (laplacian_input, reduced_dim) = if self.use_dims_reduction && n_features > 64 {
            let n_centroids = clustered_dm.shape().0;

            // Compute target dimension using JL bound
            let jl_dim = compute_jl_dimension(n_centroids, self.rp_eps);
            let target_dim = jl_dim.min(n_features / 2);

            if target_dim < n_features {
                info!(
                    "Applying random projection: {} centroids × {} features -> {} features (ε={:.2})",
                    n_centroids, n_features, target_dim, self.rp_eps
                );

                // Create implicit projection
                let implicit_proj = ImplicitProjection::new(n_features, target_dim);

                // Project centroids using the implicit projection
                let projected = crate::reduction::project_matrix(&clustered_dm, &implicit_proj);

                let compression = n_features as f64 / target_dim as f64;
                info!(
                    "Projection complete: {:.1}x compression, projection stored as seed (8 bytes)",
                    compression
                );

                // Store the projection for query-time use
                aspace.projection_matrix = Some(implicit_proj);
                aspace.reduced_dim = Some(target_dim);

                (projected, target_dim)
            } else {
                debug!(
                    "Target dimension {} >= original {}, skipping projection",
                    target_dim, n_features
                );
                (clustered_dm.clone(), n_features)
            }
        } else {
            debug!("Random projection disabled or dimension too small");
            (clustered_dm.clone(), n_features)
        };

        info!(
            "Building Laplacian matrix on {} × {} input",
            laplacian_input.shape().0,
            reduced_dim
        );

        // Resolve λτ-graph params with conservative defaults
        info!("Building Laplacian matrix with configured parameters");

        // 3) Compute synthetic indices on resulting graph
        let gl = GraphFactory::build_laplacian_matrix_from_k_cluster(
            laplacian_input,
            self.lambda_eps,
            self.lambda_k,
            self.lambda_topk,
            self.lambda_p,
            self.lambda_sigma,
            self.normalise,
            self.sparsity_check,
            n_items,
        );
        debug!("Laplacian matrix built successfully");

        // Branch: if spectral L_2 laplacian is required, compute
        // if aspace.signals is not set, gl.matrix will be used
        if self.prebuilt_spectral {
            // Compute signals FxF laplacian
            trace!("Building spectral Laplacian for ArrowSpace");
            aspace = GraphFactory::build_spectral_laplacian(aspace, &gl);
            debug!(
                "Spectral Laplacian built with signals shape: {:?}",
                aspace.signals.shape()
            );
        }

        // Compute taumode lambdas
        info!(
            "Computing taumode lambdas with synthesis: {:?}",
            self.synthesis
        );
        TauMode::compute_taumode_lambdas(&mut aspace, &gl, self.synthesis);

        let lambda_stats = {
            let lambdas = aspace.lambdas();
            let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max: f64 = lambdas.iter().fold(0.0, |a, &b| a.max(b));
            let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            (min, max, mean)
        };

        debug!(
            "Lambda computation completed - min: {:.6}, max: {:.6}, mean: {:.6}",
            lambda_stats.0, lambda_stats.1, lambda_stats.2
        );

        info!("ArrowSpace build completed successfully");
        (aspace, gl)
    }

    // Clustering and Reduction: methods used to prepare the raw data before
    //  computing Laplacian.

    /// Scans rows linearly, assigns to nearest centroid if within radius,
    /// else creates a new cluster up to max_clusters. Outliers beyond cap are skipped.
    /// Returns (centroids_matrix, assignments, sizes).
    fn run_incremental_clustering_with_sampling(
        &self,
        rows: &[Vec<f64>],
        nfeatures: usize,
        max_clusters: usize,
        radius: f64,
        sampler: Arc<Mutex<dyn InlineSampler>>,
    ) -> (DenseMatrix<f64>, Vec<Option<usize>>, Vec<usize>) {
        let nrows = rows.len();

        info!("Starting incremental clustering with inline sampling");
        debug!(
            "Parameters: max_clusters={}, radius={:.4}",
            max_clusters, radius
        );

        // Shared clustering state
        let centroids = Mutex::new(Vec::<Vec<f64>>::new());
        let counts = Mutex::new(Vec::<usize>::new());
        let assignments = Mutex::new(vec![None; nrows]);

        let process_row = |row_idx: usize| {
            let row = &rows[row_idx];

            // Snapshot current centroids for sampling decision
            let cent_snap: Vec<Vec<f64>> = {
                let c = centroids.lock().unwrap();
                c.clone()
            };

            // Compute nearest centroid distance for sampling decision
            let (_best_idx, best_dist_sq) = if cent_snap.is_empty() {
                (0, f64::INFINITY)
            } else {
                Self::nearest_centroid(row, &cent_snap)
            };


            if self.sampling.is_some() {
                // Implement sampling
                let current_cent_count = cent_snap.len();

                let mut smp = sampler.lock().unwrap();
                let keep = smp.should_keep(row, best_dist_sq, current_cent_count, max_clusters);

                // if not keep, go to the next row
                if !keep { return; }
            }

            // Clustering logic
            if cent_snap.is_empty() {
                // First centroid
                let mut c = centroids.lock().unwrap();
                let mut k = counts.lock().unwrap();
                let mut a = assignments.lock().unwrap();
                c.push(row.clone());
                k.push(1);
                a[row_idx] = Some(0);
                return;
            }

            let current_len = {
                let c = centroids.lock().unwrap();
                c.len()
            };

            // Create new cluster or assign to nearest
            if current_len < max_clusters && best_dist_sq > radius {
                let mut c = centroids.lock().unwrap();
                let mut k = counts.lock().unwrap();
                let mut a = assignments.lock().unwrap();

                let new_idx = c.len();
                c.push(row.clone());
                k.push(1);
                a[row_idx] = Some(new_idx);
            } else {
                if current_len >= max_clusters && best_dist_sq > radius {
                    // Drop outlier
                    trace!(
                        "Dropping outlier row {} with dist²={:.6}",
                        row_idx,
                        best_dist_sq
                    );
                    return;
                }

                // Assign to nearest centroid with inline update
                let mut c = centroids.lock().unwrap();
                let mut k = counts.lock().unwrap();
                let mut a = assignments.lock().unwrap();

                // Recompute nearest with current centroids under lock
                let (best, _best_d) = Self::nearest_centroid(row, &c);

                let k_old = k[best] as f64;
                let k_new = k_old + 1.0;
                for j in 0..nfeatures {
                    let mu = c[best][j];
                    let delta = row[j] - mu;
                    c[best][j] = mu + delta / k_new;
                }
                k[best] += 1;
                a[row_idx] = Some(best);
            }
        };

        // Process rows in parallel (sampling decisions are thread-safe via state snapshots)
        if self.deterministic_clustering {
            (0..nrows).for_each(process_row);
        } else {
            (0..nrows).into_par_iter().for_each(process_row);
        }

        let final_centroids = centroids.into_inner().unwrap();
        let final_counts = counts.into_inner().unwrap();
        let final_assignments = assignments.into_inner().unwrap();

        // Build output matrix
        let x_out = &final_centroids.len().max(1);
        let mut flat = Vec::<f64>::with_capacity(x_out * nfeatures);
        for c in &final_centroids {
            flat.extend_from_slice(c);
        }

        let centroids_dm: DenseMatrix<f64> = if *x_out > 0 && !final_centroids.is_empty() {
            debug!(
                "Centroids:  {:?}\n : nitems->{} nfeatures->{}",
                flat, x_out, nfeatures
            );
            let dm = DenseMatrix::from_iterator(flat.iter().map(|x| *x), *x_out, nfeatures, 1);
            dm
        } else {
            warn!("No clusters created; returning zero matrix");
            let inline_sampling = self.sampling.as_ref().unwrap();
            panic!("No clusters created from data, sampling: {}", inline_sampling);
            #[allow(unreachable_code)]
            DenseMatrix::from_2d_vec(&vec![vec![0.0 as f64; nfeatures]; *x_out]).unwrap()
        };

        if self.sampling.is_some() {
            let smp = sampler.lock().unwrap();
            let (sampled, discarded) = smp.get_stats();
            let sampling_ratio = sampled as f64 / nrows as f64;

            debug!(
                "Inline sampling complete: {} kept ({:.2}%), {} discarded",
                sampled,
                sampling_ratio * 100.0,
                discarded
            );
            debug!(
                "Clustering produced {} centroids from {} rows ({}% sampling)",
                final_centroids.len(),
                nrows,
                sampling_ratio * 100.0
            );
            #[cfg(not(test))]
            assert!(
                sampling_ratio > 0.325 && sampling_ratio < 0.89,
                "sampling_rate not in the interval 0.325..0.875 but {sampling_ratio}"
            );
        } else {
            debug!(
                "Clustering produced {} centroids from {} rows (100% sampling)",
                final_centroids.len(),
                nrows
            );
        }

        (centroids_dm, final_assignments, final_counts)
    }

    /// Linear-scan nearest centroid helper: returns (index, squared_distance).
    fn nearest_centroid(row: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_dist2 = f64::INFINITY;
        for (i, c) in centroids.iter().enumerate() {
            let mut d2 = 0.0;
            for (a, b) in row.iter().zip(c.iter()) {
                let diff = a - b;
                d2 += diff * diff;
            }
            if d2 < best_dist2 {
                best_dist2 = d2;
                best_idx = i;
            }
        }
        (best_idx, best_dist2)
    }
}
