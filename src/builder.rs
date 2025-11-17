//! ArrowSpace builder and pipelines (Eigen / Energy).
//!
//! This module configures and builds `ArrowSpace` instances and their associated
//! Laplacians from raw item vectors. It supports multiple pipelines:
//! - EigenMaps (`build` and `build_for_persistence` with `Pipeline::Eigen`)
//! - EnergyMaps (`build_energy` and `build_for_persistence` with `Pipeline::Energy`)

use log::{debug, info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::str::FromStr;

use crate::clustering::ClusteringHeuristic;
use crate::core::{ArrowSpace, TAUDEFAULT};
use crate::eigenmaps::{ClusteredOutput, EigenMaps};
use crate::energymaps::EnergyMaps;
use crate::energymaps::{EnergyMapsBuilder, EnergyParams};
use crate::graph::GraphLaplacian;
use crate::sampling::SamplerType;
use crate::taumode::TauMode;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Pipeline {
    Eigen,
    Energy,
    Default,
}

impl FromStr for Pipeline {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "eigen" => Ok(Pipeline::Eigen),
            "energy" => Ok(Pipeline::Energy),
            "default" => Ok(Pipeline::Default),
            _ => Err(()),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct ArrowSpaceBuilder {
    pub(crate) n_items_original: usize,
    pub prebuilt_spectral: bool, // true if spectral laplacian has been computed

    // Lambda-graph parameters (the canonical path)
    // A good starting point is to choose parameters that keep the λ-graph broadly connected but sparse,
    // and set the kernel to behave nearly linearly for small gaps so it doesn't overpower cosine;
    // a practical default is: lambda_eps ≈ 1e-3, lambda_k ≈ 3–10, lambda_p = 2.0,
    // lambda_sigma = None (which defaults σ to eps)
    pub(crate) lambda_eps: f64,
    pub(crate) lambda_k: usize,
    pub(crate) lambda_topk: usize,
    pub(crate) lambda_p: f64,
    pub(crate) lambda_sigma: Option<f64>,
    pub(crate) normalise: bool, // using normalisation is not relevant for taumode, do not use if are not sure
    pub(crate) sparsity_check: bool,

    // activate sampling, default false
    pub sampling: Option<SamplerType>,

    // Synthetic index configuration (used `with_synthesis`)
    pub synthesis: TauMode, // (tau_mode)

    /// Max clusters X (default: nfeatures; cap on centroids)
    pub(crate) cluster_max_clusters: Option<usize>,
    /// Squared L2 threshold for new cluster creation (default 1.0)
    pub(crate) cluster_radius: f64,
    pub(crate) clustering_seed: Option<u64>,
    pub(crate) deterministic_clustering: bool,

    // dimensionality reduction with random projection (dafault false)
    pub(crate) use_dims_reduction: bool,
    pub(crate) extra_dims_reduction: bool,
    pub(crate) rp_eps: f64,

    // persistence directory
    persistence: Option<(String, std::path::PathBuf)>,
}

impl Default for ArrowSpaceBuilder {
    fn default() -> Self {
        debug!("Creating ArrowSpaceBuilder with default parameters");
        Self {
            n_items_original: 0,
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
            extra_dims_reduction: false,
            rp_eps: 0.3,
            // persistence directory
            persistence: None,
        }
    }
}

impl ClusteringHeuristic for ArrowSpaceBuilder {}

impl ArrowSpaceBuilder {
    pub fn new() -> Self {
        info!("Initializing new ArrowSpaceBuilder");
        Self::default()
    }

    /// copy all the static parameters to generate a similar builder from the original
    pub fn copy_params(&self) -> Self {
        let mut result = Self::default();
        result.prebuilt_spectral = self.prebuilt_spectral;
        result.synthesis = self.synthesis;
        result.lambda_eps = self.lambda_eps;
        result.lambda_k = self.lambda_k;
        result.lambda_topk = self.lambda_topk;
        result.lambda_p = self.lambda_p;
        result.lambda_sigma = self.lambda_sigma;
        result.normalise = self.normalise;
        result.sparsity_check = self.sparsity_check;
        result.sampling = self.sampling.clone();
        result.use_dims_reduction = self.use_dims_reduction;
        result.rp_eps = self.rp_eps;
        result.persistence = self.persistence.clone();
        result
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
            "Configuring lambda graph: eps={:?}, k={}, p={}, sigma={:?}",
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
        warn!(
            "with_spectral is an experimental feature, results may be unprecise. Keep the default to false"
        );
        self.prebuilt_spectral = compute_spectral;
        self
    }

    pub fn with_sparsity_check(mut self, sparsity_check: bool) -> Self {
        info!("Setting sparsity check flag: {}", sparsity_check);
        self.sparsity_check = sparsity_check;
        self
    }

    pub fn with_inline_sampling(mut self, sampling: Option<SamplerType>) -> Self {
        let value = if sampling.as_ref().is_none() {
            "None".to_string()
        } else {
            format!("{}", sampling.as_ref().unwrap())
        };
        info!("Configuring inline sampling: {}", value);
        self.sampling = sampling;
        self
    }

    /// Enable dimensionality reduction in clustering
    pub fn with_dims_reduction(mut self, enable: bool, eps: Option<f64>) -> Self {
        self.use_dims_reduction = enable;
        self.rp_eps = eps.unwrap_or(0.5); // default JL tolerance
        self
    }

    /// Enable extra-dimensionality reduction after clustering (energymaps only, optional)
    pub fn with_extra_dims_reduction(mut self, enable: bool) -> Self {
        assert!(
            self.use_dims_reduction == true,
            "extra dims reduction needs base reduction"
        );
        self.extra_dims_reduction = enable;
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

    /// Files are saved in Parquet format with Snappy compression for efficiency.
    ///
    /// # Arguments
    /// * `path` - Directory path where artifacts will be saved
    ///
    /// # Example
    /// ```ignore
    /// use arrowspace::builder::ArrowSpaceBuilder;
    ///
    /// let builder = ArrowSpaceBuilder::new()
    ///     .with_lambda_graph(0.5, 5, 3, 2.0, None)
    ///     .with_persistence("./checkpoints");
    /// ```
    ///
    /// # Note
    /// This method is only available when the `storage` feature is enabled.
    #[cfg(feature = "storage")]
    pub fn with_persistence(mut self, path: impl AsRef<std::path::Path>, name: String) -> Self {
        let path_buf: std::path::PathBuf = path.as_ref().to_path_buf();
        info!("Enabling persistence at: {}", path_buf.display());
        self.persistence = Some((name, path_buf));
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
        self.n_items_original = n_items;
        let n_features = rows.first().map(|r| r.len()).unwrap_or(0);
        let start = std::time::Instant::now();

        // set baseline for topk
        self.define_result_k();

        info!(
            "Building ArrowSpace from {} items with {} features",
            n_items, n_features
        );
        debug!(
            "Build configuration: eps={:?}, k={}, p={}, sigma={:?}, normalise={}, synthesis={:?}",
            self.lambda_eps,
            self.lambda_k,
            self.lambda_p,
            self.lambda_sigma,
            self.normalise,
            self.synthesis
        );

        // Save raw input if persistence is enabled
        #[cfg(feature = "storage")]
        {
            if let Some((ref name, ref path)) = self.persistence {
                use crate::storage::StorageError;
                use crate::storage::parquet::save_dense_matrix_with_builder;

                // Create temporary ArrowSpace for saving raw data
                let temp_aspace = ArrowSpace::new(rows.clone(), self.synthesis);

                let saved: Result<(), StorageError> = save_dense_matrix_with_builder(
                    &temp_aspace.data,
                    path.clone(),
                    &format!("{}-raw_input", name),
                    Some(&self),
                );
                match saved {
                    Ok(_) => debug!("raw-input saved"),
                    Err(StorageError::Parquet(err)) => {
                        panic!("saving failed for raw-input {}", err)
                    }
                    _ => panic!("Error with {:?}", saved),
                };
            }
        }

        // ============================================================
        // Stage 1: Clustering with sampling and optional projection
        // ============================================================
        let ClusteredOutput {
            mut aspace,
            centroids,
            n_items: _n_items,
            n_features: _n_features,
            ..
        } = ArrowSpace::start_clustering(&mut self, rows.clone());

        // Save clustered centroids if persistence is enabled
        #[cfg(feature = "storage")]
        {
            if let Some((ref name, ref path)) = self.persistence {
                use crate::storage::StorageError;
                use crate::storage::parquet::save_dense_matrix_with_builder;

                let saved: Result<(), StorageError> = save_dense_matrix_with_builder(
                    &centroids,
                    path.clone(),
                    &format!("{}-clustered-dm", name),
                    Some(&self),
                );
                match saved {
                    Ok(_) => debug!("clustered_dm saved"),
                    Err(StorageError::Parquet(err)) => {
                        panic!("saving failed for clustered_dm {}", err)
                    }
                    _ => panic!("Error with {:?}", saved),
                };
            }
        }

        // Save laplacian input (projected centroids) if persistence is enabled
        #[cfg(feature = "storage")]
        {
            if let Some((ref name, ref path)) = self.persistence {
                use crate::storage::StorageError;
                use crate::storage::parquet::save_dense_matrix_with_builder;

                let saved: Result<(), StorageError> = save_dense_matrix_with_builder(
                    &centroids,
                    path.clone(),
                    &format!("{}-laplacian-input", name),
                    Some(&self),
                );
                match saved {
                    Ok(_) => debug!("laplacian_input saved"),
                    Err(StorageError::Parquet(err)) => {
                        panic!("saving failed for laplacian_input {}", err)
                    }
                    _ => panic!("Error with {:?}", saved),
                };
            }
        }

        // ============================================================
        // Stage 2: Build item-graph Laplacian
        // ============================================================
        let gl = aspace.eigenmaps(&self, &centroids, n_items);

        // Save graph Laplacian matrix if persistence is enabled
        #[cfg(feature = "storage")]
        {
            if let Some((ref name, ref path)) = self.persistence {
                use crate::storage::StorageError;
                use crate::storage::parquet::save_sparse_matrix_with_builder;

                let saved: Result<(), StorageError> = save_sparse_matrix_with_builder(
                    &gl.matrix,
                    path.clone(),
                    &format!("{}-gl-matrix", name),
                    Some(&self),
                );
                match saved {
                    Ok(_) => debug!("gl.matrix saved"),
                    Err(StorageError::Parquet(err)) => {
                        panic!("saving failed for gl.matrix {}", err)
                    }
                    _ => panic!("Error with {:?}", saved),
                };
            }
        }

        // ============================================================
        // Stage 3: Optional spectral feature Laplacian (F×F) if prebuilt_spectral is true
        // ============================================================
        if self.prebuilt_spectral {
            // Save spectral signals if persistence is enabled
            #[cfg(feature = "storage")]
            {
                if let Some((ref name, ref path)) = self.persistence {
                    use crate::storage::StorageError;
                    use crate::storage::parquet::save_sparse_matrix_with_builder;

                    let saved: Result<(), StorageError> = save_sparse_matrix_with_builder(
                        &aspace.signals,
                        path.clone(),
                        &format!("{}-aspace-signals", name),
                        Some(&self),
                    );
                    match saved {
                        Ok(_) => debug!("aspace.signals saved"),
                        Err(StorageError::Parquet(err)) => {
                            panic!("saving failed for aspace.signals {}", err)
                        }
                        _ => panic!("Error with {:?}", saved),
                    };
                }
            }
        }

        // ============================================================
        // Stage 4: Compute taumode lambdas
        // ============================================================
        info!(
            "Computing taumode lambdas with synthesis: {:?}",
            self.synthesis
        );
        aspace.compute_taumode(&gl);
        // create the sorted index
        aspace.build_lambdas_sorted();

        // Save lambdas if persistence is enabled
        #[cfg(feature = "storage")]
        {
            if let Some((ref name, ref path)) = self.persistence {
                use crate::storage::StorageError;
                use crate::storage::parquet::save_lambda_with_builder;

                let saved: Result<(), StorageError> = save_lambda_with_builder(
                    &aspace.lambdas,
                    path.clone(),
                    &format!("{}-lambdas", name),
                    Some(&self),
                );
                match saved {
                    Ok(_) => debug!("{}-lambdas saved", name),
                    Err(StorageError::Parquet(err)) => {
                        panic!("saving failed for {}-lambdas {}", name, err)
                    }
                    _ => panic!("Error with {:?}", saved),
                };
            }
        }

        let lambda_stats = {
            let lambdas = aspace.lambdas();
            let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max: f64 = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            (min, max, mean)
        };

        debug!(
            "Lambda computation completed - min: {:.6}, max: {:.6}, mean: {:.6}",
            lambda_stats.0, lambda_stats.1, lambda_stats.2
        );

        info!(
            "Total ArrowSpaceBuilder construction time: {:?}",
            start.elapsed()
        );
        debug!("ArrowSpaceBuilder configuration: {}", self);
        info!("ArrowSpace build completed successfully");

        (aspace, gl)
    }

    /// Same as build but passing a `DenseMatrix` instead of a `Vec<Vec<..>>`
    pub fn build_for_persistence(
        mut self,
        rows: DenseMatrix<f64>,
        pipeline: &str,
        energy_params: Option<EnergyParams>,
    ) -> (ArrowSpace, GraphLaplacian) {
        let n_items = rows.shape().0;
        self.n_items_original = n_items;
        let n_features = rows.shape().1;
        let start = std::time::Instant::now();

        // set baseline for topk
        self.define_result_k();

        info!(
            "Building ArrowSpace from {} items with {} features",
            n_items, n_features
        );
        debug!(
            "Build configuration: eps={:?}, k={}, p={}, sigma={:?}, normalise={}, synthesis={:?}",
            self.lambda_eps,
            self.lambda_k,
            self.lambda_p,
            self.lambda_sigma,
            self.normalise,
            self.synthesis
        );

        let pipeline = match pipeline.parse::<Pipeline>() {
            Ok(p) => p,
            Err(_) => panic!("Invalid pipeline value: {}", pipeline),
        };

        match pipeline {
            Pipeline::Eigen => {
                // ============================================================
                // Stage 1: Clustering with sampling and optional projection
                // ============================================================
                let ClusteredOutput {
                    mut aspace,
                    centroids,
                    n_items: _n_items,
                    n_features: _n_features,
                    ..
                } = ArrowSpace::start_clustering_dense(&mut self, rows.clone());

                // ============================================================
                // Stage 2: Build item-graph Laplacian
                // ============================================================
                let gl = aspace.eigenmaps(&self, &centroids, n_items);

                // ============================================================
                // Stage 4: Compute taumode lambdas
                // ============================================================
                info!(
                    "Computing taumode lambdas with synthesis: {:?}",
                    self.synthesis
                );
                aspace.compute_taumode(&gl);
                // create the sorted index
                aspace.build_lambdas_sorted();

                let lambda_stats = {
                    let lambdas = aspace.lambdas();
                    let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max: f64 = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
                    (min, max, mean)
                };

                debug!(
                    "Lambda computation completed - min: {:.6}, max: {:.6}, mean: {:.6}",
                    lambda_stats.0, lambda_stats.1, lambda_stats.2
                );

                info!(
                    "Total ArrowSpaceBuilder construction time: {:?}",
                    start.elapsed()
                );
                debug!("ArrowSpaceBuilder configuration: {}", self);
                info!("ArrowSpace build completed successfully");

                (aspace, gl)
            }
            Pipeline::Energy | Pipeline::Default => {
                assert!(
                    self.use_dims_reduction == true,
                    "When using energy pipeline, dim reduction is needed"
                );
                assert!(
                    energy_params.is_some(),
                    "if using energy pipeline, energy_params should be some"
                );
                if self.prebuilt_spectral == true {
                    panic!(
                        "Spectral mode not compatible with energy pipeline, please do not enable for energy search"
                    );
                }
                self.n_items_original = rows.shape().0;

                // ============================================================
                // Stage 1: Clustering with sampling and optional projection
                // ============================================================n
                let ClusteredOutput {
                    mut aspace,
                    mut centroids,
                    ..
                } = ArrowSpace::start_clustering_dense(&mut self, rows);

                // check that projection has been applied or not
                if aspace.projection_matrix.is_some() && aspace.nfeatures > 64 {
                    assert_ne!(
                        centroids.shape().1,
                        aspace.nfeatures,
                        "aspace is now projected"
                    );
                } else {
                    assert_eq!(
                        centroids.shape().1,
                        aspace.nfeatures,
                        "aspace has not been projected"
                    );
                }

                // Step 2: Optional optical compression on centroids
                if let Some(tokens) = energy_params.as_ref().unwrap().optical_tokens {
                    // mutate centroids with compression
                    centroids = ArrowSpace::optical_compress_centroids(
                        &centroids,
                        tokens,
                        energy_params.as_ref().unwrap().trim_quantile,
                    );
                }

                // Step 3: Bootstrap Laplacian on centroids
                let l0: GraphLaplacian =
                    ArrowSpace::bootstrap_centroid_laplacian(&centroids, &self);

                assert_eq!(centroids.shape().0, l0.nnodes, "l0 is still non-projected");

                // Step 4: Diffuse and split to create sub_centroids
                let sub_centroids: DenseMatrix<f64> = ArrowSpace::diffuse_and_split_subcentroids(
                    &centroids,
                    &l0,
                    &energy_params.as_ref().unwrap(),
                );

                assert_eq!(sub_centroids.shape().1, centroids.shape().1);

                // Step 5: apply EXTRA JL dimensionality reduction (optional `with_extra_dims_reduction`)
                let (n_subcentroids, current_features) = sub_centroids.shape();
                let (sub_centroids, reduced_dim) = if self.use_dims_reduction
                    && self.extra_dims_reduction
                    && current_features > 10001
                {
                    info!("Apply EXTRA JL dimensionality reduction");
                    use crate::reduction::{
                        ImplicitProjection, compute_jl_dimension, project_matrix,
                    };

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
                    (sub_centroids, current_features)
                };

                info!(
                    "Energy graph: {:?} centroids → {:?} sub_centroids (reduced_dim={})",
                    centroids.shape(),
                    sub_centroids.shape(),
                    reduced_dim
                );

                // Step 6: Build Laplacian on sub_centroids using energy dispersion
                let (gl_energy, _, _) =
                    self.build_energy_laplacian(&sub_centroids, &energy_params.as_ref().unwrap());

                assert_eq!(
                    gl_energy.shape().1,
                    sub_centroids.shape().1,
                    "Graph cols ({}) must match sub_centroids features ({})",
                    gl_energy.shape().1,
                    sub_centroids.shape().1
                );

                // Step 7: Compute lambdas on sub_centroids ONLY
                // Store sub_centroids for query mapping
                aspace.sub_centroids = Some(sub_centroids.clone());
                let sub_centroids_shape = sub_centroids.shape();

                // Create a sub-ArrowSpace to match gl_energy
                let mut subcentroid_space =
                    ArrowSpace::subcentroids_from_dense_matrix(sub_centroids.clone());
                subcentroid_space.taumode = aspace.taumode;
                subcentroid_space.projection_matrix = aspace.projection_matrix.clone();
                subcentroid_space.reduced_dim = aspace.reduced_dim.clone();
                // safeguard to clear signals
                subcentroid_space.signals = sprs::CsMat::empty(sprs::CSR, 0);

                assert_eq!(
                    subcentroid_space.nfeatures,
                    gl_energy.shape().1,
                    "Subcentroid count must match energy graph dimensions"
                );

                info!(
                    "Computing lambdas on {} sub_centroids...",
                    subcentroid_space.nitems
                );

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
                    aspace.nitems, sub_centroids_shape
                );

                // Step 8: Compute taumode
                // epsilon for considering lambdas "tied"
                let epsilon: f64 = 1e-11;

                // Parallel assignment using taumode distance
                let results: Vec<(usize, f64, f64)> = (0..aspace.nitems)
                    .into_par_iter()
                    .map(|i| {
                        let item = aspace.get_item(i);

                        // project only if unprojected
                        let projected_item = if aspace.projection_matrix.is_some()
                            && item.item.len() == aspace.projection_matrix.as_ref().unwrap().original_dim
                        {
                            aspace.project_query(&item.item)
                        } else if aspace.projection_matrix.is_none()
                            || item.item.len() == aspace.projection_matrix.as_ref().unwrap().reduced_dim
                        {
                            item.item.to_owned()
                        } else {
                            panic!(
                                "Check the projection pipeline, item seems neither projected nor unprojected. \n\
                                   input item len: {:?} \
                                   projection matrix is set: {} \
                                   projection matrix original dims: {} \
                                   projection matrix reduced dims: {}",
                                item.item.len(),
                                aspace.projection_matrix.as_ref().is_some(),
                                aspace.projection_matrix.as_ref().unwrap().original_dim,
                                aspace.projection_matrix.as_ref().unwrap().reduced_dim
                            )
                        };

                        // 1) Compute item's synthetic lambda via taumode
                        let item_lambda = aspace.prepare_query_item(&projected_item, &gl_energy);

                        // 2) Find nearest subcentroid by linear synthetic distance in lambda-space
                        //    distance := |lambda_item - lambda_subcentroid|
                        let mut best_idx = 0usize;
                        let mut best_dist = f64::INFINITY;

                        for sc_idx in 0..sub_centroids.shape().0 {
                            let sc_lambda = subcentroid_space.lambdas[sc_idx];
                            let lambda_dist = (item_lambda - sc_lambda).abs();
                            if lambda_dist < best_dist {
                                best_dist = lambda_dist;
                                best_idx = sc_idx;
                            }
                        }

                        // 3) Tie-break with cosine on projected space if multiple subcentroids tie within epsilon
                        //    Collect all candidates at the same minimal lambda distance within epsilon
                        let mut candidates: Vec<usize> = Vec::new();
                        for sc_idx in 0..sub_centroids.shape().0 {
                            let sc_lambda = subcentroid_space.lambdas[sc_idx];
                            let lambda_dist = (item_lambda - sc_lambda).abs();
                            if (lambda_dist - best_dist).abs() < epsilon {
                                candidates.push(sc_idx);
                            }
                        }

                        if candidates.len() > 1 {
                            let item_norm_proj: f64 =
                                projected_item.iter().map(|x| x * x).sum::<f64>().sqrt();
                            // fallback to zero-safe cosine
                            let mut best_cos = f64::NEG_INFINITY;
                            let mut best_sc = best_idx;

                            for sc_idx in candidates {
                                // read centroid row into a temporary slice or iterator
                                let mut dot = 0.0f64;
                                let mut cent_norm_sq = 0.0f64;
                                for (a, b) in projected_item
                                    .iter()
                                    .zip(sub_centroids.get_row(sc_idx).iterator(0))
                                {
                                    dot += a * b;
                                    cent_norm_sq += b * b;
                                }
                                let cent_norm = cent_norm_sq.sqrt();
                                let cosine = if item_norm_proj > 0.0 && cent_norm > 0.0 {
                                    dot / (item_norm_proj * cent_norm)
                                } else {
                                    0.0
                                };

                                if cosine > best_cos {
                                    best_cos = cosine;
                                    best_sc = sc_idx;
                                }
                            }

                            best_idx = best_sc;
                        }

                        // 4) Compute norm on ORIGINAL item for cosine metadata consumers
                        let norm: f64 = item.item.iter().map(|x| x * x).sum::<f64>().sqrt();

                        // Return the chosen centroid index, store that centroid's lambda, and the item norm
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

                aspace.build_lambdas_sorted();

                info!(
                    "Item λ assigned: min={:.6}, max={:.6}, mean={:.6}",
                    aspace.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    aspace.lambdas.iter().fold(0.0_f64, |a, &b| a.max(b)),
                    aspace.lambdas.iter().sum::<f64>() / aspace.nitems as f64
                );

                debug!(
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
        }
    }
}

impl fmt::Display for ArrowSpaceBuilder {
    /// Format ArrowSpaceBuilder as comma-separated key=value pairs (cookie-style).
    ///
    /// Output format: "key1=value1, key2=value2, ..."
    /// This format can be parsed into a HashMap<String, String> using cookie parsers
    /// or simple string splitting.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = ArrowSpaceBuilder::new()
    ///     .with_synthesis(TauMode::Median);
    ///
    /// let config_string = builder.to_string();
    /// println!("{}", config_string);
    ///
    /// // Parse back to HashMap
    /// let config_map: HashMap<String, String> = parse_builder_config(&config_string);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "prebuilt_spectral={}, \
             lambda_eps={}, \
             lambda_k={}, \
             lambda_topk={}, \
             lambda_p={}, \
             lambda_sigma={}, \
             normalise={}, \
             sparsity_check={}, \
             sampling={}, \
             synthesis={:?}, \
             cluster_max_clusters={}, \
             cluster_radius={}, \
             clustering_seed={}, \
             deterministic_clustering={}, \
             use_dims_reduction={}, \
             rp_eps={}, \
             persistence={}",
            self.prebuilt_spectral,
            self.lambda_eps,
            self.lambda_k,
            self.lambda_topk,
            self.lambda_p,
            self.lambda_sigma
                .map_or("None".to_string(), |v| v.to_string()),
            self.normalise,
            self.sparsity_check,
            self.sampling
                .as_ref()
                .map_or("None".to_string(), |s| s.to_string()),
            self.synthesis,
            self.cluster_max_clusters
                .map_or("None".to_string(), |v| v.to_string()),
            self.cluster_radius,
            self.clustering_seed
                .map_or("None".to_string(), |v| v.to_string()),
            self.deterministic_clustering,
            self.use_dims_reduction,
            self.rp_eps,
            self.persistence
                .as_ref()
                .map_or("None".to_string(), |s| s.1.display().to_string())
        )
    }
}

/// Configuration value that can hold different types while preserving type information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    Bool(bool),
    Usize(usize),
    F64(f64),
    String(String),
    OptionF64(Option<f64>),
    OptionUsize(Option<usize>),
    OptionU64(Option<u64>),
    TauMode(TauMode),
    OptionSamplerType(Option<SamplerType>),
}

impl ConfigValue {
    // Convenience methods for type extraction
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ConfigValue::Usize(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ConfigValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    // Convenience extraction methods
    pub fn as_tau_mode(&self) -> Option<&TauMode> {
        match self {
            ConfigValue::TauMode(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_sampler_type(&self) -> Option<&Option<SamplerType>> {
        match self {
            ConfigValue::OptionSamplerType(v) => Some(v),
            _ => None,
        }
    }
}

impl ArrowSpaceBuilder {
    pub fn builder_config_typed(&self) -> HashMap<String, ConfigValue> {
        let mut config = HashMap::new();

        config.insert(
            "prebuilt_spectral".to_string(),
            ConfigValue::Bool(self.prebuilt_spectral),
        );
        config.insert("lambda_eps".to_string(), ConfigValue::F64(self.lambda_eps));
        config.insert("lambda_k".to_string(), ConfigValue::Usize(self.lambda_k));
        config.insert(
            "lambda_topk".to_string(),
            ConfigValue::Usize(self.lambda_topk),
        );
        config.insert("lambda_p".to_string(), ConfigValue::F64(self.lambda_p));
        config.insert(
            "lambda_sigma".to_string(),
            ConfigValue::OptionF64(self.lambda_sigma),
        );
        config.insert("normalise".to_string(), ConfigValue::Bool(self.normalise));
        config.insert(
            "sparsity_check".to_string(),
            ConfigValue::Bool(self.sparsity_check),
        );
        config.insert(
            "synthesis".to_string(),
            ConfigValue::TauMode(self.synthesis.clone()),
        );
        config.insert(
            "sampling".to_string(),
            ConfigValue::OptionSamplerType(self.sampling.clone()),
        );
        config.insert(
            "cluster_max_clusters".to_string(),
            ConfigValue::OptionUsize(self.cluster_max_clusters),
        );
        config.insert(
            "cluster_radius".to_string(),
            ConfigValue::F64(self.cluster_radius),
        );
        config.insert(
            "clustering_seed".to_string(),
            ConfigValue::OptionU64(self.clustering_seed),
        );
        config.insert(
            "deterministic_clustering".to_string(),
            ConfigValue::Bool(self.deterministic_clustering),
        );
        config.insert(
            "use_dims_reduction".to_string(),
            ConfigValue::Bool(self.use_dims_reduction),
        );
        config.insert("rp_eps".to_string(), ConfigValue::F64(self.rp_eps));

        config
    }
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Primitive types
            ConfigValue::Bool(v) => write!(f, "{}", v),
            ConfigValue::Usize(v) => write!(f, "{}", v),
            ConfigValue::F64(v) => write!(f, "{}", v),
            ConfigValue::String(v) => write!(f, "{}", v),

            // Optional primitive types
            ConfigValue::OptionF64(opt) => match opt {
                Some(v) => write!(f, "{}", v),
                None => write!(f, "None"),
            },
            ConfigValue::OptionUsize(opt) => match opt {
                Some(v) => write!(f, "{}", v),
                None => write!(f, "None"),
            },
            ConfigValue::OptionU64(opt) => match opt {
                Some(v) => write!(f, "{}", v),
                None => write!(f, "None"),
            },

            // Custom domain types
            ConfigValue::TauMode(tau) => write!(f, "{}", tau),
            ConfigValue::OptionSamplerType(opt) => match opt {
                Some(sampler) => write!(f, "{:?}", sampler),
                None => write!(f, "None"),
            },
        }
    }
}
