use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::clustering::ClusteringHeuristic;
use crate::core::{ArrowSpace, TAUDEFAULT};
use crate::eigenmaps::{ClusteredOutput, EigenMaps};
use crate::graph::GraphLaplacian;
use crate::sampling::SamplerType;
use crate::taumode::TauMode;

#[derive(Clone, Debug)]
pub enum PairingStrategy {
    FastPair,            // 1-NN union via Smartcore FastPair
    Default,             // O(n^2) path
    CoverTreeKNN(usize), // k for k-NN build
}

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
    pub(crate) synthesis: TauMode, // (tau_mode)

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
        info!("Setting sparsity check falg: {}", sparsity_check);
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
