//! # Eigen Maps for ArrowSpace
//!
//! This module exposes the internal stages of the ArrowSpaceBuilder::build pipeline
//! as a trait-based API, enabling custom workflows that interleave clustering,
//! Laplacian construction, λ computation, and spectral analysis with external logic.
//!
//! # Pipeline Stages
//!
//! 1. **Clustering**: Optimal-K selection, inline sampling, incremental clustering,
//!    and optional Johnson-Lindenstrauss projection to compress high-dimensional centroids.
//! 2. **Eigenmaps**: Item-graph Laplacian construction from clustered centroids using
//!    the builder's λ-graph parameters (eps, k, topk, p, sigma, normalization policy).
//! 3. **Taumode**: Parallel per-row λ computation via TauMode synthetic index transform,
//!    storing a single spectral roughness scalar per row without retaining the graph.
//! 4. **Spectral** (optional): Laplacian-of-Laplacian (F×F feature graph) for higher-order
//!    spectral analysis when spectral signals are explicitly required.
//! 5. **Search**: λ-aware nearest-neighbor search blending semantic cosine similarity
//!    with λ proximity, using the precomputed index λs and query λ prepared on-the-fly.
//!
//! # Design Philosophy
//!
//! - **Trait-based**: All stages are methods on the `EigenMaps` trait implemented for
//!   `ArrowSpace`, enabling extension and mocking for testing workflows.
//! - **One-shot λ computation**: The `compute_taumode` step computes λ once using the
//!   parallel routines in taumode.rs, storing a single scalar per row; the Laplacian
//!   can be discarded afterward, preserving spectral information without graph storage.
//! - **Projection-aware**: When JL projection is used, query vectors are automatically
//!   projected at search time to match the reduced-dimension index space, ensuring
//!   consistent λ computation and similarity metrics.
//! - **Logging-first**: All stages emit structured logs (info/debug/trace) for observability
//!   during index construction and search, compatible with env_logger or tracing backends.
//!
//! # Usage Example
//!
//! ```ignore
//! use arrowspace::eigenmaps::{EigenMaps, ClusteredOutput};
//! use arrowspace::builder::ArrowSpaceBuilder;
//!
//! let mut builder = ArrowSpaceBuilder::new()
//!     .with_lambda_graph(1e-3, 6, 3, 2.0, None)
//!     .with_synthesis(TauMode::Median);
//!
//! // Stage 1: Clustering
//! let ClusteredOutput { mut aspace, centroids, n_items, .. } =
//!     aspace.start_clustering(&mut builder, rows);
//!
//! // Stage 2: Eigenmaps (Laplacian construction)
//! let gl = aspace.eigenmaps(&builder, &centroids, n_items);
//!
//! // Stage 3: Compute λ values (parallel)
//! aspace.compute_taumode(&gl);
//!
//! // Optional Stage 4: Spectral feature graph
//! aspace = aspace.spectral(&gl);
//!
//! // Stage 5: Search with λ-aware ranking
//! let hits = aspace.search(&query_vec, &gl, k, alpha);
//! ```

use log::{debug, info, trace};
use std::sync::{Arc, Mutex};

use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::builder::ArrowSpaceBuilder;
use crate::clustering::{self, ClusteringHeuristic};
use crate::core::{ArrowItem, ArrowSpace};
use crate::graph::{GraphFactory, GraphLaplacian};
use crate::reduction::{compute_jl_dimension, ImplicitProjection};
use crate::sampling::{InlineSampler, SamplerType};
use crate::taumode::TauMode;

/// Output of the clustering stage: centroids, projected dimensions, and metadata-enriched ArrowSpace.
#[derive(Clone, Debug)]
pub struct ClusteredOutput {
    /// ArrowSpace with cluster assignments, sizes, radius, and optional projection matrix.
    pub aspace: ArrowSpace,
    /// Clustered centroids (X × F' where X ≤ max_clusters, F' is reduced_dim or original F).
    pub centroids: DenseMatrix<f64>,
    /// Effective dimensionality after optional JL projection (F' ≤ F).
    pub reduced_dim: usize,
    /// Original dataset row count (N).
    pub n_items: usize,
    /// Original dataset column count (F).
    pub n_features: usize,
}

/// This trait decomposes the `ArrowSpaceBuilder::build`` pipeline into explicit stages
/// for custom workflows, debugging, and analysis. All stages preserve the semantics
/// of the canonical build path: clustering heuristics, projection policies, λ-graph
/// parameters, and taumode λ computation are applied consistently.
pub trait EigenMaps {
    /// Stage 1: Optimal-K clustering with sampling and optional JL projection.
    ///
    /// Computes clustering parameters (X, radius, intrinsic_dim) using the builder's
    /// heuristic, runs incremental clustering with the configured sampler, and applies
    /// JL projection if enabled and beneficial. Returns centroids and an ArrowSpace
    /// enriched with cluster metadata and projection state.
    ///
    /// # Arguments
    /// - `builder`: ArrowSpaceBuilder with configured clustering, sampling, and projection.
    /// - `rows`: Original dataset as Vec<Vec<f64>> (N × F).
    ///
    /// # Returns
    /// `ClusteredOutput` containing centroids (X × F'), enriched ArrowSpace, and dimensions.
    fn start_clustering(
        builder: &mut ArrowSpaceBuilder,
        rows: Vec<Vec<f64>>,
    ) -> ClusteredOutput;

    /// Stage 2: Construct item-graph Laplacian from clustered centroids.
    ///
    /// Builds the λ-graph using the builder's eps, k, topk, p, sigma, normalization,
    /// and sparsity-check parameters. The Laplacian is computed over centroids (graph
    /// nodes are centroids, edges weighted by λ-proximity kernel), transposed internally
    /// to match the item-as-node convention.
    ///
    /// # Arguments
    /// - `builder`: ArrowSpaceBuilder with λ-graph configuration.
    /// - `centroids`: X × F' matrix from clustering stage.
    /// - `n_items`: Original dataset row count (for nnodes tracking).
    ///
    /// # Returns
    /// `GraphLaplacian` with nnodes = n_items, ready for taumode computation.
    fn eigenmaps(
        &mut self,
        builder: &ArrowSpaceBuilder,
        centroids: &DenseMatrix<f64>,
        n_items: usize,
    ) -> GraphLaplacian;

    /// Stage 3: Compute per-row λ values using TauMode synthetic index transform (parallel).
    ///
    /// Computes a single spectral roughness scalar per row by blending Rayleigh energy
    /// with local Dirichlet dispersion, normalized via the ArrowSpace's taumode policy
    /// (Median, Mean, etc.). This is the "compute once, store scalar S_r" design from
    /// taumode.rs, enabling λ-aware search without retaining the graph Laplacian.
    ///
    /// # Arguments
    /// - `gl`: GraphLaplacian from eigenmaps stage.
    ///
    /// # Side Effects
    /// Mutates `self.lambdas` in place with computed λ values for all rows.
    fn compute_taumode(&mut self, gl: &GraphLaplacian);

    /// Stage 5: λ-aware nearest-neighbor search with precomputed λ values.
    ///
    /// Prepares query λ by projecting the query vector (if projection was used during
    /// indexing) and computing its Rayleigh or synthetic λ against the Laplacian. Ranks
    /// index rows by blending cosine similarity (weighted by alpha) with λ proximity
    /// (weighted by 1 - alpha), using the precomputed index λs.
    ///
    /// # Arguments
    /// - `item`: Query vector in original F-dimensional space.
    /// - `gl`: GraphLaplacian used for query λ preparation.
    /// - `k`: Number of nearest neighbors to return.
    /// - `alpha`: Semantic similarity weight in [0, 1] (1 = pure cosine, 0 = pure λ).
    ///
    /// # Returns
    /// Vec of (row_index, combined_similarity_score) sorted descending, length ≤ k.
    ///
    /// # Panics
    /// Panics (in debug builds) if `compute_taumode` was not called before search.
    fn search(
        &mut self,
        item: &[f64],
        gl: &GraphLaplacian,
        k: usize,
        alpha: f64,
    ) -> Vec<(usize, f64)>;
}

impl EigenMaps for ArrowSpace {
    fn start_clustering(
        builder: &mut ArrowSpaceBuilder,
        rows: Vec<Vec<f64>>,
    ) -> ClusteredOutput {
        let n_items = rows.len();
        let n_features = rows.first().map(|r| r.len()).unwrap_or(0);

        info!(
            "EigenMaps::start_clustering: N={} items, F={} features",
            n_items, n_features
        );

        // Prepare base ArrowSpace with the builder's taumode (will be used in compute_taumode)
        debug!("Creating ArrowSpace with taumode: {:?}", builder.synthesis);
        let mut aspace = ArrowSpace::new(rows.clone(), builder.synthesis);

        // Configure inline sampler matching builder policy
        let sampler: Arc<Mutex<dyn InlineSampler>> = match builder.sampling.clone() {
            Some(SamplerType::Simple(r)) => {
                debug!("Using Simple sampler with ratio {:.2}", r);
                Arc::new(Mutex::new(SamplerType::new_simple(r)))
            }
            Some(SamplerType::DensityAdaptive(r)) => {
                debug!("Using DensityAdaptive sampler with ratio {:.2}", r);
                Arc::new(Mutex::new(SamplerType::new_density_adaptive(r)))
            }
            None => {
                debug!("No sampling configured, using full dataset");
                Arc::new(Mutex::new(SamplerType::new_simple(1.0)))
            }
        };

        // Auto-compute optimal clustering parameters via heuristic
        info!("Computing optimal clustering parameters");
        let (k_opt, radius, intrinsic_dim) =
            builder.compute_optimal_k(&rows, n_items, n_features, builder.clustering_seed);
        debug!(
            "Optimal clustering: K={}, radius={:.6}, intrinsic_dim={}",
            k_opt, radius, intrinsic_dim
        );

        builder.cluster_max_clusters = Some(k_opt);
        builder.cluster_radius = radius;

        // Run incremental clustering with sampling
        info!(
            "Running incremental clustering: max_clusters={}, radius={:.6}",
            k_opt, radius
        );
        let (clustered_dm, assignments, sizes) =
            clustering::run_incremental_clustering_with_sampling(
                builder,
                &rows,
                n_features,
                k_opt,
                radius,
                sampler,
            );

        let n_clusters = clustered_dm.shape().0;
        info!(
            "Clustering complete: {} centroids, {} items assigned",
            n_clusters,
            assignments.iter().filter(|x| x.is_some()).count()
        );

        // Store clustering metadata in ArrowSpace
        aspace.n_clusters = n_clusters;
        aspace.cluster_assignments = assignments;
        aspace.cluster_sizes = sizes;
        aspace.cluster_radius = radius;

        // Optional JL projection for high-dimensional datasets
        let (centroids, reduced_dim) = if builder.use_dims_reduction && n_features > 64 {
            let jl_dim = compute_jl_dimension(n_clusters, builder.rp_eps);
            let target_dim = jl_dim.min(n_features / 2);

            if target_dim < n_features {
                info!(
                    "Applying JL projection: {} features → {} dimensions (ε={:.2})",
                    n_features, target_dim, builder.rp_eps
                );
                let implicit_proj = ImplicitProjection::new(n_features, target_dim);
                let projected = crate::reduction::project_matrix(&clustered_dm, &implicit_proj);

                aspace.projection_matrix = Some(implicit_proj);
                aspace.reduced_dim = Some(target_dim);

                let compression = n_features as f64 / target_dim as f64;
                info!(
                    "Projection complete: {:.1}x compression, stored as 8-byte seed",
                    compression
                );

                (projected, target_dim)
            } else {
                debug!(
                    "JL target dimension {} >= original {}, skipping projection",
                    target_dim, n_features
                );
                (clustered_dm.clone(), n_features)
            }
        } else {
            debug!("JL projection disabled or dimension too small");
            (clustered_dm.clone(), n_features)
        };

        trace!("Clustering stage complete, returning ClusteredOutput");
        ClusteredOutput {
            aspace,
            centroids,
            reduced_dim,
            n_items,
            n_features,
        }
    }

    fn eigenmaps(
        &mut self,
        builder: &ArrowSpaceBuilder,
        centroids: &DenseMatrix<f64>,
        n_items: usize,
    ) -> GraphLaplacian {
        let (n_centroids, n_features) = centroids.shape();
        info!(
            "EigenMaps::eigenmaps: Building Laplacian from {} centroids × {} features",
            n_centroids, n_features
        );
        debug!(
            "λ-graph parameters: eps={}, k={}, topk={}, p={}, sigma={:?}, normalize={}",
            builder.lambda_eps,
            builder.lambda_k,
            builder.lambda_topk,
            builder.lambda_p,
            builder.lambda_sigma,
            builder.normalise
        );

        let gl = GraphFactory::build_laplacian_matrix_from_k_cluster(
            &centroids,
            builder.lambda_eps,
            builder.lambda_k,
            builder.lambda_topk,
            builder.lambda_p,
            builder.lambda_sigma,
            builder.normalise,
            builder.sparsity_check,
            n_items,
        );

        if builder.prebuilt_spectral {
            // Stage 4 (optional): Construct F×F feature Laplacian (Laplacian-of-Laplacian).
            //
            // Builds a spectral feature graph by transposing the item Laplacian and computing
            // a new Laplacian over features (columns become nodes, edges weighted by feature
            // correlation across items modulated by the item graph). Stores result in self.signals.
            // ### Why negative lambdas are valid in this case
            // The Rayleigh quotient \$ R(L, x) = \frac{x^T L x}{x^T x} \$ can be **negative** when:
            // 1. The Laplacian $L$ is **not positive semi-definite** (e.g., unnormalized Laplacians or
            //   feature-space Laplacians with negative eigenvalues)
            // 2. The numerator $x^T L x$ is negative for some vectors $x$
            //
            // For the **spectral F×F feature Laplacian**, the matrix represents relationships between features (not items),
            //   and the resulting Laplacian can have negative eigenvalues depending on the feature correlation structure.
            trace!("Building spectral Laplacian for ArrowSpace");
            GraphFactory::build_spectral_laplacian(self, &gl);
            debug!(
                "Spectral Laplacian built with signals shape: {:?}",
                self.signals.shape()
            );
        }

        info!(
            "Laplacian construction complete: {}×{} matrix, {} non-zeros, {:.2}% sparse",
            gl.shape().0,
            gl.shape().1,
            gl.nnz(),
            GraphLaplacian::sparsity(&gl.matrix) * 100.0
        );

        gl
    }

    fn compute_taumode(&mut self, gl: &GraphLaplacian) {
        info!(
            "EigenMaps::compute_taumode: Computing λ values for {} items using {:?}",
            self.nitems, self.taumode
        );
        debug!(
            "Laplacian: {} nodes, {} non-zeros",
            gl.nnodes,
            gl.matrix.nnz()
        );

        // Parallel per-row λ computation via TauMode synthetic index transform
        TauMode::compute_taumode_lambdas_parallel(self, gl, self.taumode);

        let lambda_stats = {
            let min = self.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self.lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean = self.lambdas.iter().sum::<f64>() / self.lambdas.len() as f64;
            (min, max, mean)
        };

        info!(
            "λ computation complete: min={:.6}, max={:.6}, mean={:.6}",
            lambda_stats.0, lambda_stats.1, lambda_stats.2
        );
    }

    // fn spectral(self, gl: &GraphLaplacian) -> Self {
    //     info!(
    //         "EigenMaps::spectral: Building F×F feature Laplacian for {} features",
    //         self.nfeatures
    //     );
    //     debug!(
    //         "Input Laplacian: {}×{}, graph params: {:?}",
    //         gl.shape().0,
    //         gl.shape().1,
    //         gl.graph_params
    //     );

    //     let aspace_with_signals = GraphFactory::build_spectral_laplacian(self, gl);

    //     let (signal_rows, signal_cols) = aspace_with_signals.signals.shape();
    //     info!(
    //         "Spectral Laplacian complete: {}×{} signals matrix, {:.2}% sparse",
    //         signal_rows,
    //         signal_cols,
    //         GraphLaplacian::sparsity(&aspace_with_signals.signals) * 100.0
    //     );

    //     aspace_with_signals
    // }
    
    fn search(
        &mut self,
        item: &[f64],
        gl: &GraphLaplacian,
        k: usize,
        alpha: f64,
    ) -> Vec<(usize, f64)> {
        info!(
            "EigenMaps::search: k={}, alpha={:.2}, query_dim={}",
            k,
            alpha,
            item.len()
        );

        // Ensure λs have been precomputed
        debug_assert!(
            self.lambdas[0..self.nitems.min(4)]
                .iter()
                .any(|&v| v != 0.0)
                || self.nitems == 0,
            "call compute_taumode(...) before search to populate lambdas"
        );

        trace!("Preparing query λ with projection and taumode policy");
        let q_lambda = self.prepare_query_item(item, gl);
        let projected_query = self.project_query(item);

        debug!(
            "Query λ={:.6}, projected_dim={}",
            q_lambda,
            projected_query.len()
        );

        let q = ArrowItem::new(projected_query, q_lambda);

        // λ-aware semantic ranking
        let results = self.search_lambda_aware(&q, k, alpha);

        info!(
            "Search complete: {} results returned, top_score={:.6}",
            results.len(),
            results.first().map(|(_, s)| *s).unwrap_or(0.0)
        );

        results
    }
}
