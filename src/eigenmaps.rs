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

use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::builder::ArrowSpaceBuilder;
use crate::core::{ArrowItem, ArrowSpace};
use crate::graph::{GraphFactory, GraphLaplacian};
use crate::taumode::TauMode;

/// This trait decomposes the `ArrowSpaceBuilder::build`` pipeline into explicit stages
/// for custom workflows, debugging, and analysis. All stages preserve the semantics
/// of the canonical build path: clustering heuristics, projection policies, λ-graph
/// parameters, and taumode λ computation are applied consistently.
pub trait EigenMaps {
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
    /// build items-graph laplacian
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
            let max = self
                .lambdas
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean = self.lambdas.iter().sum::<f64>() / self.lambdas.len() as f64;
            (min, max, mean)
        };

        info!(
            "λ computation complete: min={:.6}, max={:.6}, mean={:.6}",
            lambda_stats.0, lambda_stats.1, lambda_stats.2
        );
    }

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
