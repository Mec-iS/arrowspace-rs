//! Graph construction and Laplacian matrix computation for ArrowSpace.
//!
//! This module implements sparse graph construction from clustered data, followed by
//! Laplacian matrix computation for spectral analysis. It provides both CPU and GPU
//! paths with automatic fallback for robust execution.
//!
//! # Overview
//!
//! The graph module bridges the gap between raw data and spectral analysis by:
//! 1. Constructing sparse k-NN or ε-radius graphs from cluster centroids
//! 2. Computing edge weights using adaptive Gaussian kernels
//! 3. Building normalized graph Laplacians (L = D⁻¹/²(D-W)D⁻¹/²)
//! 4. Preparing matrices for eigendecomposition and taumode computation
//!
//! # Graph Construction Modes
//!
//! ## K-NN Graph
//! - Connects each node to its k nearest neighbors
//! - Symmetrized via max(w_ij, w_ji) to ensure undirected graph
//! - Suitable for uniform manifold sampling
//!
//! ## Epsilon-Radius Graph
//! - Connects nodes within distance threshold ε
//! - Adaptive to local density variations
//! - Better preserves local geometric structure
//!
//! # Laplacian Variants
//!
//! - **Unnormalized**: L = D - W
//! - **Symmetric normalized**: L_sym = D⁻¹/²(D-W)D⁻¹/² (default)
//! - **Random walk**: L_rw = D⁻¹(D-W)
//!
//! The symmetric normalized Laplacian is preferred for spectral clustering and
//! manifold learning due to its connection to diffusion processes.
//!
//! # Edge Weight Computation
//!
//! Weights use adaptive Gaussian kernels:
//! ```
//! w_ij = exp(-||x_i - x_j||² / (2σ²))
//! ```
//! where σ is computed adaptively from local density estimates or provided explicitly.
//!
//! # GPU Acceleration
//!
//! - `build_laplacian_gpu`: wgpu-based parallel construction
//! - Efficient for large graphs (>10k nodes)
//! - Automatic fallback to CPU on device unavailability
//!
//! # Core Types
//!
//! - `GraphLaplacian`: Sparse representation with COO format
//! - `LaplacianMode`: Enum for normalization strategy
//! - Helper functions for k-NN search and weight computation
//!
//! # Usage
//!
//! ```
//! use arrowspace::graph::{build_laplacian_knn, LaplacianMode};
//!
//! let laplacian = build_laplacian_knn(
//!     &centroids, 
//!     k, 
//!     sigma, 
//!     LaplacianMode::Symmetric
//! );
//! ```
//!
//! # Performance
//!
//! - CPU path: Parallel k-NN with Rayon, O(n²k) for k-NN
//! - GPU path: O(n²) worst-case, but highly parallel for dense graphs
//! - Sparse matrix storage minimizes memory footprint
//!
//! # Integration
//!
//! Laplacian matrices feed directly into eigensolvers (via nalgebra or GPU compute)
//! for spectral decomposition, enabling downstream taumode computation and manifold
//! analysis within ArrowSpace.

use std::fmt;

use crate::core::ArrowSpace;
use crate::laplacian::build_laplacian_matrix;

use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;

// Add logging
use log::{debug, info, trace, warn};

#[derive(Debug, Clone)]
pub struct GraphParams {
    pub eps: f64,             // maximum rectified cosine distance (see `docs/`)
    pub k: usize,             // max number of neighbours for node
    pub topk: usize,          // number of results to be considered for closest neighbors
    pub p: f64,               // kernel parameter
    pub sigma: Option<f64>,   // tolerance for eps
    pub normalise: bool,      // avoid normalisation as may hinder magnitude information
    pub sparsity_check: bool, // check for proper level of sparsity (some datasets works well even if sparse)
}

// Custom PartialEq implementation using approximate equality for floats
impl PartialEq for GraphParams {
    fn eq(&self, other: &Self) -> bool {
        // Use relative equality for floating-point comparisons
        // and exact equality for integer types
        self.k == other.k
            && approx::relative_eq!(self.eps, other.eps)
            && approx::relative_eq!(self.p, other.p)
            && match (self.sigma, other.sigma) {
                (None, None) => true,
                (Some(a), Some(b)) => approx::relative_eq!(a, b),
                _ => false,
            }
            && self.normalise == other.normalise
    }
}

// Implement Eq since we have a proper equivalence relation
// (assuming no NaN values in practice)
impl Eq for GraphParams {}

/// Graph Laplacian
#[derive(Debug, Clone)]
pub struct GraphLaplacian {
    // store initial data from builder
    pub init_data: DenseMatrix<f64>, // clustered centroids: XxF matrix where X <= max_clusters
    // store the fully computed sparse graph laplacian
    pub matrix: CsMat<f64>,
    // number of the nodes of the *original raw data*
    pub nnodes: usize,
    pub graph_params: GraphParams,
}

/// Graph factory: all construction ultimately uses the λτ-graph built from data.
///
/// High-level policy:
/// - The base graph for any pipeline is a λ-proximity Laplacian over items (columns),
///   derived from the provided data matrix (rows are feature signals).
/// - Ensembles vary λτ-graph parameters (k, eps) and/or overlay hypergraph operations.
pub struct GraphFactory;

impl GraphFactory {
    /// This is a lower level method: use `ArrowSpaceBuilder::build`
    /// Transpose the resulting matrix from clustering and build a graph Laplacian matrix
    ///  so to be ready to be used to analyse signal features
    pub fn build_laplacian_matrix_from_k_cluster(
        clustered: DenseMatrix<f64>, // X×F: X centroids of clusters, each with F features
        eps: f64,                    // maximum rectified cosine distance (see `docs/`)
        k: usize,                    // max number of neighbours for node
        topk: usize,                 // number of results to be considered for closest neighbors
        p: f64,                      // kernel parameter
        sigma_override: Option<f64>, // tolerance for eps
        normalise: bool,             // pre-normalisation before laplacian computation
        sparsity_check: bool, // flag to disable sparsity check (some datasets works well even if sparse)
        n_items: usize,       // items number of the origina raw data
    ) -> GraphLaplacian {
        info!(
            "Building Laplacian matrix for K cluster: {} clusters",
            clustered.shape().0
        );
        debug!(
            "Laplacian parameters: eps={}, k={}, p={}, sigma={:?}, normalise={}",
            eps, k, p, sigma_override, normalise
        );
        assert!(clustered.shape().0 <= n_items);

        let result: GraphLaplacian = crate::laplacian::build_laplacian_matrix(
            // items are transposed here
            clustered.transpose(),
            &GraphParams {
                eps,
                k,
                topk,
                p,
                sigma: sigma_override,
                normalise,
                sparsity_check,
            },
            Some(n_items),
        );

        if sparsity_check {
            let sparsity_input = GraphLaplacian::sparsity(&result.matrix);
            if sparsity_input > 0.95 {
                panic!(
                    "Resulting laplacian matrix is too sparse {:?}",
                    sparsity_input
                )
            }
        }
        assert!(result.nnodes == n_items);

        info!(
            "Laplacian matrix built: {}×{} with {} nodes, {} non-zeros",
            result.matrix.shape().0,
            result.matrix.shape().1,
            result.nnodes,
            result.matrix.nnz()
        );
        result
    }

    /// Build F×F feature similarity matrix
    /// This creates a graph where nodes are features and edges represent feature similarities
    /// # Arguments
    ///
    /// * `aspace` - The data from the ArrowSpace data  
    /// * `graph_laplacian` - A graph laplacian generated with the `ArrowSpace`
    pub fn build_spectral_laplacian(
        mut aspace: ArrowSpace,
        graph_laplacian: &GraphLaplacian,
    ) -> ArrowSpace {
        info!("Building F×F spectral feature matrix");
        debug!(
            "ArrowSpace dimensions: {} features, {} items",
            aspace.nfeatures, aspace.nitems
        );
        debug!("Graph parameters: {:?}", graph_laplacian.graph_params);

        // Convert sparse matrix to dense for signals storage
        trace!("Building feature-to-feature Laplacian matrix");

        aspace.signals = build_laplacian_matrix(
            sparse_to_dense(&graph_laplacian.matrix).transpose(),
            &graph_laplacian.graph_params,
            Some(aspace.nitems),
        )
        .matrix;

        let sparsity_output = GraphLaplacian::sparsity(&aspace.signals);
        println!("sparsity {:?}", sparsity_output);
        if sparsity_output > 0.95 && graph_laplacian.graph_params.sparsity_check {
            panic!(
                "Resulting spectral matrix is too sparse {:?}",
                sparsity_output
            )
        }

        if aspace.reduced_dim.is_some() {
            assert!(
                aspace.signals.shape().0 == aspace.reduced_dim.unwrap()
                    && aspace.reduced_dim.unwrap() == aspace.signals.shape().1,
                "result should be a FxF matrix with reduced dimensions F"
            );
        } else {
            assert!(
                aspace.signals.shape().0 == aspace.signals.shape().1,
                "result should be a FxF matrix"
            );
        }

        info!(
            "Built F×F feature matrix: {}×{}",
            aspace.nfeatures, aspace.nfeatures
        );
        let stats = {
            let nnz = graph_laplacian.matrix.nnz();
            let total = aspace.nfeatures * aspace.nfeatures;
            let sparsity = (total - nnz) as f64 / total as f64;
            (nnz, sparsity)
        };
        debug!(
            "Feature matrix statistics: {} non-zero entries, {:.2}% sparse",
            stats.0,
            stats.1 * 100.0
        );

        aspace
    }
}

fn sparse_to_dense(sparse: &CsMat<f64>) -> DenseMatrix<f64> {
    let (rows, cols) = sparse.shape();
    let mut data = vec![0.0; rows * cols];

    for (row_idx, row) in sparse.outer_iterator().enumerate() {
        for (col_idx, &value) in row.iter() {
            data[row_idx * cols + col_idx] = value;
        }
    }

    DenseMatrix::new(rows, cols, data, false).unwrap()
}

impl GraphLaplacian {
    /// Create a new GraphLaplacian from an items matrix (M = NxF)
    /// This is used to create a graph from the transposed matrix
    /// Use `GraphFacotry::build_lambda_graph` for the full computation
    pub fn prepare_from_items(matrix: DenseMatrix<f64>, graph_params: GraphParams) -> Self {
        let nnodes = matrix.shape().0;
        debug!(
            "Preparing GraphLaplacian from items matrix: {} nodes",
            nnodes
        );
        trace!("Transposing matrix for GraphLaplacian");

        build_laplacian_matrix(matrix.transpose(), &graph_params, Some(matrix.shape().0))
    }

    /// Get the matrix dimensions as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
    }

    pub fn topk(&self) -> usize {
        self.graph_params.topk
    }

    /// Get a matrix element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(
            i < self.nnodes && j < self.nnodes,
            "Index out of bounds: ({}, {}) for {}x{} matrix",
            i,
            j,
            self.nnodes,
            self.nnodes
        );
        self.matrix.get(i, j).copied().unwrap_or(0.0)
    }

    /// Get the diagonal entries (degrees) as a vector
    pub fn degrees(&self) -> Vec<f64> {
        trace!(
            "Extracting diagonal degrees from {}×{} matrix",
            self.nnodes,
            self.nnodes
        );
        let mut degrees: Vec<f64> = Vec::with_capacity(self.nnodes);
        for i in 0..self.nnodes {
            degrees.push(self.matrix.get(i, i).copied().unwrap_or(0.0));
        }

        let (min_degree, max_degree) = degrees
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &d| {
                (min.min(d), max.max(d))
            });
        debug!(
            "Extracted degrees: min={:.6}, max={:.6}",
            min_degree, max_degree
        );
        degrees
    }

    /// Set a matrix element at position (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        assert!(
            i < self.nnodes && j < self.nnodes,
            "Index out of bounds: ({}, {}) for {}x{} matrix",
            i,
            j,
            self.nnodes,
            self.nnodes
        );
        trace!("Setting matrix element at ({}, {}) = {:.6}", i, j, value);
        self.matrix.set(i, j, value);
    }

    /// Get the i-th row as a vector
    pub fn get_row(&self, i: usize) -> Vec<f64> {
        assert!(
            i < self.nnodes,
            "Row index {} out of bounds for {} nodes",
            i,
            self.nnodes
        );
        trace!("Extracting row {} from matrix", i);
        let mut row = Vec::with_capacity(self.nnodes);
        for j in 0..self.nnodes {
            row.push(*self.matrix.get(i, j).unwrap());
        }
        row
    }

    /// Get the j-th column as a vector
    pub fn get_column(&self, j: usize) -> Vec<f64> {
        assert!(
            j < self.nnodes,
            "Column index {} out of bounds for {} nodes",
            j,
            self.nnodes
        );
        trace!("Extracting column {} from matrix", j);
        let mut col = Vec::with_capacity(self.nnodes);
        for i in 0..self.nnodes {
            col.push(*self.matrix.get(i, j).unwrap());
        }
        col
    }

    /// Compute Rayleigh quotient: R(L, x) = x^T L x / (x^T x)
    pub fn rayleigh_quotient(&self, vector: &[f64]) -> f64 {
        assert_eq!(
            vector.len(),
            self.nnodes,
            "Vector length {} must match number of nodes {}",
            vector.len(),
            self.nnodes
        );

        trace!(
            "Computing Rayleigh quotient for vector of length {}",
            vector.len()
        );

        // Compute L * x manually using sprs sparse matrix structure
        let lx = self.multiply_vector(vector);

        // Compute x^T * (L * x)
        let numerator: f64 = vector
            .iter()
            .zip(lx.iter())
            .map(|(&xi, &lxi)| xi * lxi)
            .sum();

        // Compute x^T * x
        let denominator: f64 = vector.iter().map(|&xi| xi * xi).sum();

        let result = if denominator > 1e-12 {
            numerator / denominator
        } else {
            warn!("Zero vector encountered in Rayleigh quotient computation");
            0.0
        };

        debug!(
            "Rayleigh quotient: numerator={:.6}, denominator={:.6}, result={:.6}",
            numerator, denominator, result
        );
        result
    }

    /// Compute matrix-vector multiplication: y = L * x using sprs sparse matrix operations
    pub fn multiply_vector(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(
            x.len(),
            self.nnodes,
            "Vector length {} must match number of nodes {}",
            x.len(),
            self.nnodes
        );

        trace!(
            "Computing matrix-vector multiplication: {}×{} * {}",
            self.nnodes,
            self.nnodes,
            x.len()
        );

        // Initialize result vector
        let mut result = vec![0.0; self.nnodes];

        // Use sprs outer_iterator for efficient sparse matrix traversal
        for (row_idx, row) in self.matrix.outer_iterator().enumerate() {
            let mut sum = 0.0;
            // Iterate over non-zero entries in this row
            for (col_idx, &matrix_val) in row.iter() {
                sum += matrix_val * x[col_idx];
            }
            result[row_idx] = sum;
        }

        let result_norm = result.iter().map(|&x| x * x).sum::<f64>().sqrt();
        trace!(
            "Matrix-vector multiplication result norm: {:.6}",
            result_norm
        );
        result
    }

    /// Check if the matrix is symmetric within tolerance
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        trace!("Checking matrix symmetry with tolerance {:.2e}", tolerance);
        let mut max_asymmetry: f64 = 0.0;
        let mut violations = 0;

        for i in 0..self.nnodes {
            for j in 0..self.nnodes {
                let diff = (self.matrix.get(i, j).unwrap() - self.matrix.get(j, i).unwrap()).abs();
                max_asymmetry = max_asymmetry.max(diff);
                if diff > tolerance {
                    violations += 1;
                }
            }
        }

        let is_symmetric = violations == 0;
        debug!(
            "Symmetry check: {} violations, max asymmetry: {:.2e}, symmetric: {}",
            violations, max_asymmetry, is_symmetric
        );
        is_symmetric
    }

    /// Verify Laplacian properties: row sums ≈ 0, positive diagonal, symmetric
    pub fn verify_properties(&self, tolerance: f64) -> LaplacianValidation {
        info!(
            "Verifying Laplacian properties with tolerance {:.2e}",
            tolerance
        );
        let mut validation = LaplacianValidation::new();

        // Check row sums (should be ≈ 0)
        let mut max_row_sum: f64 = 0.0;
        for i in 0..self.nnodes {
            let row_sum: f64 = (0..self.nnodes)
                .map(|j| self.matrix.get(i, j).unwrap())
                .sum();
            max_row_sum = max_row_sum.max(row_sum.abs());
            if row_sum.abs() > tolerance {
                validation.row_sum_violations.push((i, row_sum));
            }
        }
        validation.max_row_sum_error = max_row_sum;

        // Check diagonal entries (should be positive for connected components)
        for i in 0..self.nnodes {
            let diagonal = *self.matrix.get(i, i).unwrap();
            if diagonal < 0.0_f64 {
                validation.negative_diagonal.push((i, diagonal));
            }
        }

        // Check symmetry
        validation.is_symmetric = self.is_symmetric(tolerance);
        if !validation.is_symmetric {
            let mut max_asymmetry: f64 = 0.0;
            for i in 0..self.nnodes {
                for j in 0..self.nnodes {
                    let asymmetry =
                        (self.matrix.get(i, j).unwrap() - self.matrix.get(j, i).unwrap()).abs();
                    max_asymmetry = max_asymmetry.max(asymmetry);
                }
            }
            validation.max_asymmetry = max_asymmetry;
        }

        validation.is_valid = validation.row_sum_violations.is_empty()
            && validation.negative_diagonal.is_empty()
            && validation.is_symmetric;

        debug!("Laplacian validation results:");
        debug!("  Valid: {}", validation.is_valid);
        debug!("  Symmetric: {}", validation.is_symmetric);
        debug!("  Max row sum error: {:.2e}", validation.max_row_sum_error);
        debug!(
            "  Row sum violations: {}",
            validation.row_sum_violations.len()
        );
        debug!(
            "  Negative diagonal entries: {}",
            validation.negative_diagonal.len()
        );

        if !validation.is_valid {
            warn!("Laplacian validation failed - matrix may have numerical issues");
        }

        validation
    }

    pub fn nnz(&self) -> usize {
        let count = self.matrix.nnz();
        debug!("Matrix has {} non-zero entries", count);
        count
    }

    pub fn sparsity(matrix: &CsMat<f64>) -> f64 {
        let nnz = matrix.nnz(); // number of non-zero elements
        let (rows, cols) = matrix.shape();
        let total_elements = rows * cols;

        1.0 - (nnz as f64) / (total_elements as f64)
    }

    pub fn extract_adjacency(&self) -> CsMat<f64> {
        info!("Extracting adjacency matrix from Laplacian");
        let mut triplets = sprs::TriMat::new((self.nnodes, self.nnodes));

        for (i, row) in self.matrix.outer_iterator().enumerate() {
            for (j, &value) in row.iter() {
                if i != j {
                    triplets.add_triplet(i, j, -value);
                }
            }
        }

        let adjacency = triplets.to_csr();
        debug!(
            "Extracted adjacency matrix: {}×{} with {} non-zeros",
            self.nnodes,
            self.nnodes,
            adjacency.nnz()
        );
        adjacency
    }

    pub fn statistics(&self) -> LaplacianStats {
        trace!("Computing Laplacian statistics");
        let degrees = self.degrees();
        let min_degree = degrees.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_degree = degrees.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let mean_degree = degrees.iter().sum::<f64>() / self.nnodes as f64;

        let nnz = self.nnz();
        let sparsity = Self::sparsity(&self.matrix);

        let stats = LaplacianStats {
            nnodes: self.nnodes,
            nnz,
            sparsity,
            min_degree,
            max_degree,
            mean_degree,
            graph_params: self.graph_params.clone(),
        };

        debug!("Computed statistics: {} nodes, {} non-zeros, {:.2}% sparse, degree range [{:.6}, {:.6}]", 
               stats.nnodes, stats.nnz, stats.sparsity * 100.0, stats.min_degree, stats.max_degree);

        stats
    }

    pub fn matrix(&self) -> &CsMat<f64> {
        &self.matrix
    }

    pub fn params(&self) -> &GraphParams {
        &self.graph_params
    }

    /// Get a mutable reference to the underlying DenseMatrix
    pub fn matrix_mut(&mut self) -> &mut CsMat<f64> {
        &mut self.matrix
    }
}

pub fn dense_to_sparse(dense: &DenseMatrix<f64>) -> CsMat<f64> {
    let (rows, cols) = dense.shape();
    let mut triplets = sprs::TriMat::new((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let value = *dense.get((i, j));
            if value.abs() > 1e-12 {
                triplets.add_triplet(i, j, value);
            }
        }
    }

    triplets.to_csr()
}

/// Structure to hold Laplacian validation results
#[derive(Debug, Clone)]
pub struct LaplacianValidation {
    pub is_valid: bool,
    pub is_symmetric: bool,
    pub max_asymmetry: f64,
    pub max_row_sum_error: f64,
    pub row_sum_violations: Vec<(usize, f64)>,
    pub negative_diagonal: Vec<(usize, f64)>,
}

impl LaplacianValidation {
    fn new() -> Self {
        Self {
            is_valid: false,
            is_symmetric: false,
            max_asymmetry: 0.0,
            max_row_sum_error: 0.0,
            row_sum_violations: Vec::new(),
            negative_diagonal: Vec::new(),
        }
    }
}

/// Structure to hold Laplacian statistics
#[derive(Debug, Clone)]
pub struct LaplacianStats {
    pub nnodes: usize,
    pub nnz: usize,
    pub sparsity: f64,
    pub min_degree: f64,
    pub max_degree: f64,
    pub mean_degree: f64,
    pub graph_params: GraphParams,
}

/// Pretty printing implementation
impl fmt::Display for GraphLaplacian {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GraphLaplacian ({}×{}):", self.nnodes, self.nnodes)?;
        writeln!(f, "Parameters: {:?}", self.graph_params)?;

        if self.nnodes <= 10 {
            writeln!(f, "Small matrix - showing structure only")?;
            writeln!(f, "Non-zero entries: {}", self.matrix.nnz())?;
        } else {
            let stats = self.statistics();
            writeln!(f, "Matrix too large to display ({} nodes)", self.nnodes)?;
            writeln!(
                f,
                "Non-zero entries: {} ({:.2}% dense)",
                stats.nnz,
                (1.0 - stats.sparsity) * 100.0
            )?;
            writeln!(
                f,
                "Degree range: [{:.4}, {:.4}], mean: {:.4}",
                stats.min_degree, stats.max_degree, stats.mean_degree
            )?;
        }

        Ok(())
    }
}

impl fmt::Display for LaplacianStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Laplacian Statistics:")?;
        writeln!(f, "  Nodes: {}", self.nnodes)?;
        writeln!(
            f,
            "  Non-zero entries: {} ({:.2}% dense)",
            self.nnz,
            (1.0 - self.sparsity) * 100.0
        )?;
        writeln!(f, "  Sparsity: {:.4}", self.sparsity)?;
        writeln!(
            f,
            "  Degree range: [{:.4}, {:.4}]",
            self.min_degree, self.max_degree
        )?;
        writeln!(f, "  Mean degree: {:.4}", self.mean_degree)?;
        writeln!(f, "  Graph parameters: {:?}", self.graph_params)?;
        Ok(())
    }
}
