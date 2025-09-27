use std::fmt;

use crate::core::ArrowSpace;

use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;

// Add logging
use log::{debug, info, trace, warn};

#[derive(Debug, Clone)]
pub struct GraphParams {
    pub eps: f64,
    pub k: usize,
    pub p: f64,
    pub sigma: Option<f64>,
    pub normalise: bool, // avoid normalisation as may hinder magnitude information
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
    // store the fully computed graph laplacian
    pub matrix: DenseMatrix<f64>,
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
    /// Build a graph Laplacian matrix and transpose it so to be ready to
    ///  be used to analyse signal features
    ///
    pub fn build_laplacian_matrix(
        items: Vec<Vec<f64>>, // N×F: N items, each with F features
        eps: f64,
        k: usize,
        p: f64,
        sigma_override: Option<f64>,
        normalise: bool,
    ) -> GraphLaplacian {
        info!("Building Laplacian matrix for {} items", items.len());
        debug!(
            "Laplacian parameters: eps={}, k={}, p={}, sigma={:?}, normalise={}",
            eps, k, p, sigma_override, normalise
        );

        let result = crate::laplacian::build_laplacian_matrix(
            items,
            &GraphParams { eps, k, p, sigma: sigma_override, normalise },
        );

        info!(
            "Laplacian matrix built: {}×{} with {} nodes",
            result.matrix.shape().0,
            result.matrix.shape().1,
            result.nnodes
        );
        result
    }

    /// Build F×F feature similarity matrix
    /// This creates a graph where nodes are features and edges represent feature similarities
    /// # Arguments
    ///
    /// * `matrix` - The data from the ArrowSpace data  
    /// * `graph_params` - A graph where nodes correspond to the vector dimensions
    pub fn build_spectral_laplacian(
        mut aspace: ArrowSpace,
        graph_params: &GraphParams,
    ) -> ArrowSpace {
        info!("Building F×F spectral feature matrix");
        debug!(
            "ArrowSpace dimensions: {} features, {} items",
            aspace.nfeatures, aspace.nitems
        );
        debug!("Graph parameters: {:?}", graph_params);

        trace!("Extracting feature columns for transpose");
        let vec_transpose = {
            let mut vec_transpose = Vec::with_capacity(aspace.nfeatures);
            // Extract each column directly from the original matrix
            for col_idx in 0..aspace.nfeatures {
                let column_vec: Vec<f64> =
                    aspace.data.get_col(col_idx).iterator(0).copied().collect();

                vec_transpose.push(column_vec);
            }

            vec_transpose
        };

        debug!(
            "Transposed {} features into {} column vectors",
            aspace.nfeatures,
            vec_transpose.len()
        );

        // Build Laplacian matrix directly for features (F×F) using existing pipeline
        trace!("Building feature-to-feature Laplacian matrix");
        let tmp = crate::laplacian::build_laplacian_matrix(vec_transpose, graph_params);

        assert!(
            tmp.matrix.shape().0 == aspace.nfeatures
                && aspace.nfeatures == tmp.matrix.shape().1,
            "result should be a FxF matrix"
        );

        // Store the F×F matrix
        aspace.signals = tmp.matrix;

        info!("Built F×F feature matrix: {}×{}", aspace.nfeatures, aspace.nfeatures);
        let stats = {
            let nnz = aspace.signals.iter().filter(|&&x| x.abs() > 1e-15).count();
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

impl GraphLaplacian {
    /// Create a new GraphLaplacian from an items matrix
    /// This is used to create a graph from the transposed matrix
    /// Use `GraphFacotry::build_lambda_graph` for the full computation
    pub fn prepare_from_items(
        matrix: DenseMatrix<f64>,
        graph_params: GraphParams,
    ) -> Self {
        let nnodes = matrix.shape().1;
        debug!("Preparing GraphLaplacian from items matrix: {} nodes", nnodes);
        trace!("Transposing matrix for GraphLaplacian");
        // Transpose into features matrix
        Self { matrix: matrix.transpose(), nnodes, graph_params }
    }

    /// Get the matrix dimensions as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
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
        *self.matrix.get((i, j))
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
        self.matrix.set((i, j), value);
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
            degrees.push(*self.matrix.get((i, i)));
        }

        let (min_degree, max_degree) = degrees
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &d| {
                (min.min(d), max.max(d))
            });
        debug!("Extracted degrees: min={:.6}, max={:.6}", min_degree, max_degree);

        degrees
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
            row.push(*self.matrix.get((i, j)));
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
            col.push(*self.matrix.get((i, j)));
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

        trace!("Computing Rayleigh quotient for vector of length {}", vector.len());

        // Compute x^T L x (numerator)
        let mut numerator = 0.0;
        for i in 0..self.nnodes {
            for j in 0..self.nnodes {
                numerator += vector[i] * self.matrix.get((i, j)) * vector[j];
            }
        }

        // Compute x^T x (denominator)
        let denominator: f64 = vector.iter().map(|&x| x * x).sum();

        let result = if denominator > 1e-15 {
            numerator / denominator
        } else {
            warn!("Zero vector encountered in Rayleigh quotient computation");
            0.0 // Return 0 for zero vector
        };

        debug!(
            "Rayleigh quotient: numerator={:.6}, denominator={:.6}, result={:.6}",
            numerator, denominator, result
        );
        result
    }

    /// Compute matrix-vector multiplication: y = L * x
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

        let mut result = vec![0.0; self.nnodes];
        for (i, res) in result.iter_mut().enumerate() {
            for (j, mul) in x.iter().enumerate() {
                *res += self.matrix.get((i, j)) * mul;
            }
        }

        let result_norm = result.iter().map(|&x| x * x).sum::<f64>().sqrt();
        trace!("Matrix-vector multiplication result norm: {:.6}", result_norm);
        result
    }

    /// Check if the matrix is symmetric within tolerance
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        trace!("Checking matrix symmetry with tolerance {:.2e}", tolerance);
        let mut max_asymmetry: f64 = 0.0;
        let mut violations = 0;

        for i in 0..self.nnodes {
            for j in 0..self.nnodes {
                let diff = (self.matrix.get((i, j)) - self.matrix.get((j, i))).abs();
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
        info!("Verifying Laplacian properties with tolerance {:.2e}", tolerance);
        let mut validation = LaplacianValidation::new();

        // Check row sums (should be ≈ 0)
        let mut max_row_sum: f64 = 0.0;
        for i in 0..self.nnodes {
            let row_sum: f64 = (0..self.nnodes).map(|j| self.matrix.get((i, j))).sum();
            max_row_sum = max_row_sum.max(row_sum.abs());
            if row_sum.abs() > tolerance {
                validation.row_sum_violations.push((i, row_sum));
            }
        }
        validation.max_row_sum_error = max_row_sum;

        // Check diagonal entries (should be positive for connected components)
        for i in 0..self.nnodes {
            let diagonal = *self.matrix.get((i, i));
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
                        (self.matrix.get((i, j)) - self.matrix.get((j, i))).abs();
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
        debug!("  Row sum violations: {}", validation.row_sum_violations.len());
        debug!("  Negative diagonal entries: {}", validation.negative_diagonal.len());

        if !validation.is_valid {
            warn!("Laplacian validation failed - matrix may have numerical issues");
        }

        validation
    }

    /// Get the number of non-zero entries in the matrix
    pub fn nnz(&self) -> usize {
        trace!("Counting non-zero entries in {}×{} matrix", self.nnodes, self.nnodes);
        let mut count = 0;
        for i in 0..self.nnodes {
            for j in 0..self.nnodes {
                if self.matrix.get((i, j)).abs() > 1e-15 {
                    count += 1;
                }
            }
        }
        debug!("Found {} non-zero entries", count);
        count
    }

    /// Get the sparsity ratio (fraction of zero entries)
    pub fn sparsity(&self) -> f64 {
        let total_entries = self.nnodes * self.nnodes;
        let non_zero_entries = self.nnz();
        let sparsity = (total_entries - non_zero_entries) as f64 / total_entries as f64;
        debug!(
            "Matrix sparsity: {:.4} ({} zeros out of {} total entries)",
            sparsity,
            total_entries - non_zero_entries,
            total_entries
        );
        sparsity
    }

    /// Extract the adjacency matrix (negative of off-diagonal Laplacian entries)
    pub fn extract_adjacency(&self) -> DenseMatrix<f64> {
        info!("Extracting adjacency matrix from Laplacian");
        let mut adjacency_data = Vec::with_capacity(self.nnodes * self.nnodes);

        for i in 0..self.nnodes {
            for j in 0..self.nnodes {
                if i == j {
                    adjacency_data.push(0.0); // No self-loops
                } else {
                    // Adjacency weight = -Laplacian off-diagonal
                    adjacency_data.push(-self.matrix.get((i, j)));
                }
            }
        }

        let adjacency = DenseMatrix::from_iterator(
            adjacency_data.into_iter(),
            self.nnodes,
            self.nnodes,
            0,
        );

        debug!("Extracted adjacency matrix: {}×{}", self.nnodes, self.nnodes);
        adjacency
    }

    /// Get statistics about the Laplacian matrix
    pub fn statistics(&self) -> LaplacianStats {
        trace!("Computing Laplacian statistics");
        let degrees = self.degrees();
        let min_degree = degrees.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_degree = degrees.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let mean_degree = degrees.iter().sum::<f64>() / self.nnodes as f64;

        let nnz = self.nnz();
        let sparsity = self.sparsity();

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

    /// Get a reference to the underlying DenseMatrix
    pub fn matrix(&self) -> &DenseMatrix<f64> {
        &self.matrix
    }

    /// Get a mutable reference to the underlying DenseMatrix
    pub fn matrix_mut(&mut self) -> &mut DenseMatrix<f64> {
        &mut self.matrix
    }

    /// Clone the graph parameters
    pub fn params(&self) -> &GraphParams {
        &self.graph_params
    }
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
            // Show full matrix for small sizes
            for i in 0..self.nnodes {
                write!(f, "Row {}: [", i)?;
                for j in 0..self.nnodes {
                    write!(f, "{:8.4} ", self.matrix.get((i, j)))?;
                }
                writeln!(f, "]")?;
            }
        } else {
            // Show summary for large matrices
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
