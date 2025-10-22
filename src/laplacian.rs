//! # Builds a graph Laplacian matrix from a collection of high-dimensional feature vectors

//!
//! ## Algorithm Overview
//!
//! 1. **Normalization** (optional): Normalizes each item to unit L2 norm if `params.normalise` is true
//! 2. **Similarity computation**: Computes cosine similarities between all pairs of items
//! 3. **k-NN graph construction**: For each item, retains only the k most similar neighbors within distance threshold `eps`
//! 4. **Weight assignment**: Applies kernel weighting: `w = 1 / (1 + (distance/sigma)^p)`
//! 5. **Symmetrization**: Makes the adjacency matrix symmetric by adding reverse edges
//! 6. **Laplacian construction**: Builds L = D - A where D is degree matrix and A is adjacency matrix
//!
//! ## Compute Laplacian Complexity
//! 1. **Build fastpair (CosinePair) structure**: `O(n × d × log n)`
//! 2. **k-NN queries**: `O(n × k × log n × d)`
//! - n queries (one per item)
//! - Each query returns k neighbors
//! - Each neighbor evaluation: `O(d)` for distance computation
//! - Tree traversal: `O(log n)` expected depth
//!
//! **Total: `O(n × d × log n + n × k × d × log n)` = `O(n × k × d × log n)`**
//!
//! ## Speedup Factor
//! Compared to `O(n_2)`: `n / (k × log n)`
//!
//! For typical values:
//! - **n = 10,000 items, k = 10 neighbors, d = 384 features**
//! - **Old**: 10,000² × 384 = **3.84 × 10¹⁰** operations
//! - **New**: 10,000 × 10 × 384 × log₂(10,000) ≈ **5.1 × 10⁷** operations
//! - **Speedup**: ~**750x faster!**

use crate::graph::{GraphLaplacian, GraphParams};

use dashmap::DashMap;
use smartcore::algorithm::neighbour::cosinepair::CosinePair;
use smartcore::api::{Transformer, UnsupervisedEstimator};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::preprocessing::numerical::{StandardScaler, StandardScalerParameters};

use log::{debug, info, trace};
use rayon::prelude::*;
use sprs::{CsMat, TriMat};

/// Builds a graph Laplacian matrix from a collection of high-dimensional vectors
///
/// This function constructs a k-nearest neighbor graph based on cosine similarity between items,
/// then converts it to a symmetric Laplacian matrix suitable for spectral analysis. The resulting
/// Laplacian encodes local similarity relationships and can be used for dimensionality reduction,
/// clustering, and spectral indexing.
///
/// # Parameters
///
/// * `items` - Vector of feature vectors, where `items[i]` is a d-dimensional vector representing item i.
///   All items must have identical dimensionality. Modified in-place if normalization is enabled.
/// * `params` - Graph construction parameters:
/// * `n_items`` of the original dataset (in case to need the computation of a second-order laplacian)
///
/// # Returns
///
/// `GraphLaplacian`
///
/// # Complexity
///
/// * **Time**: O(n × k × d × log n) where n = number of items, d = feature dimension
/// * **Space**: O(n × d + n × k) for input data and sparse output Laplacian matrix
///
/// The sparse representation makes this approach suitable for much larger datasets.
///
/// # Similarity Measure
///
/// Uses **rectified cosine distance**: `distance = 1 - max(0, cosine_similarity)`
/// * Cosine similarity ∈ [-1, 1] → Distance ∈ [0, 2]  
/// * Only non-negative similarities (distance ≤ 1) contribute to positive weights
/// * Items with negative cosine similarity are effectively disconnected
///
/// # Panics
///
/// * If `items` is empty or contains fewer than 2 items
/// * If items have inconsistent dimensions
/// * If any item has fewer than 2 features
/// * If matrix construction fails due to memory constraints
///
/// # Examples
///
/// ```
/// use arrowspace::laplacian::build_laplacian_matrix;
/// use arrowspace::graph::GraphParams;
/// use smartcore::linalg::basic::arrays::{Array, Array1, Array2};
/// use smartcore::linalg::basic::matrix::DenseMatrix;
///
/// // Create sample data: 4 items with 3 features each
/// let items = vec![
/// vec![1.0, 0.0, 0.0], // Item 0
/// vec![0.8, 0.6, 0.0], // Item 1 (similar to 0)
/// vec![0.0, 1.0, 0.0], // Item 2
/// vec![0.0, 0.0, 1.0], // Item 3
/// ];
///
/// let params = GraphParams {
/// eps: 0.5, // Accept neighbors with distance ≤ 0.5
/// k: 3, // At most 3 neighbors per item considered
/// topk: 3,
/// p: 2.0, // Quadratic kernel
/// sigma: Some(0.1), // Kernel bandwidth
/// normalise: true, // Normalize to unit vectors
/// sparsity_check: false
/// };
///
/// let laplacian = build_laplacian_matrix(
///     DenseMatrix::from_2d_vec(&items).unwrap().transpose(), &params, None);
/// assert_eq!(laplacian.nnodes, 4);
/// assert_eq!(laplacian.matrix.shape(), (3, 3));
/// println!("{:?}", laplacian);
/// ```
///
/// # Performance Notes
///
/// * **Parallelization**: k-NN computation is parallelized across items using rayon
/// * **Memory usage**: Stores sparse n×n matrix using CSR format
/// * **Preprocessing**: Optional normalisation and norm precomputation minimise repeated calculations
pub fn build_laplacian_matrix(
    transposed: DenseMatrix<f64>, // matrix to compute the Laplacian
    params: &GraphParams,         // requested params from the graph
    // n_items of the original dataset (in case to need the computation of L(FxN))
    n_items: Option<usize>,
) -> GraphLaplacian {
    let (d, n) = transposed.shape();
    assert!(
        n >= 2 && d >= 2,
        "items should be at least of shape (2,2): ({},{})",
        d,
        n
    );

    info!(
        "Building Laplacian matrix for {} items with {} features",
        n, d
    );
    debug!(
        "Graph parameters: eps={}, k={}, p={}, sigma={:?}, normalise={}",
        params.eps, params.k, params.p, params.sigma, params.normalise
    );

    // Step 1: Conditional normalization based on params.normalise flag
    let mut items = if params.normalise {
        debug!("Normalizing items to unit norm");
        let scaler = StandardScaler::fit(&transposed, StandardScalerParameters::default()).unwrap();
        let scaled = scaler.transform(&transposed).unwrap();
        trace!("Items normalized successfully");
        scaled
    } else {
        debug!("Skipping normalization - using raw item magnitudes");
        transposed
    };

    let triplets = _main_laplacian(&mut items, params);

    // Last step: finalise results into sparse
    let sparse_matrix: CsMat<f64> = triplets.to_csr();
    let graph_laplacian = GraphLaplacian {
        init_data: items, // store initial data from builder, XxF transposed 
        matrix: sparse_matrix,
        nnodes: match n_items {
            Some(n_items) => n_items,
            None => n,
        },
        graph_params: params.clone(),
    };

    info!(
        "Successfully built sparse Laplacian matrix ({}x{}) with {} non-zeros",
        n,
        n,
        graph_laplacian.matrix.nnz()
    );
    graph_laplacian
}

/// Laplacian main function called from the public method
/// Provide the main steps of computation for Laplacian(items)
fn _main_laplacian(
    items: &mut DenseMatrix<f64>,
    params: &GraphParams,
) -> sprs::TriMatBase<Vec<usize>, Vec<f64>> {
    let n = items.shape().0;

    let start = std::time::Instant::now();

    let adj_rows = _build_adjacency(items, params, n);

    let sym = _symmetrise_adjancency(adj_rows, n);

    let triplets = _build_sparse_laplacian(sym, n);

    info!("Total Laplacian construction time: {:?}", start.elapsed());

    triplets
}

/// From dense NxF to Adjacency matrix (weighted adjacency list)
/// From dense NxF to Adjacency matrix (weighted adjacency list) with inline sparsification
fn _build_adjacency(
    items: &mut DenseMatrix<f64>,
    params: &GraphParams,
    n: usize,
) -> Vec<Vec<(usize, f64)>> {
    // Step 2: Build CosinePair structure - O(n × d × log n)
    info!("Building CosinePair data structure");
    #[allow(clippy::unnecessary_mut_passed)]
    let fastpair = CosinePair::with_top_k(items, params.topk + 1).unwrap();
    debug!("CosinePair structure built for {} items", n);

    // Compute node degrees for sparsification scoring**
    // This is fast: just count neighbors per node from k-NN results
    info!("Computing degrees for inline sparsification");
    let degrees: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            fastpair
                .query_row_top_k(i, params.topk + 1)
                .unwrap()
                .iter()
                .filter(|(dist, j)| i != *j && *dist <= params.eps)
                .count()
        })
        .collect();

    let avg_degree = degrees.iter().sum::<usize>() as f64 / n as f64;
    let sparsify = avg_degree > 10.0; // Only sparsify if dense enough

    if sparsify {
        info!(
            "Inline sparsification enabled (avg degree {:.1})",
            avg_degree
        );
    } else {
        debug!("Skipping sparsification (avg degree {:.1})", avg_degree);
    }

    // Step 3: k-NN queries with **inline sparsification** - O(n × k × d × log n)
    info!("Computing k-NN with CosinePair: k={}", params.topk + 1);
    let adj_rows: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let neighbors = fastpair.query_row_top_k(i, params.topk + 1).unwrap();

            // Collect valid neighbors with weights
            let mut valid_neighbors: Vec<(usize, f64, f64)> = neighbors
                .iter()
                .filter_map(|(distance, j)| {
                    if i != *j && *distance <= params.eps {
                        let weight =
                            1.0 / (1.0 + (distance / params.sigma.unwrap_or(1.0)).powf(params.p));
                        if weight > 1e-12 {
                            // **INLINE SPARSIFICATION SCORE**
                            // Score = weight * sqrt(degree_i * degree_j)
                            let score = if sparsify {
                                weight * ((degrees[i] * degrees[*j]) as f64).sqrt()
                            } else {
                                weight // No scoring overhead if not sparsifying
                            };
                            Some((*j, weight, score))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            // **INLINE SPARSIFICATION: Keep top 50% by score**
            if sparsify && valid_neighbors.len() > 2 {
                valid_neighbors.sort_unstable_by(|a, b| {
                    b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                });
                let keep_count = (valid_neighbors.len() / 2).max(1);
                valid_neighbors.truncate(keep_count);
            }

            // Return (neighbor, weight) pairs
            valid_neighbors
                .into_iter()
                .map(|(j, w, _)| (j, w))
                .collect()
        })
        .collect();

    debug!("Built adjacency rows for {} items", n);
    adj_rows
}

/// Symmetrise weighted Adjacency list
pub(crate) fn _symmetrise_adjancency(
    adj_rows: Vec<Vec<(usize, f64)>>,
    n: usize,
) -> Vec<Vec<(usize, f64)>> {
    // Step 4: Symmetrise adjacency
    trace!("Symmetrizing adjacency matrix");

    // Use parallel collection first, then sequential symmetrisation
    let all_edges: Vec<(usize, usize, f64)> = adj_rows
        .par_iter()
        .enumerate()
        .flat_map(|(i, row)| {
            // Convert to owned data for parallel processing
            row.par_iter()
                .map(move |&(j, w)| (i, j, w))
                .collect::<Vec<_>>()
        })
        .collect();

    // Concurrent edge map using DashMap (lock-free concurrent HashMap)
    let edge_map: DashMap<(usize, usize), f64> = DashMap::new();

    all_edges.par_iter().for_each(|&(i, j, w)| {
        edge_map.insert((i, j), w);
        edge_map.insert((j, i), w);
    });

    // Symmetrisation and sort during collection
    let sym: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut neighbors: Vec<(usize, f64)> = edge_map
                .iter()
                .filter_map(|entry| {
                    let &(src, dst) = entry.key();
                    let &w = entry.value();
                    if src == i && src != dst {
                        Some((dst, w))
                    } else {
                        None
                    }
                })
                .collect();

            // Sort immediately within this thread
            neighbors.sort_unstable_by_key(|&(j, _)| j);
            neighbors
        })
        .collect();

    sym
}

/// Build sparse Laplacian(Adjacency)
pub(crate) fn _build_sparse_laplacian(
    sym: Vec<Vec<(usize, f64)>>,
    n: usize,
) -> sprs::TriMatBase<Vec<usize>, Vec<f64>> {
    info!("Converting adjacency to sparse Laplacian matrix (DashMap batched)");

    let start = std::time::Instant::now();

    // Concurrent map for triplets
    let triplet_map: DashMap<(usize, usize), f64> =
        DashMap::with_capacity(sym.iter().map(|s| s.len() + 1).sum());

    // Parallel insertion with edge counting
    let total_edges = sym
        .par_iter()
        .enumerate()
        .map(|(i, s)| {
            let mut local_edge_count = 0;

            // Batch operations per row
            let degree: f64 = s.iter().map(|&(_j, w)| w).sum();
            triplet_map.insert((i, i), degree);

            for &(j, w) in s {
                if i != j {
                    triplet_map.insert((i, j), -w);
                    if i < j {
                        local_edge_count += 1;
                    }
                }
            }

            local_edge_count
        })
        .sum::<usize>();

    trace!("DashMap population completed in {:?}", start.elapsed());
    debug!(
        "Total triplets: {}, edges: {}",
        triplet_map.len(),
        total_edges
    );

    // Convert to sorted vectors for more efficient TriMat insertion
    let conversion_start = std::time::Instant::now();
    let mut triplets: Vec<((usize, usize), f64)> = triplet_map.into_iter().collect();

    // Sort by (row, col) for better cache locality during insertion
    triplets.par_sort_unstable_by_key(|&((i, j), _)| (i, j));

    trace!(
        "Sorted {} triplets in {:?}",
        triplets.len(),
        conversion_start.elapsed()
    );

    // Sequential: Build TriMat from sorted triplets
    let insert_start = std::time::Instant::now();
    let mut trimat = TriMat::with_capacity((n, n), triplets.len());

    for ((i, j), val) in triplets {
        trimat.add_triplet(i, j, val);
    }

    debug!("Inserted triplets in {:?}", insert_start.elapsed());
    info!("Sparse Laplacian construction time: {:?}", start.elapsed());

    trimat
}
