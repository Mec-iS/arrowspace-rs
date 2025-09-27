use crate::graph::{GraphLaplacian, GraphParams};

use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;

use std::collections::BTreeMap;

/// Optimized build_laplacian_matrix using efficient construction and cosine similarity
/// Returns a GraphLaplacian with DenseMatrix for compatibility with SmartCore operations
pub fn build_laplacian_matrix(
    items: Vec<Vec<f64>>,
    params: &GraphParams,
) -> GraphLaplacian {
    let n_items = items.len();
    let n_features = items[0].len();

    #[cfg(debug_assertions)]
    {
        println!("=== BUILDING OPTIMIZED LAPLACIAN MATRIX (DenseMatrix) ===");
        println!(
            "Items: {}, Features: {}, Parameters: {:?}",
            n_items, n_features, params
        );
    }

    if n_items < 2 || n_features < 2 {
        panic!("items should be at least of shape (2,2): ({},{})", n_items, n_features);
    }

    // Precompute item norms for cosine similarity (avoids recomputation)
    let norms: Vec<f64> = items
        .iter()
        .map(|item| (item.iter().map(|&x| x * x).sum::<f64>()).sqrt())
        .collect();

    // Use BTreeMap for deterministic ordering and efficient sparse construction
    let mut adj = vec![BTreeMap::<usize, f64>::new(); n_items];
    let sigma = params.sigma.unwrap_or_else(|| params.eps.max(1e-9));

    #[cfg(debug_assertions)]
    println!("Computing adjacency with cosine similarity...");

    // Build adjacency using cosine similarity with eps and k constraints
    for i in 0..n_items {
        let mut candidates: Vec<(usize, f64, f64)> = Vec::new(); // (j, distance, weight)

        for j in 0..n_items {
            if i == j {
                continue;
            }

            // Compute cosine similarity efficiently
            let denom = norms[i] * norms[j];
            let cosine_sim = if denom > 1e-15 {
                let dot: f64 =
                    items[i].iter().zip(items[j].iter()).map(|(a, b)| a * b).sum();
                (dot / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            // Convert similarity to distance: d = 1 - sim (for sim in [0,1])
            let distance = 1.0 - cosine_sim.max(0.0);

            // Apply eps constraint
            if distance <= params.eps {
                // Apply kernel weighting: w = 1 / (1 + (d/sigma)^p)
                let normalized_dist = distance / sigma;
                let weight = 1.0 / (1.0 + normalized_dist.powf(params.p));

                if weight > 1e-15 {
                    candidates.push((j, distance, weight));
                }
            }
        }

        // Apply k constraint: sort by distance (ascending) and take top k
        candidates.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        // Keep only top k neighbors
        if candidates.len() > params.k {
            candidates.truncate(params.k);
        }

        // Add to adjacency map
        for (j, _dist, weight) in candidates {
            adj[i].insert(j, weight);
        }
    }

    #[cfg(debug_assertions)]
    println!("Symmetrizing adjacency matrix...");

    // Symmetrize by union (same as original implementation)
    for i in 0..n_items {
        let keys: Vec<_> = adj[i].keys().copied().collect();
        for j in keys {
            let w = *adj[i].get(&j).unwrap_or(&0.0);
            if w > 1e-15 {
                let back_entry = adj[j].entry(i).or_insert(0.0);
                if *back_entry < 1e-15 {
                    *back_entry = w;
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    println!("Converting to DenseMatrix Laplacian...");

    // Convert to DenseMatrix format: L = D - A
    let mut laplacian_data = Vec::with_capacity(n_items * n_items);

    for i in 0..n_items {
        // Compute degree for row i
        let degree: f64 = adj[i].values().sum();

        for j in 0..n_items {
            let value = if i == j {
                // Diagonal entry: degree
                degree
            } else {
                // Off-diagonal entry: -weight (negative of adjacency)
                -adj[i].get(&j).unwrap_or(&0.0)
            };
            laplacian_data.push(value);
        }
    }

    // Create DenseMatrix from the data
    let matrix =
        DenseMatrix::from_iterator(laplacian_data.into_iter(), n_items, n_items, 0);

    GraphLaplacian { matrix, nnodes: n_items, graph_params: params.clone() }
}

/// Alternative version that builds adjacency first, then converts to Laplacian
/// Useful for debugging or when you need access to the adjacency matrix
pub fn build_laplacian_matrix_with_adjacency(
    items: &Vec<Vec<f64>>,
    params: &GraphParams,
) -> (GraphLaplacian, DenseMatrix<f64>) {
    let n_items = items.len();

    if n_items < 2 {
        panic!("Matrix too small")
    }

    // Build adjacency matrix using the same efficient algorithm
    let adjacency_matrix = build_adjacency_matrix(items, params);

    // Convert adjacency to Laplacian: L = D - A
    let mut laplacian_data = Vec::with_capacity(n_items * n_items);

    for i in 0..n_items {
        // Compute degree (sum of row i in adjacency matrix)
        let degree: f64 = (0..n_items).map(|j| adjacency_matrix.get((i, j))).sum();

        for j in 0..n_items {
            let value = if i == j {
                degree // Diagonal = degree
            } else {
                -adjacency_matrix.get((i, j)) // Off-diagonal = -adjacency
            };
            laplacian_data.push(value);
        }
    }

    let laplacian_matrix =
        DenseMatrix::from_iterator(laplacian_data.into_iter(), n_items, n_items, 0);

    let graph_laplacian = GraphLaplacian {
        matrix: laplacian_matrix,
        nnodes: n_items,
        graph_params: params.clone(),
    };

    (graph_laplacian, adjacency_matrix)
}

/// Helper function to build just the adjacency matrix
fn build_adjacency_matrix(
    items: &Vec<Vec<f64>>,
    params: &GraphParams,
) -> DenseMatrix<f64> {
    let n_items = items.len();

    // Precompute norms
    let norms: Vec<f64> = items
        .iter()
        .map(|item| (item.iter().map(|&x| x * x).sum::<f64>()).sqrt())
        .collect();

    let mut adj = vec![BTreeMap::<usize, f64>::new(); n_items];
    let sigma = params.sigma.unwrap_or_else(|| params.eps.max(1e-12));

    // Build adjacency (same logic as before)
    for i in 0..n_items {
        let mut candidates: Vec<(usize, f64, f64)> = Vec::new();

        for j in 0..n_items {
            if i == j {
                continue;
            }

            let denom = norms[i] * norms[j];
            let cosine_sim = if denom > 1e-15 {
                let dot: f64 =
                    items[i].iter().zip(items[j].iter()).map(|(a, b)| a * b).sum();
                (dot / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            let distance = 1.0 - cosine_sim.max(0.0);

            if distance <= params.eps {
                let normalized_dist = distance / sigma;
                let weight = 1.0 / (1.0 + normalized_dist.powf(params.p));

                if weight > 1e-15 {
                    candidates.push((j, distance, weight));
                }
            }
        }

        candidates.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        if candidates.len() > params.k {
            candidates.truncate(params.k);
        }

        for (j, _dist, weight) in candidates {
            adj[i].insert(j, weight);
        }
    }

    // Symmetrize
    for i in 0..n_items {
        let keys: Vec<_> = adj[i].keys().copied().collect();
        for j in keys {
            let w = *adj[i].get(&j).unwrap_or(&0.0);
            if w > 1e-15 {
                let back_entry = adj[j].entry(i).or_insert(0.0);
                if *back_entry < 1e-15 {
                    *back_entry = w;
                }
            }
        }
    }

    // Convert to DenseMatrix
    let mut adjacency_data = Vec::with_capacity(n_items * n_items);
    for i in 0..n_items {
        for j in 0..n_items {
            let weight = if i == j {
                0.0 // No self-loops
            } else {
                *adj[i].get(&j).unwrap_or(&0.0)
            };
            adjacency_data.push(weight);
        }
    }

    DenseMatrix::from_iterator(adjacency_data.into_iter(), n_items, n_items, 0)
}