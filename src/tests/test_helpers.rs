use std::collections::BTreeMap;

use log::{debug, info, trace};
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::{CsMat, TriMat};

use crate::graph::{GraphLaplacian, GraphParams};

/// Alternative version that builds adjacency first, then converts to Laplacian
/// Useful for debugging or when you need access to the adjacency matrix
pub fn build_laplacian_matrix_with_adjacency(
    items: &[Vec<f64>],
    params: &GraphParams,
) -> (GraphLaplacian, CsMat<f64>) {
    let n_items = items.len();
    if n_items < 2 {
        panic!("Matrix too small")
    }

    info!(
        "Building Laplacian with adjacency matrix output for {} items",
        n_items
    );

    let adjacency_matrix = build_adjacency_matrix(items, params);

    debug!("Converting adjacency to Laplacian matrix");
    let mut laplacian_triplets = TriMat::new((n_items, n_items));

    for i in 0..n_items {
        let degree: f64 = adjacency_matrix
            .outer_view(i)
            .unwrap()
            .iter()
            .map(|(_, &w)| w)
            .sum();
        laplacian_triplets.add_triplet(i, i, degree);

        for (j, &weight) in adjacency_matrix.outer_view(i).unwrap().iter() {
            if i != j {
                laplacian_triplets.add_triplet(i, j, -weight);
            }
        }
    }

    let laplacian_matrix = laplacian_triplets.to_csr();
    let graph_laplacian = GraphLaplacian {
        init_data: DenseMatrix::new(0, 0, vec![], true).unwrap(),
        matrix: laplacian_matrix,
        nnodes: n_items,
        graph_params: params.clone(),
    };

    info!("Successfully built Laplacian with adjacency matrix");
    (graph_laplacian, adjacency_matrix)
}

/// if the adjaceny matrix is already available, compute its L
#[allow(dead_code)]
pub fn build_laplacian_from_adjacency(adj_rows: Vec<Vec<(usize, f64)>>) -> CsMat<f64> {
    let n = adj_rows.len();

    let sym = crate::laplacian::_symmetrise_adjancency(adj_rows, n);

    let triplets = crate::laplacian::_build_sparse_laplacian(sym, n);

    // Last step: finalise results into sparse
    triplets.to_csr()
}

/// Helper function to build just the adjacency matrix
fn build_adjacency_matrix(items: &[Vec<f64>], params: &GraphParams) -> CsMat<f64> {
    let n_items = items.len();
    debug!("Building adjacency matrix for {} items", n_items);

    let norms: Vec<f64> = items
        .iter()
        .map(|item| (item.iter().map(|&x| x * x).sum::<f64>()).sqrt())
        .collect();
    trace!("Precomputed norms for all items");

    let mut adj = vec![BTreeMap::<usize, f64>::new(); n_items];
    let sigma = params.sigma.unwrap_or_else(|| params.eps.max(1e-12));
    debug!("Using sigma={} for adjacency computation", sigma);

    for i in 0..n_items {
        let mut candidates: Vec<(usize, f64, f64)> = Vec::new();
        for j in 0..n_items {
            if i == j {
                continue;
            }

            let denom = norms[i] * norms[j];
            let cosine_sim = if denom > 1e-12 {
                let dot: f64 = items[i]
                    .iter()
                    .zip(items[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (dot / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            let distance = 1.0 - cosine_sim.max(0.0);
            if distance <= params.eps {
                let normalized_dist = distance / sigma;
                let weight = 1.0 / (1.0 + normalized_dist.powf(params.p));
                if weight > 1e-12 {
                    candidates.push((j, distance, weight));
                }
            }
        }

        candidates.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let results_k: usize = params.topk;
        if candidates.len() > results_k {
            candidates.truncate(results_k);
        }

        if i % 50 == 0 {
            trace!(
                "Item {} has {} candidates within eps threshold",
                i,
                candidates.len()
            );
        }

        for (j, _dist, weight) in candidates {
            adj[i].insert(j, weight);
        }
    }

    trace!("Symmetrizing adjacency matrix");
    for i in 0..n_items {
        let keys: Vec<_> = adj[i].keys().copied().collect();
        for j in keys {
            let w = *adj[i].get(&j).unwrap_or(&0.0);
            if w > 1e-12 {
                let back_entry = adj[j].entry(i).or_insert(0.0);
                if *back_entry < 1e-12 {
                    *back_entry = w;
                }
            }
        }
    }

    trace!("Converting to sparse CSR format");
    let mut triplets = TriMat::new((n_items, n_items));
    for (i, ad) in adj.iter().enumerate() {
        for (&j, &weight) in ad.iter() {
            if i != j && weight > 1e-12 {
                triplets.add_triplet(i, j, weight);
            }
        }
    }

    let adjacency_matrix = triplets.to_csr();
    debug!(
        "Successfully built sparse adjacency matrix with {} non-zeros",
        adjacency_matrix.nnz()
    );
    adjacency_matrix
}
