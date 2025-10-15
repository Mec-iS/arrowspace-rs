//! # SF-GRASS: Simplified, fast spectral sparsification
//!
//! **Key optimizations:**
//! 1. Single-level coarsening only (no multilevel hierarchy)
//! 2. Fast degree-based edge scoring (no expensive spectral embedding)
//! 3. Simple greedy MST (no union-find overhead for small graphs)
//! 4. Skip sparsification for graphs already sparse enough
//! 5. Minimal allocations and cloning

use log::{debug, info};
use rayon::prelude::*;

/// Lightweight spectral sparsifier using degree-based approximation
pub struct SfGrassSparsifier {
    target_ratio: f64, // Target edge retention ratio (0.5 = keep 50% of edges)
}

impl SfGrassSparsifier {
    pub fn new() -> Self {
        Self {
            target_ratio: 0.5, // Keep 50% of edges by default
        }
    }

    /// Configure custom retention ratio
    pub fn with_target_ratio(mut self, ratio: f64) -> Self {
        self.target_ratio = ratio.clamp(0.1, 1.0);
        self
    }

    /// Main entry: sparsify adjacency graph with minimal overhead
    pub fn sparsify_graph(
        &self,
        adj_rows: &[Vec<(usize, f64)>],
        n_nodes: usize,
    ) -> Vec<Vec<(usize, f64)>> {
        debug!(
            "Sparsifying adjacency matrix for number of nodes {:?}",
            n_nodes
        );
        // **PARALLEL: Count total edges**
        let orig_edges: usize = adj_rows.par_iter().map(|r| r.len()).sum();
        let avg_degree = orig_edges as f64 / n_nodes as f64;

        // **FAST PATH: Skip if already sparse**
        if avg_degree < 10.0 {
            info!(
                "SF-GRASS: Graph already sparse (avg degree {:.1}), skipping",
                avg_degree
            );
            return adj_rows.to_vec();
        }

        info!(
            "SF-GRASS: Sparsifying {} nodes, {} edges (avg degree {:.1})",
            n_nodes, orig_edges, avg_degree
        );

        // **PARALLEL: Compute degrees**
        let degrees: Vec<usize> = adj_rows.par_iter().map(|r| r.len()).collect();

        // **PARALLEL: Score and filter edges per node**
        let sparsified: Vec<Vec<(usize, f64)>> = adj_rows
            .par_iter()
            .enumerate()
            .map(|(i, neighbors)| {
                if neighbors.is_empty() {
                    return Vec::new();
                }

                let degree_i = degrees[i];

                // **PARALLEL: Score all edges for this node**
                let mut scored_edges: Vec<(usize, f64, f64)> = neighbors
                    .par_iter()
                    .map(|&(j, weight)| {
                        // Score: weight * sqrt(degree_i * degree_j)
                        // This approximates spectral importance (hub connectivity)
                        let degree_product = (degree_i * degrees[j]) as f64;
                        let score = weight * degree_product.sqrt();
                        (j, weight, score)
                    })
                    .collect();

                // Sort by score descending (keep highest-scoring edges)
                scored_edges.sort_unstable_by(|a, b| {
                    b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Keep top edges proportional to target ratio
                // Ensure at least 1 edge per node for connectivity
                let keep_count = ((neighbors.len() as f64 * self.target_ratio).ceil() as usize)
                    .max(1)
                    .min(neighbors.len());

                scored_edges.truncate(keep_count);

                // Strip scores, return (neighbor, weight) pairs
                scored_edges.into_iter().map(|(j, w, _)| (j, w)).collect()
            })
            .collect();

        // **PARALLEL: Count final edges**
        let sparse_edges: usize = sparsified.par_iter().map(|r| r.len()).sum();
        let reduction = 100.0 * (1.0 - sparse_edges as f64 / orig_edges as f64);

        info!(
            "SF-GRASS: {} → {} edges ({:.1}% reduction)",
            orig_edges, sparse_edges, reduction
        );

        sparsified
    }
}

impl Default for SfGrassSparsifier {
    fn default() -> Self {
        Self::new()
    }
}
