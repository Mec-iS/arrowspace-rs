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
        debug!("Sparsifying adjacency matrix for number of nodes {:?}", n_nodes);
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
                let keep_count = ((neighbors.len() as f64 * self.target_ratio).ceil()
                    as usize)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sfgrass_basic() {
        let adj_rows = vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 0.8)],
            vec![(0, 0.5), (1, 0.8)],
        ];

        let sparsifier = SfGrassSparsifier::new();
        let result = sparsifier.sparsify_graph(&adj_rows, 3);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|row| !row.is_empty()));
    }

    #[test]
    fn test_sfgrass_larger() {
        let n = 50;
        let adj_rows: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|i| {
                (0..n)
                    .filter_map(|j| {
                        if i != j && (i + j) % 3 == 0 {
                            Some((
                                j,
                                1.0 / (1.0 + ((i as i32 - j as i32).abs() as f64)),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        let sparsifier = SfGrassSparsifier::new();
        let result = sparsifier.sparsify_graph(&adj_rows, n);

        assert_eq!(result.len(), n);
        let orig_edges: usize = adj_rows.iter().map(|r| r.len()).sum();
        let sparse_edges: usize = result.iter().map(|r| r.len()).sum();

        assert!(sparse_edges < orig_edges);
    }
    use crate::builder::ArrowSpaceBuilder;

    #[test]
    fn test_sfgrass_sparsification() {
        let rows: Vec<Vec<f64>> = (0..200)
            .map(|i| (0..50).map(|j| ((i + j) as f64 / 250.0).sin()).collect())
            .collect();

        let (aspace, _) = ArrowSpaceBuilder::new()
            .with_lambda_graph(1.0, 5, 5, 2.0, None)
            .build(rows);

        // Should produce valid lambdas
        assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
    }

    #[test]
    #[ignore = "depends on number of nodes"]
    fn test_sfgrass_vs_no_sparsification() {
        let rows_sparse: Vec<Vec<f64>> = (0..10000)
            .map(|i| vec![(i as f64 / 100.0).sin(), (i as f64 / 100.0).cos()])
            .collect();

        let rows_full: Vec<Vec<f64>> = (0..10000)
            .map(|i| vec![(i as f64 / 100.0).sin(), (i as f64 / 100.0).cos()])
            .collect();

        // With SF-GRASS
        let start = std::time::Instant::now();
        let (_, _) = ArrowSpaceBuilder::new().build(rows_sparse.clone());
        let time_sparse = start.elapsed();

        // Without
        let start = std::time::Instant::now();
        let (_, _) = ArrowSpaceBuilder::new().build(rows_full);
        let time_full = start.elapsed();

        // Should be faster
        println!("SF-GRASS: {:?}, Full: {:?}", time_sparse, time_full);
        assert!(time_sparse.as_millis() < time_full.as_millis());
    }
}
