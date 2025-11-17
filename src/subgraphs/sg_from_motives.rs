//! Motif-based subgraph extraction via the energy pipeline.
//!
//! This module provides motif detection over subcentroid graphs (energy mode)
//! and materializes subgraphs anchored at the centroid level, with mappings
//! back to original items.
//!
//! # Invariants
//!
//! For every motif subgraph `sg`:
//! 1. `sg.node_indices` are **centroid** indices in the parent's `init_data` (F × X_centroids).
//! 2. `sg.item_indices` are the original **item** indices from the motif in item space.
//! 3. `sg.laplacian.init_data` is F × X_motif (centroids for this motif).
//! 4. `sg.laplacian.matrix` is F × F (feature Laplacian for this motif).
//! 5. `sg.laplacian.nnodes` is X_motif (number of centroids in this motif).
//!
//! # Usage
//!
//! Motif-based subgraphs are only available via **energy mode**, which operates
//! on the subcentroid graph (a coarser, cheaper graph than the full item graph).

use log::{debug, info};
use rayon::prelude::*;
use smartcore::linalg::basic::{
    arrays::{Array, Array2},
    matrix::DenseMatrix,
};
use std::collections::HashSet;

use crate::core::ArrowSpace;
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::motives::Motives;
use crate::subgraphs::{Subgraph, SubgraphConfig, SubgraphsMotive};

impl Subgraph {
    /// Build a motif subgraph with centroid matrix and feature Laplacian.
    ///
    /// Algorithm:
    /// 1. Extract columns `nodes` from `parent.init_data` (F × N) → F × X local matrix.
    /// 2. Call `build_laplacian_matrix(local_fx, &parent.graph_params, Some(n_items), energy)`
    ///    to compute the F × F feature Laplacian for this motif.
    /// 3. Assemble a `GraphLaplacian` with:
    ///    - `init_data = local_fx` (F × X)
    ///    - `matrix = feature_laplacian.matrix` (F × F)
    ///    - `nnodes = X` (number of motif centroids)
    pub fn from_parent(parent: &GraphLaplacian, nodes: &[usize], n_items: Option<usize>) -> Self {
        let x_motif = nodes.len();
        let (f_parent, n_parent) = parent.init_data.shape();

        debug!(
            "Building motif subgraph with {} centroids from parent with {} centroids (F={})",
            x_motif, n_parent, f_parent
        );

        // 1. Slice init_data columns: F × X_motif (features × motif centroids).
        let sub_init_fx = extract_columns(&parent.init_data, nodes);

        // 2. Build feature Laplacian L(F × F) from this local F × X matrix.
        let params = &parent.graph_params;
        let graph_params = GraphParams {
            eps: params.eps,
            k: params.k,
            topk: params.topk,
            p: params.p,
            sigma: params.sigma,
            normalise: params.normalise,
            sparsity_check: params.sparsity_check,
        };

        let feature_gl =
            build_laplacian_matrix(sub_init_fx.clone(), &graph_params, n_items, parent.energy);

        let (lf_rows, lf_cols) = feature_gl.matrix.shape();
        debug_assert_eq!(lf_rows, f_parent);
        debug_assert_eq!(lf_cols, f_parent);

        // 3. Build local GraphLaplacian:
        let local_gl = GraphLaplacian {
            init_data: sub_init_fx,
            matrix: feature_gl.matrix,
            nnodes: x_motif,
            graph_params: feature_gl.graph_params.clone(),
            energy: feature_gl.energy,
        };

        debug!(
            "Subgraph feature Laplacian built: {} motif centroids, {} features, {} nnz",
            x_motif,
            f_parent,
            local_gl.matrix.nnz()
        );

        Subgraph {
            node_indices: nodes.to_vec(),
            item_indices: None,
            laplacian: local_gl,
            rayleigh: None,
        }
    }

    /// Compute and cache the Rayleigh indicator for this subgraph.
    pub fn compute_rayleigh(&mut self) {
        let (f_dim, _) = self.laplacian.init_data.shape();
        if f_dim == 0 {
            self.rayleigh = Some(f64::INFINITY);
            return;
        }

        let indicator = vec![1.0; f_dim];
        let r = self.laplacian.rayleigh_quotient(&indicator);

        debug!(
            "Subgraph Rayleigh cohesion (feature-space): {:.6} over {} dims",
            r, f_dim
        );
        self.rayleigh = Some(r);
    }
}

impl SubgraphsMotive for GraphLaplacian {
    fn spot_subg_motives(&self, aspace: &ArrowSpace, cfg: &SubgraphConfig) -> Vec<Subgraph> {
        info!(
            "Spotting energy subgraphs: topl={}, mintri={}, minsize={}",
            cfg.motives.top_l, cfg.motives.min_triangles, cfg.min_size
        );

        // 1. Run energy motif detection (subcentroid space → item space).
        let item_motifs: Vec<Vec<usize>> = self.spot_motives_energy(&aspace, &cfg.motives);

        info!(
            "Energy motif detection returned {} item-space candidates",
            item_motifs.len()
        );

        // 2. Map item indices → centroid indices for each motif.
        let centroid_map = if let Some(ref cmap) = aspace.centroid_map {
            cmap.as_slice()
        } else if !aspace.cluster_assignments.is_empty() {
            let temp_map: Vec<usize> = aspace
                .cluster_assignments
                .iter()
                .map(|&opt| opt.unwrap_or(0))
                .collect();
            Box::leak(temp_map.into_boxed_slice())
        } else {
            panic!("centroid_map or cluster_assignments required for energy subgraphs");
        };

        let (_f_parent, n_centroids) = self.init_data.shape();

        // Parallelize motif processing with rayon
        let mut subgraphs: Vec<Subgraph> = item_motifs
            .into_par_iter()
            .filter(|items| items.len() >= cfg.min_size)
            .filter_map(|item_nodes| {
                // Map items → centroids.
                let centroid_set: HashSet<usize> = item_nodes
                    .iter()
                    .filter_map(|&item_idx| {
                        if item_idx < centroid_map.len() {
                            let cid = centroid_map[item_idx];
                            if cid < n_centroids { Some(cid) } else { None }
                        } else {
                            None
                        }
                    })
                    .collect();

                // Skip motifs that collapse to 0 or 1 centroid.
                if centroid_set.len() < 2 {
                    debug!(
                        "Skipping motif with {} items → {} centroids (need >= 2 for graph)",
                        item_nodes.len(),
                        centroid_set.len()
                    );
                    return None;
                }

                let mut centroid_nodes: Vec<usize> = centroid_set.into_iter().collect();
                centroid_nodes.sort_unstable();

                // Build subgraph over these centroids with feature Laplacian.
                let mut sg = Subgraph::from_parent(self, &centroid_nodes, Some(aspace.nitems));

                // Store original item indices for this motif.
                sg.item_indices = Some(item_nodes);

                if cfg.rayleigh_max.is_some() {
                    sg.compute_rayleigh();
                }

                Some(sg)
            })
            .collect();

        // 3. Filter by Rayleigh threshold if specified.
        if let Some(max_r) = cfg.rayleigh_max {
            subgraphs.retain(|sg| sg.rayleigh.map(|r| r <= max_r).unwrap_or(true));

            debug!(
                "After Rayleigh filter (max={:.3}): {} subgraphs remain",
                max_r,
                subgraphs.len()
            );
        }

        info!(
            "Extracted {} energy subgraphs (min_size={}, rayleigh_max={:?})",
            subgraphs.len(),
            cfg.min_size,
            cfg.rayleigh_max
        );

        subgraphs
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Internal Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Extract specific columns from a DenseMatrix (F × N) → F × X.
fn extract_columns(matrix: &DenseMatrix<f64>, col_indices: &[usize]) -> DenseMatrix<f64> {
    let (rows, _cols) = matrix.shape();
    let n_sub = col_indices.len();

    let mut data = Vec::with_capacity(rows * n_sub);
    for row_idx in 0..rows {
        for &col_idx in col_indices {
            data.push(*matrix.get((row_idx, col_idx)));
        }
    }

    DenseMatrix::from_iterator(data.into_iter(), rows, n_sub, 1)
}
