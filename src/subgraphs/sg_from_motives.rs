//! Motif-based subgraph extraction over an existing Laplacian.
//!
//! This module extends the `Motives` trait to return fully materialized
//! `Subgraph` objects instead of bare node-index sets. Each `Subgraph` is a
//! subset of nodes from an existing `GraphLaplacian`, together with:
//! - a local centroid/subcentroid matrix X × F (`laplacian.init_data`), and
//! - a feature-space Laplacian L(F × F) (`laplacian.matrix`) built from that X × F.
//!
//! # Invariants
//!
//! For every motif subgraph `sg`:
//! 1. `sg.node_indices` are indices in the *parent* Laplacian's centroid/subcentroid space.
//! 2. `sg.laplacian.init_data` is the X × F matrix formed by taking rows
//!    `sg.node_indices` from the parent's `init_data`.
//! 3. `sg.laplacian.matrix` is an F × F feature-graph Laplacian built by
//!    calling `build_laplacian_matrix` on `sg.laplacian.init_data.transpose()`
//!    with the same `GraphParams` as the parent.
//!
//! This makes motif subgraphs consistent with centroid-based subgraphs: both
//! treat `init_data` as the local centroid matrix and `matrix` as the
//! feature-space Laplacian for that matrix.

use log::{debug, info};
use smartcore::linalg::basic::{
    arrays::{Array, Array2},
    matrix::DenseMatrix,
};
use std::collections::{HashMap, HashSet};

use crate::core::ArrowSpace;
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::motives::{MotiveConfig, Motives};
use crate::subgraphs::{Subgraph, SubgraphConfig, Subgraphs};

impl Subgraph {
    /// Build a motif subgraph with centroid/subcentroid matrix and feature Laplacian.
    ///
    /// Algorithm:
    /// 1. Slice `parent.init_data` rows at `nodes` → X × F local matrix.
    /// 2. Call `build_laplacian_matrix(local.transpose(), &parent.graph_params, Some(n_items), energy=false)`
    ///    to compute the F × F feature Laplacian for this motif.
    /// 3. Assemble a `GraphLaplacian` with `init_data = local` and `matrix = feature_laplacian.matrix`.
    ///
    /// The resulting Laplacian has consistent graph parameters with the parent and
    /// encodes the feature-space relationships for just this motif's nodes.
    pub fn from_parent(parent: &GraphLaplacian, nodes: &[usize], n_items: Option<usize>) -> Self {
        let n_sub = nodes.len();
        let (n_parent, _) = parent.init_data.shape();

        debug!(
            "Building motif subgraph with {} nodes from parent with {} nodes",
            n_sub, n_parent
        );

        // 1. Slice init_data rows: centroids/subcentroids for these nodes (X × F).
        let sub_init = extract_submatrix(&parent.init_data, nodes);

        // 2. Build feature Laplacian L(F × F) from this local centroid matrix.
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

        // Local feature matrix is F × X for build_laplacian_matrix.
        let transposed = sub_init.transpose();
        let feature_gl = build_laplacian_matrix(transposed, &graph_params, n_items, parent.energy);

        // 3. Build local GraphLaplacian:
        // - init_data: X × F local centroids
        // - matrix: F × F feature Laplacian for this motif
        let local_gl = GraphLaplacian {
            init_data: sub_init,
            matrix: feature_gl.matrix,
            nnodes: feature_gl.nnodes,
            graph_params: feature_gl.graph_params.clone(),
            energy: feature_gl.energy,
        };

        debug!(
            "Subgraph feature Laplacian built: {} features, {} nnz",
            local_gl.nnodes,
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
    ///
    /// Uses the local Laplacian and a uniform indicator vector over *features*.
    /// If you prefer Rayleigh over nodes, change the dimension accordingly.
    pub fn compute_rayleigh(&mut self) {
        let n = self.laplacian.nnodes; // number of features in local feature graph
        if n == 0 {
            self.rayleigh = Some(f64::INFINITY);
            return;
        }

        let indicator = vec![1.0; n];
        let r = self.laplacian.rayleigh_quotient(&indicator);

        debug!(
            "Subgraph Rayleigh cohesion (feature-space): {:.6} over {} dims",
            r, n
        );
        self.rayleigh = Some(r);
    }
}

impl Subgraphs for GraphLaplacian {
    fn spot_subgraphs_eigen(&self, cfg: SubgraphConfig) -> Vec<Subgraph> {
        info!(
            "Spotting eigen subgraphs: topl={}, mintri={}, minsize={}",
            cfg.motives.top_l, cfg.motives.min_triangles, cfg.min_size
        );

        // 1. Run motif detection to get node sets (centroid indices).
        let motifs: Vec<Vec<usize>> = self.spot_motives_eigen(&cfg.motives);

        info!("Motif detection returned {} candidates", motifs.len());

        // 2. Materialize each motif as a Subgraph
        let mut subgraphs: Vec<Subgraph> = motifs
            .into_iter()
            .filter(|nodes| nodes.len() >= cfg.min_size)
            .map(|nodes| {
                // For centroid-only motifs we typically pass None for n_items.
                let mut sg = Subgraph::from_parent(self, &nodes, None);

                if cfg.rayleigh_max.is_some() {
                    sg.compute_rayleigh();
                }

                sg
            })
            .collect();

        // 3. Filter by Rayleigh threshold if specified
        if let Some(max_r) = cfg.rayleigh_max {
            subgraphs.retain(|sg| sg.rayleigh.map(|r| r <= max_r).unwrap_or(true));

            debug!(
                "After Rayleigh filter (max={:.3}): {} subgraphs remain",
                max_r,
                subgraphs.len()
            );
        }

        info!(
            "Extracted {} subgraphs (min_size={}, rayleigh_max={:?})",
            subgraphs.len(),
            cfg.min_size,
            cfg.rayleigh_max
        );

        subgraphs
    }

    fn spot_subgraphs_energy(&self, aspace: &ArrowSpace, cfg: SubgraphConfig) -> Vec<Subgraph> {
        info!(
            "Spotting energy subgraphs: topl={}, mintri={}, minsize={}",
            cfg.motives.top_l, cfg.motives.min_triangles, cfg.min_size
        );

        // 1. Run energy motif detection (subcentroid space → item space)
        let item_motifs: Vec<Vec<usize>> = self.spot_motives_energy(&aspace, &cfg.motives);

        info!(
            "Energy motif detection returned {} item-space candidates",
            item_motifs.len()
        );

        // 2. Map back to subcentroid space for local Laplacian construction
        let centroid_map = aspace
            .centroid_map
            .as_ref()
            .expect("centroid_map required for energy subgraphs");

        let mut subgraphs: Vec<Subgraph> = item_motifs
            .into_iter()
            .filter(|items| items.len() >= cfg.min_size)
            .map(|item_nodes| {
                // Map items → subcentroids
                let sc_set: HashSet<usize> = item_nodes
                    .iter()
                    .filter_map(|&item_idx| {
                        if item_idx < centroid_map.len() {
                            Some(centroid_map[item_idx])
                        } else {
                            None
                        }
                    })
                    .collect();

                let mut sc_nodes: Vec<usize> = sc_set.into_iter().collect();
                sc_nodes.sort_unstable();

                // Build local Laplacian over subcentroids with feature Laplacian.
                // Here, n_items is aspace.nitems for consistency with feature graph logic.
                let mut sg = Subgraph::from_parent(self, &sc_nodes, Some(aspace.nitems));

                // Store original item indices for this motif.
                sg.item_indices = Some(item_nodes);

                if cfg.rayleigh_max.is_some() {
                    sg.compute_rayleigh();
                }

                sg
            })
            .collect();

        // 3. Filter by Rayleigh threshold if specified
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

/// Extract a submatrix by selecting specific rows from a DenseMatrix (X × F).
fn extract_submatrix(matrix: &DenseMatrix<f64>, row_indices: &[usize]) -> DenseMatrix<f64> {
    let (_rows, cols) = matrix.shape();
    let n_sub = row_indices.len();

    let mut data = Vec::with_capacity(n_sub * cols);
    for &row_idx in row_indices {
        for col_idx in 0..cols {
            data.push(*matrix.get((row_idx, col_idx)));
        }
    }

    DenseMatrix::from_iterator(data.into_iter(), n_sub, cols, 1)
}
