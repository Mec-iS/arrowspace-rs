//! Centroid-based subgraphs and hierarchy built from ArrowSpace data.
//!
//! Invariants at each level:
//! - `laplacian.init_data`: F × N_l (features × nodes at this level).
//! - `laplacian.matrix`: F × F feature-graph Laplacian built from `init_data`.
//! - `laplacian.nnodes`: N_l, number of nodes (centroids) at this level.
//! - `root_indices[c]`: original item indices summarized by node c.

use log::info;
use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::core::ArrowSpace;
use crate::graph::{GraphLaplacian, GraphParams};
use crate::laplacian::build_laplacian_matrix;
use crate::subgraphs::{
    CentroidGraphParams, CentroidHierarchy, CentroidNode, Subgraph, SubgraphsCentroid,
};

impl SubgraphsCentroid for GraphLaplacian {
    fn spot_subg_centroids(
        &self,
        aspace: &ArrowSpace,
        params: &CentroidGraphParams,
    ) -> Vec<Subgraph> {
        info!(
            "Building centroid hierarchy: max_depth={}, min_centroids={}",
            params.max_depth, params.min_centroids
        );

        let hierarchy = CentroidHierarchy::from_centroid_graph(aspace, self, &params);

        let subgraphs = hierarchy.all_subgraphs();

        info!(
            "Extracted {} centroid subgraphs across {} levels",
            subgraphs.len(),
            hierarchy.levels.len()
        );

        subgraphs
    }

    fn build_centroid_hierarchy(
        &self,
        aspace: &ArrowSpace,
        params: CentroidGraphParams,
    ) -> CentroidHierarchy {
        info!(
            "Building full centroid hierarchy: max_depth={}, min_centroids={}",
            params.max_depth, params.min_centroids
        );

        let hierarchy = CentroidHierarchy::from_centroid_graph(aspace, self, &params);

        info!(
            "Built hierarchy with {} levels, {} total subgraphs",
            hierarchy.levels.len(),
            hierarchy.count_subgraphs()
        );

        hierarchy
    }
}

impl CentroidHierarchy {
    /// Build a centroid hierarchy from an existing centroid Laplacian and ArrowSpace.
    ///
    /// This is the internal constructor; prefer `GraphLaplacian::build_centroid_hierarchy`
    /// for the public API.
    pub fn from_centroid_graph(
        aspace: &ArrowSpace,
        gl_centroids: &GraphLaplacian,
        params: &CentroidGraphParams,
    ) -> Self {
        let centroids_fx_x = gl_centroids.init_data.clone();
        let (f_dim, x0) = centroids_fx_x.shape();

        let root_indices = build_root_indices_from_centroid_map(aspace, x0);

        let graph_params = GraphParams {
            eps: params.eps,
            k: params.k,
            topk: params.topk,
            p: params.p,
            sigma: params.sigma,
            normalise: params.normalise,
            sparsity_check: params.sparsitycheck,
        };

        let feature_gl = build_laplacian_matrix(
            centroids_fx_x.clone(),
            &graph_params,
            Some(aspace.nitems),
            false,
        );

        let (lf_rows, lf_cols) = feature_gl.matrix.shape();
        debug_assert_eq!(lf_rows, f_dim);
        debug_assert_eq!(lf_cols, f_dim);

        let root_gl = GraphLaplacian {
            init_data: centroids_fx_x,
            matrix: feature_gl.matrix,
            nnodes: x0,
            graph_params: feature_gl.graph_params.clone(),
            energy: feature_gl.energy,
        };

        let root_subgraph = Subgraph {
            node_indices: (0..x0).collect(),
            item_indices: None,
            laplacian: root_gl,
            rayleigh: None,
        };

        let parent_map: Vec<usize> = (0..x0).collect();

        let root_node = CentroidNode {
            graph: root_subgraph,
            parent_map,
            root_indices,
            children: Vec::new(),
        };

        let mut hierarchy = CentroidHierarchy {
            root: root_node.clone(),
            levels: vec![Vec::new(); params.max_depth.max(1)],
        };

        hierarchy.collect_levels(aspace, root_node, 0, params, &graph_params);
        hierarchy
    }

    fn collect_levels(
        &mut self,
        aspace: &ArrowSpace,
        node: CentroidNode,
        depth: usize,
        params: &CentroidGraphParams,
        graph_params: &GraphParams,
    ) {
        if depth >= self.levels.len() {
            self.levels.resize(depth + 1, Vec::new());
        }
        self.levels[depth].push(node.clone());

        if depth + 1 >= params.max_depth {
            return;
        }

        let x_curr = node.graph.laplacian.nnodes;
        if x_curr < params.min_centroids {
            return;
        }

        let centroids_fx_x = &node.graph.laplacian.init_data;
        let (f_dim, x_curr2) = centroids_fx_x.shape();
        debug_assert_eq!(x_curr, x_curr2);

        let centroids_xf = centroids_fx_x.transpose();
        let (labels, sub_centroids_xf) = recluster_centroids(&centroids_xf, params.k, params.seed);

        let (x_next, f_dim2) = sub_centroids_xf.shape();
        if x_next == 0 {
            return;
        }
        debug_assert_eq!(f_dim, f_dim2);

        let sub_centroids_fx_x = sub_centroids_xf.transpose();
        let next_root_indices = propagate_root_indices(&node.root_indices, &labels, x_next);

        let feature_gl = build_laplacian_matrix(
            sub_centroids_fx_x.clone(),
            graph_params,
            Some(aspace.nitems),
            false,
        );

        let (lf_rows, lf_cols) = feature_gl.matrix.shape();
        debug_assert_eq!(lf_rows, f_dim);
        debug_assert_eq!(lf_cols, f_dim);

        let sub_gl = GraphLaplacian {
            init_data: sub_centroids_fx_x,
            matrix: feature_gl.matrix,
            nnodes: x_next,
            graph_params: feature_gl.graph_params.clone(),
            energy: feature_gl.energy,
        };

        let subgraph = Subgraph {
            node_indices: (0..x_next).collect(),
            item_indices: None,
            laplacian: sub_gl,
            rayleigh: None,
        };

        let next_node = CentroidNode {
            graph: subgraph,
            parent_map: labels,
            root_indices: next_root_indices,
            children: Vec::new(),
        };

        self.collect_levels(aspace, next_node, depth + 1, params, graph_params);
    }

    /// Get all centroid nodes at a given depth (0 = root centroids).
    pub fn level(&self, depth: usize) -> &[CentroidNode] {
        self.levels.get(depth).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Total number of centroid subgraphs in the hierarchy.
    pub fn count_subgraphs(&self) -> usize {
        self.levels.iter().map(|lvl| lvl.len()).sum()
    }

    /// Flatten the hierarchy into a list of all centroid subgraphs.
    pub fn all_subgraphs(&self) -> Vec<Subgraph> {
        self.levels
            .iter()
            .flat_map(|level| level.iter().map(|node| node.graph.clone()))
            .collect()
    }
}

/// Build root item indices for the root centroid level from ArrowSpace.
///
/// Prefers `aspace.centroid_map` if populated; otherwise falls back to
/// `aspace.cluster_assignments` (which may contain `None` for unassigned items).
/// Returns a vector of length `n_root` where each entry is the list of item
/// indices assigned to that centroid.
fn build_root_indices_from_centroid_map(aspace: &ArrowSpace, n_root: usize) -> Vec<Vec<usize>> {
    let mut root_indices: Vec<Vec<usize>> = vec![Vec::new(); n_root];

    // Prefer centroid_map (definitive cluster ID per item).
    if let Some(cmap) = &aspace.centroid_map {
        for (item_idx, &cid) in cmap.iter().enumerate() {
            if cid < n_root {
                root_indices[cid].push(item_idx);
            }
        }
    } else if !aspace.cluster_assignments.is_empty() {
        // Fallback: cluster_assignments (Vec<Option<usize>>).
        for (item_idx, &assignment) in aspace.cluster_assignments.iter().enumerate() {
            if let Some(cid) = assignment {
                if cid < n_root {
                    root_indices[cid].push(item_idx);
                }
            }
            // Ignore items with assignment == None (outliers / unassigned).
        }
    } else {
        panic!("centroid_map or cluster_assignments should be set at this point");
    }

    root_indices
}

/// Propagate root item indices from parent centroids to child centroids.
///
/// `parent_root_indices[c]` is the list of items assigned to centroid c in
/// the parent level. `labels[i]` is the cluster id of centroid i in the
/// parent level, yielding `n_sub` child centroids.
///
/// The result is a vector of length `n_sub` where each entry contains all
/// items belonging to parent centroids that merged into that child.
fn propagate_root_indices(
    parent_root_indices: &[Vec<usize>],
    labels: &[usize],
    n_sub: usize,
) -> Vec<Vec<usize>> {
    let mut next_root_indices: Vec<Vec<usize>> = vec![Vec::new(); n_sub];

    for (parent_cid, items) in parent_root_indices.iter().enumerate() {
        let child_cid = labels[parent_cid];
        if child_cid < n_sub {
            next_root_indices[child_cid].extend_from_slice(items);
        }
    }

    next_root_indices
}

/// Simple centroid reclustering that does not fail.
///
/// Input:
/// - `centroids`: DenseMatrix (n × d)
/// - `k`: desired number of clusters
/// - `seed`: RNG seed (currently unused)
///
/// Output:
/// - `labels`: for each centroid i, cluster id in [0, k_eff)
/// - `new_centroids`: DenseMatrix (k_eff × d) of cluster means
pub(crate) fn recluster_centroids(
    centroids: &DenseMatrix<f64>,
    k: usize,
    _seed: Option<u64>,
) -> (Vec<usize>, DenseMatrix<f64>) {
    let (n, d) = centroids.shape();
    if n == 0 {
        return (Vec::new(), DenseMatrix::zeros(0, d));
    }

    let k_eff = k.min(n);
    let mut labels = vec![0usize; n];
    for i in 0..n {
        labels[i] = i % k_eff;
    }

    let mut sums = DenseMatrix::zeros(k_eff, d);
    let mut counts = vec![0usize; k_eff];

    for i in 0..n {
        let cid = labels[i];
        counts[cid] += 1;
        for j in 0..d {
            let val = *centroids.get((i, j));
            let cur = *sums.get((cid, j));
            sums.set((cid, j), cur + val);
        }
    }

    for cid in 0..k_eff {
        if counts[cid] > 0 {
            let c = counts[cid] as f64;
            for j in 0..d {
                let cur = *sums.get((cid, j));
                sums.set((cid, j), cur / c);
            }
        }
    }

    (labels, sums)
}
