pub mod sg_from_centroids;
pub mod sg_from_motives;

#[cfg(test)]
mod tests;

use crate::core::ArrowSpace;
use crate::graph::GraphLaplacian;
use crate::motives::MotiveConfig;

/// Configuration for subgraph extraction.
#[derive(Clone, Debug)]
pub struct SubgraphConfig {
    /// Underlying motif detection config (reused as-is).
    pub motives: MotiveConfig,

    /// Optional maximum Rayleigh quotient filter (lower = more cohesive).
    pub rayleigh_max: Option<f64>,

    /// Minimum subgraph size (nodes) to include in results.
    pub min_size: usize,
}

impl Default for SubgraphConfig {
    fn default() -> Self {
        Self {
            motives: MotiveConfig::default(),
            rayleigh_max: None,
            min_size: 3,
        }
    }
}

/// A materialized subgraph with local structure and metadata.
#[derive(Clone, Debug)]
pub struct Subgraph {
    /// Node indices in parent Laplacian space (centroids or subcentroids).
    pub node_indices: Vec<usize>,

    /// Optional mapping to original item indices (for energy pipeline).
    pub item_indices: Option<Vec<usize>>,

    /// Local Laplacian for this subgraph (nodes = node_indices.len()).
    pub laplacian: GraphLaplacian,

    /// Cached Rayleigh cohesion indicator (lower = more cohesive).
    pub rayleigh: Option<f64>,
}

/// Trait for extracting subgraphs from a graph Laplacian.
pub trait SubgraphsMotive {
    /// Spot subgraphs using energy-mode motif detection with item mapping.
    ///
    /// This wraps `spotmotivesenergy`, operating on subcentroids and mapping
    /// back to original item indices via `ArrowSpace.centroid_map`.
    fn spot_subg_motives(&self, aspace: &ArrowSpace, cfg: &SubgraphConfig) -> Vec<Subgraph>;
}

/// Trait extension for centroid-based subgraph extraction.
pub trait SubgraphsCentroid {
    /// Extract all centroid subgraphs across hierarchy levels.
    ///
    /// Returns a flat list of subgraphs from all levels of the centroid hierarchy.
    fn spot_subg_centroids(
        &self,
        aspace: &ArrowSpace,
        params: &CentroidGraphParams,
    ) -> Vec<Subgraph>;

    /// Build the full centroid hierarchy for advanced use.
    ///
    /// Returns the complete hierarchy tree, allowing level-by-level traversal
    /// and parent-child relationships.
    fn build_centroid_hierarchy(
        &self,
        aspace: &ArrowSpace,
        params: CentroidGraphParams,
    ) -> CentroidHierarchy;
}

#[derive(Clone)]
pub struct CentroidNode {
    pub graph: Subgraph,
    pub parent_map: Vec<usize>,
    pub root_indices: Vec<Vec<usize>>,
    pub children: Vec<CentroidNode>,
}

pub struct CentroidHierarchy {
    pub root: CentroidNode,
    pub levels: Vec<Vec<CentroidNode>>,
}

#[derive(Clone)]
pub struct CentroidGraphParams {
    pub eps: f64,
    pub k: usize,
    pub topk: usize,
    pub p: f64,
    pub sigma: Option<f64>,
    pub normalise: bool,
    pub sparsitycheck: bool,
    pub seed: Option<u64>,
    pub min_centroids: usize,
    pub max_depth: usize,
}

impl Default for CentroidGraphParams {
    fn default() -> Self {
        Self {
            eps: 0.5,
            k: 16,
            topk: 16,
            p: 2.0,
            sigma: None,
            normalise: false,
            sparsitycheck: false,
            seed: None,
            min_centroids: 8,
            max_depth: 2,
        }
    }
}
