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
pub trait Subgraphs {
    /// Spot subgraphs using eigen-mode motif detection.
    ///
    /// This wraps `spotmotiveseigen` and materializes each motif as a `Subgraph`
    /// with a local Laplacian and optional spectral metadata.
    fn spot_subgraphs_eigen(&self, cfg: &SubgraphConfig) -> Vec<Subgraph>;

    /// Spot subgraphs using energy-mode motif detection with item mapping.
    ///
    /// This wraps `spotmotivesenergy`, operating on subcentroids and mapping
    /// back to original item indices via `ArrowSpace.centroid_map`.
    fn spot_subgraphs_energy(&self, aspace: &ArrowSpace, cfg: &SubgraphConfig) -> Vec<Subgraph>;
}
