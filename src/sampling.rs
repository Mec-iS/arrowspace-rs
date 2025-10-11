use std::sync::atomic::AtomicUsize;

use log::{info, trace};
use rand::{rng, rngs::ThreadRng, Rng};

pub trait InlineSampler {
    fn new(target_rate: f64) -> Self;
    /// Decide keep/discard with only row data and clustering state
    fn should_keep(
        &mut self,
        row: &[f64],
        nearest_dist_sq: f64,
        centroids_count: usize,
        max_centroids: usize,
    ) -> bool;
}

pub struct DensityAdaptiveSampler {
    base_rate: f64,
    current_idx: usize,
    rng: ThreadRng,

    // Sampling statistics
    pub sampled_count: AtomicUsize,
    pub discarded_count: AtomicUsize,
}

impl InlineSampler for DensityAdaptiveSampler {
    fn new(target_rate: f64) -> Self {
        info!(
            "Density-adaptive sampler with base rate {:.2}%",
            target_rate * 100.0
        );
        Self {
            base_rate: target_rate,
            current_idx: 0,
            rng: rng(),

            sampled_count: AtomicUsize::new(0),
            discarded_count: AtomicUsize::new(0),
        }
    }

    /// Decides based on local centroid density
    fn should_keep(
        &mut self,
        _row: &[f64],
        nearest_dist_sq: f64,
        centroids_count: usize,
        max_centroids: usize,
    ) -> bool {
        self.current_idx += 1;

        // Adapt rate based on centroid saturation and distance
        let saturation = centroids_count as f64 / max_centroids as f64;

        // If far from existing centroids (high dist), keep with higher prob
        // If close to existing centroids (low dist), sample more aggressively
        let dist_factor = (nearest_dist_sq + 0.1).ln().max(0.0);

        // Lower saturation → keep more to build coverage
        // Higher saturation → sample more aggressively
        let adaptive_rate = self.base_rate * (1.0 - saturation * 0.1) * (1.0 + dist_factor * 0.3);
        let adaptive_rate = adaptive_rate.clamp(0.01, 1.0);

        let keep = self.rng.random::<f64>() < adaptive_rate;

        trace!(
            "Row {}: dist²={:.4}, sat={:.2}, rate={:.4}, keep={}",
            self.current_idx,
            nearest_dist_sq,
            saturation,
            adaptive_rate,
            keep
        );

        keep
    }
}

unsafe impl Send for DensityAdaptiveSampler {}
