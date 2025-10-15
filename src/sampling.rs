use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt;

use log::{info, trace};
use rand::{rngs::StdRng, Rng, SeedableRng};

// ============================================================================
// TRAIT DEFINITION
// ============================================================================

pub trait InlineSampler: Send {
    fn new(target_rate: f64) -> Self where Self: Sized;
    
    fn should_keep(
        &mut self,
        row: &[f64],
        nearest_dist_sq: f64,
        centroids_count: usize,
        max_centroids: usize,
    ) -> bool;

    fn get_stats(&self) -> (usize, usize);

    /// Get a display name for the sampler
    fn name(&self) -> &str;
}

// ============================================================================
// ENUM FOR DYNAMIC DISPATCH
// ============================================================================

/// Enum wrapper to use different samplers with dynamic dispatch
pub enum SamplerType {
    Simple(f64),
    DensityAdaptive(f64),
}

impl SamplerType {
    pub fn new_simple(target_rate: f64) -> SimpleRandomSampler {
        SimpleRandomSampler::new(target_rate)
    }
    
    pub fn new_density_adaptive(target_rate: f64) -> DensityAdaptiveSampler {
        DensityAdaptiveSampler::new(target_rate)
    }
}

// ============================================================================
// SIMPLE RANDOM SAMPLER
// ============================================================================

pub struct SimpleRandomSampler {
    pub(crate) keep_rate: f64,
    rng: StdRng,
    pub sampled_count: AtomicUsize,
    pub discarded_count: AtomicUsize,
}

impl InlineSampler for SimpleRandomSampler {
    fn new(target_rate: f64) -> Self {
        info!("Simple random sampler with keep rate {:.1}%", target_rate * 100.0);
        Self {
            keep_rate: target_rate,
            rng: StdRng::from_os_rng(),
            sampled_count: AtomicUsize::new(0),
            discarded_count: AtomicUsize::new(0),
        }
    }
    
    fn should_keep(&mut self, _row: &[f64], _nearest_dist_sq: f64, _centroids_count: usize, _max_centroids: usize) -> bool {
        let keep = self.rng.random_range(0.0..1.0) < self.keep_rate;
        
        // Add counting here
        if keep {
            self.sampled_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.discarded_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        
        keep
    }

    fn get_stats(&self) -> (usize, usize) {
        let sampled = self.sampled_count.load(Ordering::Relaxed);
        let discarded = self.discarded_count.load(Ordering::Relaxed);
        (sampled, discarded)
    }

    fn name(&self) -> &str {
        "SimpleRandomSampler"
    }
}

unsafe impl Send for SimpleRandomSampler {}

// ============================================================================
// DENSITY-ADAPTIVE SAMPLER
// ============================================================================

pub struct DensityAdaptiveSampler {
    pub(crate) base_rate: f64,
    current_idx: usize,
    rng: StdRng,
    pub sampled_count: AtomicUsize,
    pub discarded_count: AtomicUsize,
}

impl InlineSampler for DensityAdaptiveSampler {
    fn new(target_rate: f64) -> Self {
        info!("Density-adaptive sampler with base rate {:.2}%", target_rate * 100.0);
        Self {
            base_rate: target_rate,
            current_idx: 0,
            rng: StdRng::from_os_rng(),
            sampled_count: AtomicUsize::new(0),
            discarded_count: AtomicUsize::new(0),
        }
    }
    
    fn should_keep(&mut self, _row: &[f64], nearest_dist_sq: f64, centroids_count: usize, max_centroids: usize) -> bool {
        self.current_idx += 1;
        
        let saturation = centroids_count as f64 / max_centroids as f64;
        let dist_factor = (nearest_dist_sq + 0.1).ln().max(0.0);
        let adaptive_rate = self.base_rate * (1.0 - saturation * 0.1) * (1.0 + dist_factor * 0.3);
        let adaptive_rate = adaptive_rate.clamp(0.01, 1.0);
        
        let keep = self.rng.random_range(0.0..1.0) < adaptive_rate;
        
        // Add counting here
        if keep {
            self.sampled_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.discarded_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        
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

    fn get_stats(&self) -> (usize, usize) {
        let sampled = self.sampled_count.load(Ordering::Relaxed);
        let discarded = self.discarded_count.load(Ordering::Relaxed);
        (sampled, discarded)
    }

    fn name(&self) -> &str {
        "DensityAdaptiveSampler"
    }

}

unsafe impl Send for DensityAdaptiveSampler {}


impl fmt::Display for SamplerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerType::Simple(s) => write!(f, "Simple({})", s),
            SamplerType::DensityAdaptive(s) => write!(f, "DensityAdaptive({})", s),
        }
    }
}
