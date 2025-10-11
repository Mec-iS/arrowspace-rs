//! Incremental clustering with optimal K heuristics for `ArrowSpace`.
//!
//! These functionalities are used to reduce the raw data input for the Laplacian
//! and spectral computations to clustered raw data.
//!
//! This module provides:
//! - `ClusteringHeuristic` trait: compute optimal K from NxF data
//! - `OptimalKHeuristic`: multi-step heuristic using intrinsic dimension,
//!   Calinski-Harabasz analysis, and adaptive thresholding
//! - Helper methods for distance computations and pilot k-means
//! - Parallel implementations for performance-critical operations
//! Incremental clustering with optimal K heuristics for `ArrowSpace`.
//!
//! **DETERMINISTIC**: All random operations use fixed seed 128 for reproducibility.

use log::{debug, info, warn};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::sync::Mutex;

/// Fixed seed for deterministic clustering
const CLUSTERING_SEED: u64 = 128;

/// Trait for computing optimal clustering parameters from data.
pub trait ClusteringHeuristic {
    /// Compute optimal number of clusters K, squared-distance threshold radius,
    /// and estimated intrinsic dimension from NxF data matrix.
    fn compute_optimal_k(&self, rows: &[Vec<f64>], n: usize, f: usize) -> (usize, f64, usize)
    where
        Self: Sync,
    {
        info!("Computing optimal K for clustering: N={}, F={}", n, f);

        // Step 1: Establish bounds
        let (k_min, k_max, id_est) = self.step1_bounds(rows, n, f);
        info!(
            "step 1 bounds: K in [{}, {}], intrinsic_dim ≈ {}",
            k_min, k_max, id_est
        );

        // Step 2: Spectral gap via Calinski-Harabasz
        let sample_size = n.min(1000);
        let sample_indices: Vec<usize> = if n > sample_size {
            let mut rng = rand::rngs::StdRng::seed_from_u64(CLUSTERING_SEED);
            let mut idxs: Vec<usize> = (0..n).collect();
            idxs.shuffle(&mut rng);
            idxs[..sample_size].to_vec()
        } else {
            (0..n).collect()
        };

        let sampled_rows: Vec<Vec<f64>> = sample_indices
            .par_iter()
            .map(|&i| rows[i].clone())
            .collect();

        let k_optimal = self.step2_calinski_harabasz(&sampled_rows, k_min, k_max);
        info!("step 2 Calinski-Harabasz suggests K = {}", k_optimal);

        // Step 3: Derive radius threshold
        let radius = self.compute_threshold_from_pilot(&sampled_rows, k_optimal);
        info!("Computed radius threshold: {:.6}", radius);

        (k_optimal, radius, id_est)
    }

    // Step 1: Bounds via N/F and intrinsic dimension
    fn step1_bounds(&self, rows: &[Vec<f64>], n: usize, f: usize) -> (usize, usize, usize) {
        let id_est = self.estimate_intrinsic_dimension(rows, n, f);
        debug!("Intrinsic dimension estimate: {}", id_est);

        let k_min = ((n as f64 / 10.0).sqrt().ceil() as usize).max(2);

        let k_max_candidates = vec![f, n / 10, 5 * id_est, (n as f64).powf(0.5) as usize];

        let k_max = k_max_candidates
            .iter()
            .copied()
            .min()
            .unwrap_or(f)
            .max(k_min + 1)
            .min(n / 2);

        (k_min, k_max, id_est)
    }

    /// Estimate intrinsic dimension via Two-NN ratio method (DETERMINISTIC).
    fn estimate_intrinsic_dimension(&self, rows: &[Vec<f64>], n: usize, f: usize) -> usize {
        if n < 10 {
            return f.min(2);
        }

        let sample_size = n.min(500);
        let mut rng = rand::rngs::StdRng::seed_from_u64(CLUSTERING_SEED.wrapping_add(1));
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let sample_indices = &indices[..sample_size];

        let ratios: Vec<f64> = sample_indices
            .par_iter()
            .filter_map(|&i| {
                let row_i = &rows[i];
                let mut dists: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d2: f64 = row_i
                            .iter()
                            .zip(&rows[j])
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (j, d2.sqrt())
                    })
                    .collect();

                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                if dists.len() >= 2 {
                    let d1 = dists[0].1;
                    let d2 = dists[1].1;
                    if d1 > 1e-12 {
                        return Some(d2 / d1);
                    }
                }
                None
            })
            .collect();

        if ratios.is_empty() {
            return f.min(3);
        }

        let mean_ratio: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let id = if mean_ratio > 1.001 {
            1.0 / mean_ratio.ln()
        } else {
            f as f64
        };
        let id_clamped = (id.round() as usize).clamp(1, f);

        debug!(
            "Two-NN mean ratio: {:.4}, estimated ID: {}",
            mean_ratio, id_clamped
        );
        id_clamped
    }

    // Step 2: Calinski-Harabasz for optimal K (DETERMINISTIC)
    fn step2_calinski_harabasz(&self, rows: &[Vec<f64>], k_min: usize, k_max: usize) -> usize
    where
        Self: Sync,
    {
        let n = rows.len();
        if n < 10 {
            return k_min;
        }

        let best = Mutex::new((k_min, f64::NEG_INFINITY));

        let k_range = k_max - k_min;
        let k_step = if k_range <= 5 {
            1
        } else if k_range <= 15 {
            2
        } else {
            3
        };

        let k_candidates: Vec<usize> = (k_min..=k_max).step_by(k_step).collect();
        debug!(
            "Testing K in range [{}, {}] with step {}",
            k_min, k_max, k_step
        );

        // Parallel evaluation with deterministic seeds
        k_candidates
            .par_iter()
            .filter(|&&k| k < n && k >= 2)
            .for_each(|&k| {
                let best_ch_for_k: f64 = (0..3)
                    .into_par_iter()
                    .map(|trial| {
                        // Derive unique seed: base + k*1000 + trial
                        let trial_seed = CLUSTERING_SEED
                            .wrapping_add((k as u64) * 1000)
                            .wrapping_add(trial as u64);

                        let assignments = kmeans_lloyd(rows, k, 20, trial_seed);
                        self.calinski_harabasz_score(rows, &assignments, k)
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                let penalty = 0.8;
                let penalized_score = best_ch_for_k - penalty * (k as f64) * (n as f64).ln();

                debug!(
                    "K={}: CH={:.4}, penalized={:.4}",
                    k, best_ch_for_k, penalized_score
                );

                let mut best_guard = best.lock().unwrap();
                if penalized_score > best_guard.1 {
                    *best_guard = (k, penalized_score);
                }
            });

        let (mut best_k, mut best_score) = *best.lock().unwrap();

        // Fine-tune around best_k
        if k_step > 1 {
            let fine_range: Vec<usize> = vec![
                best_k.saturating_sub(k_step - 1),
                best_k.saturating_sub(1),
                best_k,
                (best_k + 1).min(k_max),
                (best_k + k_step - 1).min(k_max),
            ]
            .into_iter()
            .filter(|&k| k >= k_min && k <= k_max && k < n && !k_candidates.contains(&k))
            .collect();

            fine_range.par_iter().for_each(|&k| {
                let best_ch_for_k: f64 = (0..3)
                    .into_par_iter()
                    .map(|trial| {
                        // Fine-tuning seed: base + k*10000 + trial
                        let trial_seed = CLUSTERING_SEED
                            .wrapping_add((k as u64) * 10000)
                            .wrapping_add(trial as u64);

                        let assignments = kmeans_lloyd(rows, k, 20, trial_seed);
                        self.calinski_harabasz_score(rows, &assignments, k)
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                let penalty = 0.8;
                let penalized_score = best_ch_for_k - penalty * (k as f64) * (n as f64).ln();

                debug!(
                    "K={} (fine): CH={:.4}, penalized={:.4}",
                    k, best_ch_for_k, penalized_score
                );

                let mut best_guard = best.lock().unwrap();
                if penalized_score > best_guard.1 {
                    *best_guard = (k, penalized_score);
                }
            });

            let final_best = *best.lock().unwrap();
            best_k = final_best.0;
            best_score = final_best.1;
        }

        info!("Best K={} with penalized score={:.4}", best_k, best_score);
        best_k
    }

    /// Calinski-Harabasz index (parallelized).
    fn calinski_harabasz_score(&self, rows: &[Vec<f64>], assignments: &[usize], k: usize) -> f64 {
        let n = rows.len();
        let f = rows[0].len();

        if k <= 1 || k >= n {
            return 0.0;
        }

        let global_centroid: Vec<f64> = (0..f)
            .into_par_iter()
            .map(|j| rows.iter().map(|row| row[j]).sum::<f64>() / n as f64)
            .collect();

        let centroids_counts: Vec<(Vec<f64>, usize)> = (0..k)
            .into_par_iter()
            .map(|c| {
                let mut centroid = vec![0.0; f];
                let mut count = 0;

                for (i, &cluster) in assignments.iter().enumerate() {
                    if cluster == c {
                        for j in 0..f {
                            centroid[j] += rows[i][j];
                        }
                        count += 1;
                    }
                }

                if count > 0 {
                    for val in &mut centroid {
                        *val /= count as f64;
                    }
                }
                (centroid, count)
            })
            .collect();

        let bgss: f64 = centroids_counts
            .par_iter()
            .filter(|(_, count)| *count > 0)
            .map(|(centroid, count)| {
                let d2: f64 = centroid
                    .iter()
                    .zip(&global_centroid)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (*count as f64) * d2
            })
            .sum();

        let wgss: f64 = assignments
            .par_iter()
            .enumerate()
            .filter(|(_, c)| **c < k)
            .map(|(i, &c)| {
                rows[i]
                    .iter()
                    .zip(&centroids_counts[c].0)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
            })
            .sum();

        if wgss < 1e-10 {
            return 0.0;
        }

        (bgss / (k - 1) as f64) / (wgss / (n - k) as f64)
    }

    // Step 3: Adaptive threshold (DETERMINISTIC)
    fn compute_threshold_from_pilot(&self, rows: &[Vec<f64>], k: usize) -> f64 {
        let assignments = kmeans_lloyd(rows, k, 20, CLUSTERING_SEED.wrapping_add(100000));
        let f = rows[0].len();

        let centroids_counts: Vec<(Vec<f64>, usize)> = (0..k)
            .into_par_iter()
            .map(|c| {
                let mut centroid = vec![0.0; f];
                let mut count = 0;

                for (i, &cluster) in assignments.iter().enumerate() {
                    if cluster == c {
                        for j in 0..f {
                            centroid[j] += rows[i][j];
                        }
                        count += 1;
                    }
                }

                if count > 0 {
                    for val in &mut centroid {
                        *val /= count as f64;
                    }
                }
                (centroid, count)
            })
            .collect();

        let centroids: Vec<Vec<f64>> = centroids_counts.iter().map(|(c, _)| c.clone()).collect();
        let counts: Vec<usize> = centroids_counts.iter().map(|(_, cnt)| *cnt).collect();

        let mut dists: Vec<f64> = assignments
            .par_iter()
            .enumerate()
            .filter(|(_, c)| **c < k)
            .map(|(i, &c)| {
                rows[i]
                    .iter()
                    .zip(&centroids[c])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum()
            })
            .collect();

        if dists.is_empty() {
            warn!("No distances computed; using default radius 1.0");
            return 1.0;
        }

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_90_idx = ((dists.len() as f64 * 0.9).ceil() as usize).min(dists.len() - 1);
        let percentile_90 = dists[percentile_90_idx];

        let inter_dists: Vec<f64> = (0..k)
            .into_par_iter()
            .flat_map(|i| {
                ((i + 1)..k)
                    .into_par_iter()
                    .filter({
                        let value = counts.clone();
                        move |j| value[i] > 0 && value[*j] > 0
                    })
                    .map({
                        let value = centroids.clone();
                        move |j| {
                            value[i]
                                .iter()
                                .zip(&value[j])
                                .map(|(a, b)| (a - b).powi(2))
                                .sum()
                        }
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();

        let min_inter_centroid_dist2 = if !inter_dists.is_empty() {
            inter_dists.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        } else {
            f64::INFINITY
        };

        let threshold_ratio =
            if min_inter_centroid_dist2.is_finite() && min_inter_centroid_dist2 > 0.0 {
                percentile_90 / min_inter_centroid_dist2
            } else {
                1.0
            };

        if percentile_90 < 1e-8 || threshold_ratio < 0.01 {
            debug!(
                "Within-cluster variance very small (p90={:.3e}, ratio={:.3e}); using inter-centroid fallback",
                percentile_90, threshold_ratio
            );

            if !inter_dists.is_empty() {
                let radius = min_inter_centroid_dist2 * 0.15;
                debug!("Inter-centroid fallback: radius={:.6}", radius);
                return radius.max(1e-6);
            } else {
                debug!("All centroids identical; using minimum radius");
                return 1e-6;
            }
        }

        let radius = percentile_90 * 1.5;
        debug!("Standard threshold: radius={:.6}", radius);
        radius.max(1e-6)
    }
}

/// Perform K-Means clustering using Lloyd's algorithm
///
/// # Arguments
/// * `rows` - Input data as Vec<Vec<f64>> where each inner vec is a sample
/// * `k` - Number of clusters
/// * `max_iter` - Maximum iterations for convergence
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Vector of cluster assignments (0-indexed)
pub fn kmeans_lloyd(rows: &[Vec<f64>], k: usize, max_iter: usize, seed: u64) -> Vec<usize> {
    if rows.is_empty() {
        return Vec::new();
    }

    let (n, f) = (rows.len(), rows[0].len());
    let k = k.min(n);

    // Flatten row-major data
    let x: DenseMatrix<f64> =
        DenseMatrix::from_iterator(rows.into_iter().flatten().map(|x| *x), n, f, 0);

    // Create parameters with explicit seed
    let params = KMeansParameters {
        k,
        max_iter,
        seed: Some(seed),
    };

    // Fit the model
    let km = KMeans::fit(&x, params).expect("Failed to fit K-Means model");

    // Predict cluster assignments
    let labels = km.predict(&x).expect("Failed to predict labels");

    labels
}

pub fn nearest_centroid(row: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_dist2 = f64::INFINITY;

    for (i, c) in centroids.iter().enumerate() {
        let mut d2 = 0.0;
        for (a, b) in row.iter().zip(c.iter()) {
            let diff = a - b;
            d2 += diff * diff;
        }

        if d2 < best_dist2 {
            best_dist2 = d2;
            best_idx = i;
        }
    }

    (best_idx, best_dist2)
}

pub fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
