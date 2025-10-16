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

use std::sync::{Arc, Mutex};

use log::{debug, info, trace, warn};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::builder::ArrowSpaceBuilder;
use crate::sampling::InlineSampler;

/// Fixed seed for deterministic clustering
const CLUSTERING_SEED: u64 = 128;

/// Trait for computing optimal clustering parameters from data.
pub trait ClusteringHeuristic {
    /// Compute optimal number of clusters K, squared-distance threshold radius,
    /// and estimated intrinsic dimension from NxF data matrix.
    fn compute_optimal_k(
        &self,
        rows: &[Vec<f64>],
        n: usize,
        f: usize,
        seed_override: Option<u64>,
    ) -> (usize, f64, usize)
    where
        Self: Sync,
    {
        info!("Computing optimal K for clustering: N={}, F={}", n, f);

        let base_seed = seed_override.unwrap_or(CLUSTERING_SEED);

        info!(
            "Computing optimal K for clustering: N={}, F={} (seed={})",
            n, f, base_seed
        );

        let (k_min, k_max, id_est) = self.step1_bounds(rows, n, f, base_seed);

        let sample_size = n.min(1000);
        let sample_indices: Vec<usize> = if n > sample_size {
            let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed);
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

        let k_optimal = self.step2_calinski_harabasz(&sampled_rows, k_min, k_max, base_seed);

        let radius = self.compute_threshold_from_pilot(&sampled_rows, k_optimal, base_seed);

        (k_optimal, radius, id_est)
    }

    // Step 1: Bounds via N/F and intrinsic dimension
    fn step1_bounds(
        &self,
        rows: &[Vec<f64>],
        n: usize,
        f: usize,
        base_seed: u64,
    ) -> (usize, usize, usize) {
        let id_est = self.estimate_intrinsic_dimension(rows, n, f, base_seed);
        debug!("Intrinsic dimension estimate: {}", id_est);

        let k_min = ((n as f64 / 10.0).sqrt().ceil() as usize).max(2);

        let k_max_candidates = [f, n / 10, 5 * id_est, (n as f64).powf(0.5) as usize];

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
    fn estimate_intrinsic_dimension(
        &self,
        rows: &[Vec<f64>],
        n: usize,
        f: usize,
        base_seed: u64,
    ) -> usize {
        if n < 10 {
            return f.min(2);
        }

        let sample_size = n.min(500);
        let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed.wrapping_add(1));
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

    // Step 2: Calinski-Harabasz for optimal K (DETERMINISTIC with full parallelism)
    fn step2_calinski_harabasz(
        &self,
        rows: &[Vec<f64>],
        k_min: usize,
        k_max: usize,
        base_seed: u64,
    ) -> usize
    where
        Self: Sync,
    {
        let n = rows.len();
        if n < 10 {
            return k_min;
        }

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
        let k_scores: Vec<(usize, f64)> = k_candidates
            .par_iter()
            .filter(|&&k| k < n && k >= 2)
            .map(|&k| {
                let best_ch_for_k: f64 = (0..3)
                    .into_par_iter()
                    .map(|trial| {
                        // Derive unique seed: base + k*1000 + trial
                        let trial_seed = base_seed
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

                (k, penalized_score)
            })
            .collect();

        // DETERMINISTIC: Sequential max with conservative tiebreaker (prefer LARGER k)
        let (mut best_k, mut best_score) = k_scores
            .iter()
            .max_by(|(k_a, score_a), (k_b, score_b)| {
                // Primary: compare scores
                match score_a.partial_cmp(score_b) {
                    Some(std::cmp::Ordering::Greater) => std::cmp::Ordering::Greater,
                    Some(std::cmp::Ordering::Less) => std::cmp::Ordering::Less,
                    // Tiebreaker: prefer LARGER k (conservative for randomness)
                    _ => k_a.cmp(k_b),
                }
            })
            .map(|&(k, s)| (k, s))
            .unwrap_or((k_min, f64::NEG_INFINITY));

        // Fine-tune around best_k (if needed)
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

            // Parallel fine-tuning
            let fine_scores: Vec<(usize, f64)> = fine_range
                .par_iter()
                .map(|&k| {
                    let best_ch_for_k: f64 = (0..3)
                        .into_par_iter()
                        .map(|trial| {
                            // Fine-tuning seed: base + k*10000 + trial
                            let trial_seed = base_seed
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

                    (k, penalized_score)
                })
                .collect();

            // DETERMINISTIC: Sequential max for fine-tuning results
            if let Some(&(fine_k, fine_score)) =
                fine_scores.iter().max_by(|(k_a, score_a), (k_b, score_b)| {
                    match score_a.partial_cmp(score_b) {
                        Some(std::cmp::Ordering::Greater) => std::cmp::Ordering::Greater,
                        Some(std::cmp::Ordering::Less) => std::cmp::Ordering::Less,
                        // Tiebreaker: prefer LARGER k (conservative)
                        _ => k_a.cmp(k_b),
                    }
                })
            {
                if fine_score > best_score {
                    best_k = fine_k;
                    best_score = fine_score;
                }
            }
        }

        debug!("Best K={} with penalized score={:.4}", best_k, best_score);
        if best_k < k_max {
            best_k
        } else {
            k_max
        }
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
    fn compute_threshold_from_pilot(&self, rows: &[Vec<f64>], k: usize, base_seed: u64) -> f64 {
        let assignments = kmeans_lloyd(rows, k, 20, base_seed.wrapping_add(100000));
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
/// * `rows` - Input data as `Vec<Vec<f64>>` where each inner vec is a sample
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
        DenseMatrix::from_iterator(rows.into_iter().flatten().map(|x| *x), n, f, 1);

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

pub fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// Clustering and Reduction: methods used to prepare the raw data before
//  computing Laplacian.

/// Scans rows linearly, assigns to nearest centroid if within radius,
/// else creates a new cluster up to max_clusters. Outliers beyond cap are skipped.
/// Returns (centroids_matrix, assignments, sizes).
pub(crate) fn run_incremental_clustering_with_sampling(
    arrowspacebuilder: &ArrowSpaceBuilder,
    rows: &[Vec<f64>],
    nfeatures: usize,
    max_clusters: usize,
    radius: f64,
    sampler: Arc<Mutex<dyn InlineSampler>>,
) -> (DenseMatrix<f64>, Vec<Option<usize>>, Vec<usize>) {
    let nrows = rows.len();

    info!("Starting incremental clustering with inline sampling");
    debug!(
        "Parameters: max_clusters={}, radius={:.4}",
        max_clusters, radius
    );

    // Shared clustering state
    let centroids = Mutex::new(Vec::<Vec<f64>>::new());
    let counts = Mutex::new(Vec::<usize>::new());
    let assignments = Mutex::new(vec![None; nrows]);

    let process_row = |row_idx: usize| {
        let row = &rows[row_idx];

        // ============================================================
        // PHASE 1: Snapshot and decision
        // ============================================================
        let cent_snap = {
            let c = centroids.lock().unwrap();
            c.clone()
        };

        trace!(
            "Row {}: Snapshot phase - n_centroids_snapshot={}",
            row_idx,
            cent_snap.len()
        );

        // Distance from snapshot (decision basis)
        let (_snap_best_idx, snap_best_dist_sq) = if cent_snap.is_empty() {
            trace!("Row {}: Snapshot empty, setting dist²=INFINITY", row_idx);
            (0, f64::INFINITY)
        } else {
            let (idx, dist) = nearest_centroid(row, &cent_snap);
            trace!(
                "Row {}: Snapshot nearest - idx={}, dist²={:.6}",
                row_idx,
                idx,
                dist
            );
            (idx, dist)
        };

        // Sampling (if enabled)
        if arrowspacebuilder.sampling.is_some() {
            trace!("Row {}: Checking sampling filter", row_idx);
            let mut smp = sampler.lock().unwrap();
            if !smp.should_keep(row, snap_best_dist_sq, cent_snap.len(), max_clusters) {
                trace!("Row {}: REJECTED by sampling filter", row_idx);
                return;
            }
            trace!("Row {}: KEPT by sampling filter", row_idx);
        } else {
            trace!("Row {}: Sampling disabled", row_idx);
        }

        // ============================================================
        // PHASE 2: Update phase under lock
        // ============================================================
        let mut c = centroids.lock().unwrap();
        let mut k = counts.lock().unwrap();
        let mut a = assignments.lock().unwrap();

        trace!(
            "Row {}: Acquired locks - n_centroids_current={}",
            row_idx,
            c.len()
        );

        // Assert: centroids count should be >= snapshot count (monotonically increasing)
        #[cfg(test)]
        assert!(
            c.len() >= cent_snap.len(),
            "Row {}: Centroid count went backwards! snapshot={}, current={}",
            row_idx,
            cent_snap.len(),
            c.len()
        );

        // First centroid special case
        if c.is_empty() {
            trace!("Row {}: Creating FIRST centroid", row_idx);
            assert_eq!(
                cent_snap.len(),
                0,
                "Row {}: Snapshot should be empty",
                row_idx
            );
            assert_eq!(
                snap_best_dist_sq,
                f64::INFINITY,
                "Row {}: Distance should be INFINITY for empty",
                row_idx
            );

            c.push(row.clone());
            k.push(1);
            a[row_idx] = Some(0);

            trace!("Row {}: First centroid created, n_centroids=1", row_idx);
            return;
        }

        // ============================================================
        // PHASE 3: Decision based on snapshot distance
        // ============================================================
        trace!(
            "Row {}: Decision - snap_dist²={:.6}, radius²={:.6}, n_current={}, max={}",
            row_idx,
            snap_best_dist_sq,
            radius,
            c.len(),
            max_clusters
        );

        if c.len() < max_clusters && snap_best_dist_sq > (radius * 0.5) {
            // avoid overfitting the radius and falling into a single-cluster
            // CREATE NEW CLUSTER
            trace!("Row {}: CONDITION MET for new cluster: len({}) < max({}) AND dist²({:.6}) > radius²({:.6})",
                    row_idx, c.len(), max_clusters, snap_best_dist_sq, radius);

            let new_idx = c.len();

            #[cfg(test)]
            {
                // Assert: new_idx should be valid
                assert_eq!(new_idx, c.len(), "Row {}: new_idx mismatch", row_idx);
                assert!(
                    new_idx < max_clusters,
                    "Row {}: new_idx {} >= max_clusters {}",
                    row_idx,
                    new_idx,
                    max_clusters
                );
            }

            c.push(row.clone());
            k.push(1);
            a[row_idx] = Some(new_idx);

            debug!(
                "Row {}: Created centroid {}, n_centroids now={}",
                row_idx,
                new_idx,
                c.len()
            );

            // Assert: counts should match centroids
            assert_eq!(
                c.len(),
                k.len(),
                "Row {}: Centroids and counts out of sync",
                row_idx
            );
        } else if snap_best_dist_sq <= radius {
            // ASSIGN TO EXISTING CLUSTER
            trace!(
                "Row {}: ASSIGNING to existing cluster (dist²={:.6} <= radius²={:.6})",
                row_idx, snap_best_dist_sq, radius
            );

            // Recompute with current centroids for assignment
            let (best_idx, current_dist_sq) = nearest_centroid(row, &c);

            trace!("Row {}: Recomputed nearest with current - idx={}, dist²={:.6} (was {:.6} in snapshot)",
                    row_idx, best_idx, current_dist_sq, snap_best_dist_sq);

            // Assert: best_idx should be valid
            #[cfg(test)]
            assert!(
                best_idx < c.len(),
                "Row {}: best_idx {} >= n_centroids {}",
                row_idx,
                best_idx,
                c.len()
            );

            let k_old = k[best_idx] as f64;
            let k_new = k_old + 1.0;

            // Assert: count should be positive
            assert!(
                k_old > 0.0,
                "Row {}: Centroid {} has zero count",
                row_idx,
                best_idx
            );

            for j in 0..nfeatures {
                c[best_idx][j] += (row[j] - c[best_idx][j]) / k_new;
            }
            k[best_idx] += 1;
            a[row_idx] = Some(best_idx);

            debug!(
                "Row {}: Assigned to cluster {}, count now={}",
                row_idx, best_idx, k[best_idx]
            );
        } else {
            // Soft outlier policy: after we hit max_clusters, allow a relaxed assignment
            // for points that are "not too far", instead of dropping everything outright.

            // 1) Recompute distance against current centroids under the lock
            let (best_idx, current_dist_sq) = nearest_centroid(row, &c);

            // 2) Use a relaxed radius once saturated to keep more outliers
            let relax_factor = 1.5; // tune: 1.2–2.0
            let relaxed_radius = radius * relax_factor;

            if current_dist_sq <= relaxed_radius {
                // Assign as a "soft outlier" without moving the centroid (safe)
                // Alternative: tiny eta if you want some adaptation (e.g., 0.01)
                let eta = 0.0; // tune: 0.0 keeps centroids fixed for outliers
                if eta > 0.0 {
                    for j in 0..nfeatures {
                        c[best_idx][j] += eta * (row[j] - c[best_idx][j]);
                    }
                }
                // Still count the assignment for downstream stats/graph
                k[best_idx] += 1;
                a[row_idx] = Some(best_idx);

                debug!(
                    "Row {}: SOFT-ASSIGNED as outlier to cluster {} (dist²={:.6} <= relaxed {:.6})",
                    row_idx, best_idx, current_dist_sq, relaxed_radius
                );
            } else {
                // Too far even for relaxed policy → drop
                debug!(
                    "Row {}: DROPPED as outlier (dist²={:.6} > relaxed {:.6}, len={} >= max={})",
                    row_idx, current_dist_sq, relaxed_radius, c.len(), max_clusters
                );

                #[cfg(test)]
                {
                    assert_eq!(
                        c.len(),
                        max_clusters,
                        "Row {}: drop only after saturation",
                        row_idx
                    );
                    assert!(
                        current_dist_sq > relaxed_radius,
                        "Row {}: drop only if truly far",
                        row_idx
                    );
                }

                return;
            }
        }

        #[cfg(test)]
        {
            // Final assertions before releasing locks
            assert_eq!(
                c.len(),
                k.len(),
                "Row {}: Final check - centroids/counts mismatch",
                row_idx
            );
            assert!(
                c.len() <= max_clusters,
                "Row {}: Final check - exceeded max_clusters",
                row_idx
            );
        }

        trace!(
            "Row {}: Complete - n_centroids={}, n_counts={}",
            row_idx,
            c.len(),
            k.len()
        );
    };

    // Process rows in parallel (game loop all along)
    if arrowspacebuilder.deterministic_clustering {
        (0..nrows).into_iter().for_each(process_row);
    } else {
        (0..nrows).into_par_iter().for_each(process_row);
    }

    let final_centroids = centroids.into_inner().unwrap();
    let final_counts = counts.into_inner().unwrap();
    let final_assignments = assignments.into_inner().unwrap();

    // Build output matrix
    let x_out = &final_centroids.len().max(1);
    let mut flat = Vec::<f64>::with_capacity(x_out * nfeatures);
    for c in &final_centroids {
        flat.extend_from_slice(c);
    }

    let centroids_dm: DenseMatrix<f64> = if *x_out > 0 && !final_centroids.is_empty() {
        debug!(
            "Centroids:  {:?}\n : nitems->{} nfeatures->{}",
            flat, x_out, nfeatures
        );
        let dm = DenseMatrix::from_iterator(flat.iter().map(|x| *x), *x_out, nfeatures, 1);
        dm
    } else {
        warn!("No clusters created; returning zero matrix");
        let inline_sampling = arrowspacebuilder.sampling.as_ref().unwrap();
        panic!(
            "No clusters created from data, sampling: {}",
            inline_sampling
        );
        #[allow(unreachable_code)]
        DenseMatrix::from_2d_vec(&vec![vec![0.0 as f64; nfeatures]; *x_out]).unwrap()
    };

    if arrowspacebuilder.sampling.is_some() {
        let smp = sampler.lock().unwrap();
        let (sampled, discarded) = smp.get_stats();
        let sampling_ratio = sampled as f64 / nrows as f64;

        debug!(
            "Inline sampling complete: {} kept ({:.2}%), {} discarded",
            sampled,
            sampling_ratio * 100.0,
            discarded
        );
        debug!(
            "Clustering produced {} centroids from {} rows ({}% sampling)",
            final_centroids.len(),
            nrows,
            sampling_ratio * 100.0
        );
        #[cfg(not(test))]
        assert!(
            sampling_ratio > 0.325 && sampling_ratio < 0.89,
            "sampling_rate not in the interval 0.325..0.875 but {sampling_ratio}"
        );
    } else {
        debug!(
            "Clustering produced {} centroids from {} rows (100% sampling)",
            final_centroids.len(),
            nrows
        );
    }

    (centroids_dm, final_assignments, final_counts)
}

/// Linear-scan nearest centroid helper: returns (index, squared_distance).
pub(crate) fn nearest_centroid(row: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
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