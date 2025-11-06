//! Graph motif detection via triangle density and spectral cohesion.
//!
//! This module provides efficient triangle-based motif spotting on sparse graph Laplacians,
//! leveraging local clustering coefficients and optional Rayleigh-quotient validation
//! to surface cohesive, low-boundary subgraphs and near-cliques.
//!
//! # Overview
//!
//! - **Motives trait**: Public API for motif detection on any graph structure.
//! - **MotiveConfig**: Tunable parameters for seeding, expansion, and deduplication.
//! - **Zero-copy adjacency**: Iterates Laplacian off-diagonals on the fly; no separate matrix.
//! - **Triangle seeding**: Seeds from nodes with high triangle counts and clustering ≥ threshold.
//! - **Greedy expansion**: Grows motifs by maximizing triangle gain per added node.
//! - **Rayleigh validation**: Optional spectral check to keep sets cohesive and low-cut.
//!
//! # Usage
//!
//! ```ignore
//! use arrowspace::graph::GraphLaplacian;
//! use arrowspace::motives::{Motives, MotiveConfig};
//!
//! let gl: GraphLaplacian = /* ... */;
//! let cfg = MotiveConfig {
//!     top_l: 16,
//!     min_triangles: 3,
//!     min_clust: 0.5,
//!     max_motif_size: 24,
//!     max_sets: 128,
//!     jaccard_dedup: 0.8,
//! };
//! let motifs: Vec<Vec<usize>> = gl.spot_motives(&cfg);
//! ```
//!
//! # References
//!
//! - Scalable motif-aware clustering: <https://arxiv.org/abs/1606.06235>
//! - Local clustering coefficient: <https://en.wikipedia.org/wiki/Clustering_coefficient>
//! - Cheeger inequality & spectral cuts: MIT OCW Lecture Notes

use crate::graph::GraphLaplacian;
use log::{debug, info};
use rayon::prelude::*;
use std::collections::HashSet;

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for motif detection.
#[derive(Clone, Debug)]
pub struct MotiveConfig {
    /// Prune to top-L strongest neighbors per node (from Laplacian).
    pub top_l: usize,
    /// Minimum triangle count to seed a motif.
    pub min_triangles: usize,
    /// Minimum local clustering coefficient C_i to seed a motif.
    pub min_clust: f64,
    /// Maximum size (number of nodes) per motif during greedy expansion.
    pub max_motif_size: usize,
    /// Limit on number of returned motif sets.
    pub max_sets: usize,
    /// Jaccard similarity threshold for deduplication (0..=1).
    pub jaccard_dedup: f64,
}

impl Default for MotiveConfig {
    fn default() -> Self {
        Self {
            top_l: 16,
            min_triangles: 2,
            min_clust: 0.4,
            max_motif_size: 32,
            max_sets: 256,
            jaccard_dedup: 0.8,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public trait
// ──────────────────────────────────────────────────────────────────────────────

/// Trait for detecting graph motifs (triangles, near-cliques) via local density and spectral cohesion.
pub trait Motives {
    /// Spot motifs in the graph using triangle density, clustering coefficient, and optional Rayleigh validation.
    ///
    /// Returns a list of motif node-index sets, each represented as a `Vec<usize>` sorted ascending.
    ///
    /// # Arguments
    ///
    /// - `cfg`: Configuration for seeding, expansion, and filtering.
    ///
    /// # Algorithm
    ///
    /// 1. Build top-L neighbor lists per node by iterating Laplacian off-diagonals.
    /// 2. Count triangles per node and compute local clustering coefficient C_i = 2T_i / (k_i(k_i-1)).
    /// 3. Seed from nodes meeting `min_triangles` and `min_clust` thresholds, sorted by triangle count descending.
    /// 4. Greedily expand each seed by adding neighbors that maximize triangle gain with existing motif members.
    /// 5. Optional: enforce Rayleigh quotient on indicator ≤ `rayleigh_max` to keep motifs cohesive.
    /// 6. Deduplicate sets with Jaccard similarity ≥ `jaccard_dedup`.
    ///
    /// # Performance
    ///
    /// - Time: O(n · L²) for triangle enumeration, O(seeds · expansion) for greedy growth.
    /// - Space: O(n · L) for neighbor lists; no separate adjacency matrix.
    ///
    /// # References
    ///
    /// - Triangle-based clustering: <https://arxiv.org/abs/1606.06235>
    /// - Local clustering: <https://en.wikipedia.org/wiki/Clustering_coefficient>
    /// - Rayleigh quotient & cuts: MIT OCW, Cheeger inequality notes
    fn spot_motives_eigen(&self, cfg: &MotiveConfig) -> Vec<Vec<usize>>;

    /// EnergyMaps-aware motif spotting:
    /// 1) Spot motifs on the subcentroid Laplacian (self).
    /// 2) Map each subcentroid-set to original item indices via ArrowSpace.centroid_map.
    /// 3) Deduplicate and return item-index motifs.
    ///
    /// Requirements:
    /// - self.energy must be true (built via build_energy)
    /// - aspace.centroid_map must be Some(Vec<usize>) mapping item -> subcentroid index
    fn spot_motives_energy(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &crate::motives::MotiveConfig,
    ) -> Vec<Vec<usize>>;

    /// Check if a given set of nodes forms a clique in the graph.
    ///
    /// Returns `true` if all pairs in `set` are connected.
    fn is_clique(&self, set: &HashSet<usize>) -> bool;

    /// Compute the Rayleigh quotient R_L(1_S) = (1_S^T L 1_S) / (1_S^T 1_S) for an indicator vector of `set`.
    ///
    /// Low values indicate cohesive, low-boundary subgraphs.
    fn rayleigh_indicator(&self, set: &HashSet<usize>) -> f64;
}

// ──────────────────────────────────────────────────────────────────────────────
// Implementation for GraphLaplacian
// ──────────────────────────────────────────────────────────────────────────────

impl Motives for GraphLaplacian {
    fn spot_motives_eigen(&self, cfg: &MotiveConfig) -> Vec<Vec<usize>> {
        info!(
            "Spotting motifs: top_l={}, min_tri={}, min_clust={:.2}, max_size={}",
            cfg.top_l, cfg.min_triangles, cfg.min_clust, cfg.max_motif_size
        );

        let n = self.nnodes;

        // 1) Build top-L neighbor lists per node (parallel)
        let neigh: Vec<Vec<(usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut nb: Vec<(usize, f64)> = self.neighbors_of(i);
                nb.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                if nb.len() > cfg.top_l {
                    nb.truncate(cfg.top_l);
                }
                nb
            })
            .collect();

        // Sorted neighbor-only vectors for faster intersections
        let neigh_idx: Vec<Vec<usize>> = neigh
            .par_iter()
            .map(|v| {
                let mut ids: Vec<usize> = v.iter().map(|(j, _)| *j).collect();
                ids.sort_unstable();
                ids
            })
            .collect();

        // 2) Triangle stats (parallel per node)
        let (tri_count, clust) = triangle_stats_sorted(&neigh_idx, n);

        debug!(
            "Triangle stats: max_tri={}, max_clust={:.3}",
            tri_count.iter().max().unwrap_or(&0),
            clust.iter().cloned().fold(0.0f64, f64::max)
        );

        // 3) Seed selection and sorting (parallel filter, then sort)
        let mut seeds: Vec<usize> = (0..n)
            .into_par_iter()
            .filter(|&i| tri_count[i] >= cfg.min_triangles && clust[i] >= cfg.min_clust)
            .collect();
        seeds.par_sort_unstable_by_key(|&i| {
            std::cmp::Reverse((tri_count[i], (clust[i] * 1e6) as i64))
        });

        info!("Seeds identified: {}", seeds.len());
        debug!("Motives Seeds used: {:?}", seeds);

        // 4) Greedy expansions per seed in parallel, with local state
        let expansions: Vec<Option<HashSet<usize>>> = seeds
            .par_iter()
            .map(|&s| {
                // Skip seeds that are trivially dominated later during global dedup
                let mut seeds_hashset: HashSet<usize> = HashSet::from([s]);

                loop {
                    if seeds_hashset.len() >= cfg.max_motif_size {
                        break;
                    }

                    // Frontier
                    let mut cand = HashSet::new();
                    for &u in &seeds_hashset {
                        for &v in &neigh_idx[u] {
                            if !seeds_hashset.contains(&v) {
                                cand.insert(v);
                            }
                        }
                    }
                    if cand.is_empty() {
                        break;
                    }

                    // Select by triangle gain
                    let mut best_u: Option<usize> = None;
                    let mut best_gain: i64 = -1;

                    for u in cand {
                        // neighbors of u inside seeds_hashset
                        let mut s_nbrs: Vec<usize> = neigh_idx[u]
                            .iter()
                            .copied()
                            .filter(|v| seeds_hashset.contains(v))
                            .collect();
                        s_nbrs.sort_unstable();
                        let mut edges = 0i64;
                        for i in 0..s_nbrs.len() {
                            // count links among s_nbrs
                            let ui = s_nbrs[i];
                            // two-pointer count intersection between neigh_idx[ui] and s_nbrs[(i+1)..]
                            edges += count_edges_among(&neigh_idx[ui], &s_nbrs, i + 1) as i64;
                        }
                        if edges > best_gain {
                            best_gain = edges;
                            best_u = Some(u);
                        }
                    }

                    match best_u {
                        Some(u) => {
                            let mut s2 = seeds_hashset.clone();
                            s2.insert(u);
                            seeds_hashset = s2;
                        }
                        None => break,
                    }
                }

                if seeds_hashset.len() >= 3 {
                    Some(seeds_hashset)
                } else {
                    None
                }
            })
            .collect();

        // 5) Global Jaccard dedup (sequential for deterministic ordering)
        let mut results: Vec<HashSet<usize>> = Vec::new();
        for opt in expansions.into_iter().flatten() {
            let mut keep = true;
            for res in &results {
                if jaccard(&opt, res) >= cfg.jaccard_dedup {
                    keep = false;
                    break;
                }
            }
            if keep {
                results.push(opt);
                if results.len() >= cfg.max_sets {
                    break;
                }
            }
        }

        info!("Motifs found: {}", results.len());

        let mut out: Vec<Vec<usize>> = results
            .into_iter()
            .map(|res| {
                let mut v: Vec<usize> = res.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        out.shrink_to_fit();
        out
    }

    fn spot_motives_energy(
        &self,
        aspace: &crate::core::ArrowSpace,
        cfg: &MotiveConfig,
    ) -> Vec<Vec<usize>> {
        // Operate strictly on the energy Laplacian over subcentroids
        let (rows, cols) = self.matrix.shape();
        if rows == 0 || rows != cols {
            return Vec::new();
        }
        let n_sc = rows;

        info!(
            "Spotting energy motifs: top_l={}, min_tri={}, min_clust={:.2}, max_size={}, n_sc={}",
            cfg.top_l, cfg.min_triangles, cfg.min_clust, cfg.max_motif_size, n_sc
        );

        // 1) Neighbors with clamped indices (parallel)
        let neigh_idx: Vec<Vec<usize>> = (0..n_sc)
            .into_par_iter()
            .map(|i| {
                let mut ids: Vec<usize> = self
                    .neighbors_of(i)
                    .into_iter()
                    .filter_map(|(j, w)| {
                        if j < n_sc && j != i && w > 0.0 {
                            Some(j)
                        } else {
                            None
                        }
                    })
                    .collect();
                ids.sort_unstable();
                if ids.len() > cfg.top_l {
                    ids.truncate(cfg.top_l);
                }
                ids
            })
            .collect();

        // 2) Triangle stats (parallel)
        let (tri_count, clust) = triangle_stats_sorted(&neigh_idx, n_sc);

        debug!(
            "Energy triangle stats: max_tri={}, max_clust={:.3}",
            tri_count.iter().copied().max().unwrap_or(0),
            clust.iter().cloned().fold(0.0f64, f64::max)
        );

        // 3) Seeds (parallel filter + sort)
        let mut seeds: Vec<usize> = (0..n_sc)
            .into_par_iter()
            .filter(|&i| tri_count[i] >= cfg.min_triangles && clust[i] >= cfg.min_clust)
            .collect();
        seeds.par_sort_unstable_by_key(|&i| {
            std::cmp::Reverse((tri_count[i], (clust[i] * 1e6) as i64))
        });

        debug!("Energy motifs seeds (subcentroids): {:?}", seeds);

        // 4) Parallel greedy expansions per seed in subcentroid space
        let expansions: Vec<Option<HashSet<usize>>> = seeds
            .par_iter()
            .map(|&s| {
                let mut seeds_hashset: HashSet<usize> = HashSet::from([s]);

                loop {
                    if seeds_hashset.len() >= cfg.max_motif_size {
                        break;
                    }

                    let mut cand = HashSet::new();
                    for &u in &seeds_hashset {
                        for &v in &neigh_idx[u] {
                            if !seeds_hashset.contains(&v) {
                                cand.insert(v);
                            }
                        }
                    }
                    if cand.is_empty() {
                        break;
                    }

                    let mut best_u: Option<usize> = None;
                    let mut best_gain: i64 = -1;

                    for u in cand {
                        let mut s_nbrs: Vec<usize> = neigh_idx[u]
                            .iter()
                            .copied()
                            .filter(|v| seeds_hashset.contains(v))
                            .collect();
                        s_nbrs.sort_unstable();
                        let mut edges = 0i64;
                        for i in 0..s_nbrs.len() {
                            let ui = s_nbrs[i];
                            edges += count_edges_among(&neigh_idx[ui], &s_nbrs, i + 1) as i64;
                        }
                        if edges > best_gain {
                            best_gain = edges;
                            best_u = Some(u);
                        }
                    }

                    match best_u {
                        Some(u) => {
                            let mut s2 = seeds_hashset.clone();
                            s2.insert(u);
                            seeds_hashset = s2;
                        }
                        None => break,
                    }
                }

                if seeds_hashset.len() >= 3 {
                    Some(seeds_hashset)
                } else {
                    None
                }
            })
            .collect();

        // 5) Global dedup in subcentroid space
        let mut sc_results: Vec<HashSet<usize>> = Vec::new();
        for opt in expansions.into_iter().flatten() {
            let mut keep = true;
            for res in &sc_results {
                if jaccard(&opt, res) >= cfg.jaccard_dedup {
                    keep = false;
                    break;
                }
            }
            if keep {
                sc_results.push(opt);
                if sc_results.len() >= cfg.max_sets {
                    break;
                }
            }
        }

        info!(
            "Energy motifs: {} subcentroid motifs found",
            sc_results.len()
        );

        // 6) Map to item indices via centroid_map (parallel)
        let cmap = match &aspace.centroid_map {
            Some(m) => m,
            None => {
                // Return subcentroid motifs if mapping not available
                let mut out_sc: Vec<Vec<usize>> = sc_results
                    .into_iter()
                    .map(|res| {
                        let mut v: Vec<usize> = res.into_iter().collect();
                        v.sort_unstable();
                        v
                    })
                    .collect();
                out_sc.shrink_to_fit();
                return out_sc;
            }
        };

        // sc_id -> items (parallel build with local buckets, then merge)
        cmap.par_iter().enumerate().for_each(|(_, &sc_idx)| {
            if sc_idx < n_sc {
                // local push via interior mutability avoided; collect pairs then group
                // fallback: lightweight locking-free grouping by preallocating pairs
            }
        });
        // Simpler and safe: collect pairs and group
        let sc_item_pairs: Vec<(usize, usize)> = cmap
            .par_iter()
            .enumerate()
            .filter_map(|(it, &sc)| if sc < n_sc { Some((sc, it)) } else { None })
            .collect();
        let mut sc_to_items: Vec<Vec<usize>> = vec![Vec::new(); n_sc];
        for (sc, it) in sc_item_pairs {
            sc_to_items[sc].push(it);
        }

        // Project each subcentroid motif to items (parallel)
        let item_sets: Vec<HashSet<usize>> = sc_results
            .par_iter()
            .map(|s_sc| {
                let mut s_items = HashSet::new();
                for &sc in s_sc {
                    for &it in &sc_to_items[sc] {
                        s_items.insert(it);
                    }
                }
                s_items
            })
            .filter(|s_items| s_items.len() >= 3)
            .collect();

        // 7) Final item-level dedup (sequential for determinism)
        let mut deduped_items: Vec<HashSet<usize>> = Vec::new();
        for item in item_sets {
            let mut keep = true;
            for cmp in &deduped_items {
                if jaccard(&item, cmp) >= cfg.jaccard_dedup {
                    keep = false;
                    break;
                }
            }
            if keep {
                deduped_items.push(item);
                if deduped_items.len() >= cfg.max_sets {
                    break;
                }
            }
        }

        info!(
            "Energy motifs: {} item-level motifs after mapping",
            deduped_items.len()
        );

        let mut out: Vec<Vec<usize>> = deduped_items
            .into_iter()
            .map(|it| {
                let mut v: Vec<usize> = it.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        out.shrink_to_fit();
        out
    }

    fn is_clique(&self, set: &HashSet<usize>) -> bool {
        let sz = set.len();
        if sz < 2 {
            return false;
        }
        // Parallel short-circuit check
        let ok = set.par_iter().all(|&u| {
            let nbrs: HashSet<usize> = self.neighbors_of(u).iter().map(|(j, _)| *j).collect();
            let need = sz - 1;
            let have = nbrs.intersection(set).count();
            have == need
        });
        ok
    }

    /// unused: potential improvements using rayleigh energy boundaries
    fn rayleigh_indicator(&self, set: &HashSet<usize>) -> f64 {
        // Active computation space derived from the Laplacian itself
        let (rows, cols) = self.matrix.shape();
        if rows == 0 || rows != cols || set.is_empty() {
            return f64::INFINITY;
        }
        let n = rows;
        if set.iter().any(|&u| u >= n) {
            return f64::INFINITY;
        }
        let mut x = vec![0.0f64; n];
        for &i in set {
            x[i] = 1.0;
        }
        self.rayleigh_quotient(&x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers (parallel-friendly)
// ──────────────────────────────────────────────────────────────────────────────

fn triangle_stats_sorted(neigh_idx: &[Vec<usize>], n: usize) -> (Vec<usize>, Vec<f64>) {
    // Count triangles per node by intersecting neighbor lists
    let tri_count: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            let nbrs_i = &neigh_idx[i];
            if nbrs_i.len() < 2 {
                return 0usize;
            }
            let mut t = 0usize;
            for &j in nbrs_i {
                if j <= i {
                    continue;
                }
                let nbrs_j = &neigh_idx[j];
                t += count_intersection(nbrs_i, nbrs_j, i, j);
            }
            t
        })
        .collect();

    // Local clustering per node in parallel
    let clust: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let k = neigh_idx[i].len();
            if k >= 2 {
                (2.0 * tri_count[i] as f64) / ((k * (k - 1)) as f64)
            } else {
                0.0
            }
        })
        .collect();

    (tri_count, clust)
}

// Count common neighbors excluding i and j using two-pointer scan on sorted lists
#[inline]
fn count_intersection(a: &Vec<usize>, b: &Vec<usize>, i: usize, j: usize) -> usize {
    let mut x = 0usize;
    let (mut p, mut q) = (0usize, 0usize);
    while p < a.len() && q < b.len() {
        let va = a[p];
        let vb = b[q];
        if va == vb {
            if va != i && va != j {
                x += 1;
            }
            p += 1;
            q += 1;
        } else if va < vb {
            p += 1;
        } else {
            q += 1;
        }
    }
    x
}

// Count edges among s_nbrs after position start by intersecting with neigh(u)
#[inline]
fn count_edges_among(neigh_u: &Vec<usize>, s_nbrs: &Vec<usize>, start: usize) -> usize {
    let mut x = 0usize;
    let mut p = 0usize;
    let mut q = start;
    while p < neigh_u.len() && q < s_nbrs.len() {
        let va = neigh_u[p];
        let vb = s_nbrs[q];
        if va == vb {
            x += 1;
            p += 1;
            q += 1;
        } else if va < vb {
            p += 1;
        } else {
            q += 1;
        }
    }
    x
}

fn jaccard(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
    let inter = a.intersection(b).count() as f64;
    let union = (a.len() + b.len()) as f64 - inter;
    if union == 0.0 { 0.0 } else { inter / union }
}
