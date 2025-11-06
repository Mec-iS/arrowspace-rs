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
//!     rayleigh_max: Some(0.5),
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
use log::{debug, info, trace};
use smartcore::linalg::basic::arrays::Array;
use std::collections::{HashMap, HashSet};

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
    /// Optional maximum Rayleigh quotient on indicator vector to accept expansion.
    pub rayleigh_max: Option<f64>,
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
            rayleigh_max: None,
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

        // 1. Build top-L neighbor lists per node from Laplacian off-diagonals
        let neigh: Vec<Vec<(usize, f64)>> = (0..n)
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

        // 2. Convert to hash-set neighbor indices for fast triangle lookup
        let neigh_sets: Vec<HashSet<usize>> = neigh
            .iter()
            .map(|v| v.iter().map(|(j, _)| *j).collect())
            .collect();

        // 3. Triangle counting and clustering coefficient
        let (tri_count, clust) = triangle_stats(&neigh_sets, n);

        debug!(
            "Triangle stats: max_tri={}, max_clust={:.3}",
            tri_count.iter().max().unwrap_or(&0),
            clust.iter().cloned().fold(0.0f64, f64::max)
        );

        // 4. Seed selection
        let mut seeds: Vec<usize> = (0..n)
            .filter(|&i| tri_count[i] >= cfg.min_triangles && clust[i] >= cfg.min_clust)
            .collect();
        seeds.sort_unstable_by_key(|&i| std::cmp::Reverse((tri_count[i], (clust[i] * 1e6) as i64)));

        info!("Seeds identified: {}", seeds.len());
        debug!("Motives Seeds used: {:?}", seeds);

        let mut results: Vec<HashSet<usize>> = Vec::new();

        // 5. Greedy expansion from each seed
        for &s in &seeds {
            if results.iter().any(|res| res.contains(&s)) {
                continue;
            }

            let mut seed_expansion = HashSet::from([s]);

            loop {
                if seed_expansion.len() >= cfg.max_motif_size {
                    break;
                }

                // Frontier N(seed_expansion)
                let mut cand = HashSet::new();
                for &u in &seed_expansion {
                    for &v in &neigh_sets[u] {
                        if !seed_expansion.contains(&v) {
                            cand.insert(v);
                        }
                    }
                }
                if cand.is_empty() {
                    break;
                }

                // Select candidate with max triangle gain
                let mut best_u: Option<usize> = None;
                let mut best_gain: i64 = -1;

                for u in cand {
                    let s_nbrs: Vec<usize> = neigh_sets[u]
                        .intersection(&seed_expansion)
                        .copied()
                        .collect();
                    let mut edges = 0i64;
                    for i in 0..s_nbrs.len() {
                        for j in (i + 1)..s_nbrs.len() {
                            if neigh_sets[s_nbrs[i]].contains(&s_nbrs[j]) {
                                edges += 1;
                            }
                        }
                    }
                    if edges > best_gain {
                        best_gain = edges;
                        best_u = Some(u);
                    }
                }

                match best_u {
                    Some(u) => {
                        let mut s2 = seed_expansion.clone();
                        s2.insert(u);

                        // Optional Rayleigh acceptance
                        if let Some(rmax) = cfg.rayleigh_max {
                            let r_after = self.rayleigh_indicator(&s2);
                            if r_after > rmax {
                                trace!(
                                    "Rayleigh {:.4} exceeds max {:.4}, stopping expansion",
                                    r_after, rmax
                                );
                                break;
                            }
                        }

                        seed_expansion = s2;
                    }
                    None => break,
                }
            }

            // 6. Deduplicate
            let mut keep = true;
            for result in &results {
                if jaccard(&seed_expansion, result) >= cfg.jaccard_dedup {
                    keep = false;
                    break;
                }
            }
            if keep && seed_expansion.len() >= 3 {
                results.push(seed_expansion);
                if results.len() >= cfg.max_sets {
                    break;
                }
            }
        }

        info!("Motifs found: {}", results.len());

        // Convert to sorted vectors
        let mut out: Vec<Vec<usize>> = results
            .into_iter()
            .map(|seed_expansion| {
                let mut v: Vec<usize> = seed_expansion.into_iter().collect();
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
        cfg: &crate::motives::MotiveConfig,
    ) -> Vec<Vec<usize>> {
        // 0) Determine subcentroid graph size from the active energy Laplacian
        let (rows, cols) = self.matrix.shape();
        if rows == 0 || rows != cols {
            // Not a valid square Laplacian
            return Vec::new();
        }
        let n_sc = rows;

        info!(
            "Spotting energy motifs: top_l={}, min_tri={}, min_clust={:.2}, max_size={}, n_sc={}",
            cfg.top_l, cfg.min_triangles, cfg.min_clust, cfg.max_motif_size, n_sc
        );

        // 1) Build top-L neighbor lists strictly in subcentroid space, clamping indices
        let neigh: Vec<Vec<(usize, f64)>> = (0..n_sc)
            .map(|i| {
                let mut nb = self
                    .neighbors_of(i)
                    .into_iter()
                    .filter(|(j, w)| *j < n_sc && *j != i && *w > 0.0)
                    .collect::<Vec<_>>();
                nb.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                if nb.len() > cfg.top_l {
                    nb.truncate(cfg.top_l);
                }
                nb
            })
            .collect();

        // 2) Convert to set form for fast triangle checks
        let neigh_sets: Vec<HashSet<usize>> = neigh
            .iter()
            .map(|v| v.iter().map(|(j, _)| *j).collect())
            .collect();

        // 3) Triangle stats on 0..n_sc
        let (tri_count, clust) = triangle_stats(&neigh_sets, n_sc);

        debug!(
            "Energy triangle stats: max_tri={}, max_clust={:.3}",
            tri_count.iter().copied().max().unwrap_or(0),
            clust.iter().cloned().fold(0.0f64, f64::max)
        );

        // 4) Seed selection in subcentroid space
        let mut seeds: Vec<usize> = (0..n_sc)
            .filter(|&i| tri_count[i] >= cfg.min_triangles && clust[i] >= cfg.min_clust)
            .collect();
        seeds.sort_unstable_by_key(|&i| std::cmp::Reverse((tri_count[i], (clust[i] * 1e6) as i64)));

        debug!(
            "Energy motifs: seeds identified in subcentroid space: {}",
            seeds.len()
        );
        debug!("Energy motifs seeds (subcentroids): {:?}", seeds);

        // 5) Greedy expansion in subcentroid space
        let mut sc_results: Vec<HashSet<usize>> = Vec::new();

        for &s in &seeds {
            if sc_results.iter().any(|res| res.contains(&s)) {
                continue;
            }
            let mut seeds_set = HashSet::from([s]);

            loop {
                if seeds_set.len() >= cfg.max_motif_size {
                    break;
                }

                // Frontier N(S) within 0..n_sc
                let mut cand = HashSet::new();
                for &u in &seeds_set {
                    for &v in &neigh_sets[u] {
                        if v < n_sc && !seeds_set.contains(&v) {
                            cand.insert(v);
                        }
                    }
                }
                if cand.is_empty() {
                    break;
                }

                // Candidate with maximum triangle gain vs S
                let mut best_u: Option<usize> = None;
                let mut best_gain: i64 = -1;

                for u in cand {
                    // Neighbors of u inside S
                    let s_nbrs: Vec<usize> =
                        neigh_sets[u].intersection(&seeds_set).copied().collect();
                    let mut edges = 0i64;
                    for i in 0..s_nbrs.len() {
                        for j in (i + 1)..s_nbrs.len() {
                            if neigh_sets[s_nbrs[i]].contains(&s_nbrs[j]) {
                                edges += 1;
                            }
                        }
                    }
                    if edges > best_gain {
                        best_gain = edges;
                        best_u = Some(u);
                    }
                }

                match best_u {
                    Some(u) => {
                        let mut s2 = seeds_set.clone();
                        s2.insert(u);

                        // Optional Rayleigh acceptance in subcentroid space
                        if let Some(rmax) = cfg.rayleigh_max {
                            // Build indicator sized to n_sc; reject on mismatch
                            let mut x = vec![0.0f64; n_sc];
                            for &i in &s2 {
                                if i >= n_sc {
                                    x.clear();
                                    break;
                                }
                                x[i] = 1.0;
                            }
                            if x.is_empty() {
                                // bounds issue; stop growth for safety
                                break;
                            }
                            let r_after = self.rayleigh_quotient(&x);
                            if r_after > rmax {
                                trace!(
                                    "Energy motifs: Rayleigh {:.4} > {:.4}, stopping expansion",
                                    r_after, rmax
                                );
                                break;
                            }
                        }

                        seeds_set = s2;
                    }
                    None => break,
                }
            }

            // Deduplicate in subcentroid space first
            let mut keep = true;
            for sc in &sc_results {
                if jaccard(&seeds_set, sc) >= cfg.jaccard_dedup {
                    keep = false;
                    break;
                }
            }
            if keep && seeds_set.len() >= 3 {
                sc_results.push(seeds_set);
                if sc_results.len() >= cfg.max_sets {
                    break;
                }
            }
        }

        info!(
            "Energy motifs: {} subcentroid motifs found",
            sc_results.len()
        );

        // 6) Map subcentroid motifs to item indices via centroid_map
        let cmap = match &aspace.centroid_map {
            Some(m) => m,
            None => {
                // No mapping available; return subcentroid motifs as-is (optional)
                // Convert to sorted Vec<usize> for consistency
                let mut out_sc: Vec<Vec<usize>> = sc_results
                    .into_iter()
                    .map(|sc| {
                        let mut v: Vec<usize> = sc.into_iter().collect();
                        v.sort_unstable();
                        v
                    })
                    .collect();
                out_sc.shrink_to_fit();
                return out_sc;
            }
        };

        // Build inverted map: sc_id -> items
        let mut sc_to_items: HashMap<usize, Vec<usize>> = HashMap::new();
        for (item_idx, &sc_idx) in cmap.iter().enumerate() {
            if sc_idx < n_sc {
                sc_to_items.entry(sc_idx).or_default().push(item_idx);
            }
        }

        // Map each subcentroid motif to item-level set
        let mut item_sets: Vec<HashSet<usize>> = Vec::new();
        for seeds_set_sc in &sc_results {
            let mut seeds_set_items = HashSet::new();
            for &sc in seeds_set_sc {
                if let Some(bucket) = sc_to_items.get(&sc) {
                    for &it in bucket {
                        seeds_set_items.insert(it);
                    }
                }
            }
            if seeds_set_items.len() >= 3 {
                item_sets.push(seeds_set_items);
            }
        }

        // 7) Deduplicate at item level
        let mut deduped_items: Vec<HashSet<usize>> = Vec::new();
        'outer: for it in item_sets {
            for cmp in &deduped_items {
                if jaccard(&it, cmp) >= cfg.jaccard_dedup {
                    continue 'outer;
                }
            }
            deduped_items.push(it);
            if deduped_items.len() >= cfg.max_sets {
                break;
            }
        }

        info!(
            "Energy motifs: {} item-level motifs after mapping",
            deduped_items.len()
        );

        // 8) Return sorted item-index vectors
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
        for &u in set {
            let nbrs: HashSet<usize> = self.neighbors_of(u).iter().map(|(j, _)| *j).collect();
            let need = sz - 1;
            let have = nbrs.intersection(set).count();
            if have != need {
                return false;
            }
        }
        true
    }

    /// Rayleigh is useful as a final cohesion/low-boundary check, but it’s brittle as
    /// a per-step gate on k-NN graphs because even good motifs can have moderate boundary
    /// until several members are added and triangles close. Use Rayleigh as a post-filter
    /// or at coarse intervals with a calibrated threshold; let triangle density and clustering drive growth.
    fn rayleigh_indicator(&self, set: &HashSet<usize>) -> f64 {
        // Active computation space is the clustered/init_data space.
        let (rows, cols) = self.matrix.shape();
        let n = rows;
        // Invariant: Laplacian must be square and consistent with init_data.
        if rows != cols || self.init_data.shape().0 != n {
            return f64::INFINITY;
        }
        // Empty set is undefined (0/0).
        if set.is_empty() {
            return f64::INFINITY;
        }
        // Bounds-check indices for this graph space.
        if set.iter().any(|&u| u >= n) {
            return f64::INFINITY;
        }

        // Build {0,1} indicator in the same space as the Laplacian.
        let mut x = vec![0.0f64; n];
        for &i in set {
            x[i] = 1.0;
        }
        self.rayleigh_quotient(&x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: zero-copy neighbor iterator
// ──────────────────────────────────────────────────────────────────────────────

impl GraphLaplacian {
    /// Returns an iterator over (neighbor_index, weight) for node `i` by negating Laplacian off-diagonals.
    ///
    /// Since L = D - W, we have W_ij = -L_ij for i ≠ j.
    ///
    /// This materializes neighbors into a Vec to avoid sprs lifetime issues;
    /// typical degree is small (k=8-32 in kNN graphs), so overhead is minimal.
    /// Returns an iterator over neighbors of node `i`.
    pub fn neighbors_of(&self, i: usize) -> Vec<(usize, f64)> {
        match self.matrix.outer_view(i) {
            Some(row_vec) => row_vec
                .iter()
                .filter_map(|(j, &val)| {
                    if i != j {
                        let w = -val;
                        if w > 0.0 { Some((j, w)) } else { None }
                    } else {
                        None
                    }
                })
                .collect(),
            None => Vec::new(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute triangle counts and local clustering coefficients for all nodes.
///
/// Returns `(tri_count, clust)` where:
/// - `tri_count[i]` = number of triangles node i participates in
/// - `clust[i]` = local clustering coefficient C_i = 2·T_i / (k_i·(k_i-1))
fn triangle_stats(neigh: &[HashSet<usize>], n: usize) -> (Vec<usize>, Vec<f64>) {
    let mut tri_count = vec![0usize; n];
    let mut clust = vec![0.0f64; n];

    for i in 0..n {
        let deg_i = neigh[i].len();
        if deg_i < 2 {
            continue;
        }
        for &j in &neigh[i] {
            if j <= i {
                continue;
            }
            let (small, large) = if neigh[i].len() <= neigh[j].len() {
                (&neigh[i], &neigh[j])
            } else {
                (&neigh[j], &neigh[i])
            };
            for &k in small {
                if k == i || k == j {
                    continue;
                }
                if large.contains(&k) {
                    tri_count[i] += 1;
                    tri_count[j] += 1;
                    tri_count[k] += 1;
                }
            }
        }
    }

    for i in 0..n {
        let deg_i = neigh[i].len();
        if deg_i >= 2 {
            clust[i] = (2.0 * tri_count[i] as f64) / ((deg_i * (deg_i - 1)) as f64);
        }
    }

    (tri_count, clust)
}

/// Compute Jaccard similarity between two sets.
fn jaccard(a: &HashSet<usize>, b: &HashSet<usize>) -> f64 {
    let inter = a.intersection(b).count() as f64;
    let union = (a.len() + b.len()) as f64 - inter;
    if union == 0.0 { 0.0 } else { inter / union }
}
