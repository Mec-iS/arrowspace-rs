use crate::builder::ArrowSpaceBuilder;
use crate::energymaps::{EnergyMapsBuilder, EnergyParams};
use crate::motives::{MotiveConfig, Motives};
use crate::tests::test_data::make_gaussian_cliques;

use log::{debug, info};

#[test]
fn test_motives_basic() {
    crate::tests::init();

    // 3 near-cliques + outliers
    let rows = make_gaussian_cliques(12, 0.05, 15, 10, 42);

    // Build a denser, normalized graph to preserve triangle closures
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 14, 8, 2.0, None) // k=14, topk=8
        .with_normalisation(true)
        .with_sparsity_check(false)
        .build(rows);

    // Keep at least as many as topk; relax thresholds; disable Rayleigh for the first run
    let cfg = MotiveConfig {
        top_l: 16, // ≥ topk (if no motives is spotted, increase top_l)
        min_triangles: 2,
        min_clust: 0.4,
        max_motif_size: 24,
        max_sets: 100,
        jaccard_dedup: 0.8,
    };

    let motifs = gl.spot_motives_eigen(&cfg);
    info!("Found {} motifs:", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        debug!("  Motif {}: {:?}", i, m);
    }

    assert!(motifs.len() > 0);
}

#[test]
fn test_motives_basic_2() {
    crate::tests::init();

    // 3 near-cliques of 24 points each + 27 outliers = 99 total, 10D
    let rows = make_gaussian_cliques(24, 0.05, 27, 10, 42);

    // Build a denser, normalized graph to preserve triangle closures at N=99
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.3, 18, 12, 2.0, None) // slightly denser intra-group
        .with_sparsity_check(false)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None)
        .with_seed(42)
        .build(rows);

    // Keep at least as many neighbors as topk; thresholds tuned for N=99 near-cliques
    let cfg = MotiveConfig {
        top_l: 16, // ≥ topk to avoid double pruning
        min_triangles: 3,
        min_clust: 0.45,
        max_motif_size: 32,
        max_sets: 100,
        jaccard_dedup: 0.7,
    };

    let motifs = gl.spot_motives_eigen(&cfg);
    info!("Found {} motifs:", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        debug!("  Motif {}: {:?}", i, m);
    }

    assert!(!motifs.is_empty(), "Expected motifs at N=99");
}

#[test]
fn test_motives_energy_basic() {
    crate::tests::init();

    // 3 near-cliques + outliers
    let rows = make_gaussian_cliques(12, 0.05, 15, 10, 42);

    let p = EnergyParams::default();
    // Mild diffusion and balanced weights tend to give usable local density
    // p.steps, p.neighbork, etc. can be tuned in your codebase if exposed

    // Build Energy-only ArrowSpace and GraphLaplacian
    // Note: build_energy requires dimensionality reduction enabled in this codebase.
    let (aspace, gl_energy) = ArrowSpaceBuilder::new()
        .with_seed(12345)
        .with_lambda_graph(0.4, 14, 8, 2.0, None) // k=14, topk=8
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None)
        .build_energy(rows, p);

    // Keep at least as many neighbors as the energy graph retains; avoid double-pruning
    let cfg = MotiveConfig {
        top_l: 16,        // keep neighbors available from energy Laplacian
        min_triangles: 2, // permissive seeding
        min_clust: 0.4,   // moderate clustering threshold
        max_motif_size: 24,
        max_sets: 100,
        jaccard_dedup: 0.8,
    };

    let motifs = gl_energy.spot_motives_energy(&aspace, &cfg);

    info!("Found {} motifs (energy):", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        debug!("  Motif {}: {:?}", i, m);
    }

    assert!(motifs.len() > 0);
}

#[test]
fn test_motives_eigen_vs_energy_consistency() {
    // Deterministic logs and RNG
    crate::tests::init();

    // Synthetic data: 3 near-cliques + outliers
    let rows = make_gaussian_cliques(12, 0.04, 12, 10, 1337);

    // Common motif config
    let cfg = MotiveConfig {
        top_l: 16,
        min_triangles: 2, // relaxed for N_sc≈10
        min_clust: 0.35,  // allow seeding with fewer closures
        max_motif_size: 24,
        max_sets: 64,
        jaccard_dedup: 0.8,
    };

    // -----------------------
    // EigenMaps pipeline
    // -----------------------
    let (_, gl_eig) = ArrowSpaceBuilder::new()
        .with_seed(42)
        .with_lambda_graph(0.4, 14, 8, 2.0, None) // k=14, topk=8
        .with_sparsity_check(false)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None)
        .build(rows.clone());

    // Compute motifs directly in item space (EigenMaps path)
    let motifs_eig = gl_eig.spot_motives_eigen(&cfg);
    assert!(
        !motifs_eig.is_empty(),
        "EigenMaps returned 0 motifs; expected planted clusters"
    );

    // -----------------------
    // EnergyMaps pipeline
    // -----------------------
    let p = EnergyParams::default();

    let (aspace_eng, gl_eng) = ArrowSpaceBuilder::new()
        .with_seed(42)
        .with_lambda_graph(0.35, 18, 10, 2.0, None) // denser k/topk
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None)
        .build_energy(rows, p);

    // Energy-aware motifs: discovered on subcentroid graph, mapped to items
    let motifs_eng = gl_eng.spot_motives_energy(&aspace_eng, &cfg);
    debug!("Eigen motifs ({}): {:?}", motifs_eig.len(), motifs_eig);
    debug!("Energy motifs ({}): {:?}", motifs_eng.len(), motifs_eng);
    assert!(
        !motifs_eng.is_empty(),
        "EnergyMaps returned 0 motifs; expected planted clusters"
    );

    // -----------------------
    // Compare item-level motifs
    // -----------------------

    // Deduplicate both sets again with the same threshold to align comparison
    fn dedup(mut sets: Vec<Vec<usize>>, thr: f64) -> Vec<Vec<usize>> {
        use std::collections::HashSet;
        let mut out: Vec<HashSet<usize>> = Vec::new();
        for v in sets.drain(..) {
            let s: HashSet<usize> = v.into_iter().collect();
            let mut keep = true;
            for t in &out {
                let inter = s.intersection(t).count() as f64;
                let uni = (s.len() + t.len()) as f64 - inter;
                let j = if uni == 0.0 { 0.0 } else { inter / uni };
                if j >= thr {
                    keep = false;
                    break;
                }
            }
            if keep {
                out.push(s);
            }
        }
        let mut vs: Vec<Vec<usize>> = out
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        vs.sort_by_key(|v| std::cmp::Reverse(v.len()));
        vs
    }

    let eig_d = dedup(motifs_eig.clone(), 0.8);
    let eng_d = dedup(motifs_eng.clone(), 0.8);

    // Compare top-1 overlap by Jaccard; expect strong agreement on planted clusters
    fn jaccard(a: &[usize], b: &[usize]) -> f64 {
        use std::collections::HashSet;
        let sa: HashSet<usize> = a.iter().copied().collect();
        let sb: HashSet<usize> = b.iter().copied().collect();
        let inter = sa.intersection(&sb).count() as f64;
        let uni = (sa.len() + sb.len()) as f64 - inter;
        if uni == 0.0 { 0.0 } else { inter / uni }
    }

    let top_eig = eig_d.get(0).expect("no eigen motif after dedup");
    let top_eng = eng_d.get(0).expect("no energy motif after dedup");

    let j_top = jaccard(top_eig, top_eng);
    debug!(
        "Top-motif Jaccard eigen vs energy: {:.3} | |eig|={}, |eng|={}",
        j_top,
        top_eig.len(),
        top_eng.len()
    );

    // Require substantial agreement on the best motif
    assert!(
        j_top >= 0.3,
        "Top motifs disagree too much (Jaccard={:.3})",
        j_top
    );

    // Optionally, compare coverage: how many eigen motifs have a matching energy motif
    let mut matched = 0usize;
    for e in &eig_d {
        let best = eng_d.iter().map(|x| jaccard(e, x)).fold(0.0_f64, f64::max);
        if best >= 0.5 {
            matched += 1;
        }
    }
    debug!(
        "Eigen motifs matched by energy at J>=0.5: {}/{}",
        matched,
        eig_d.len()
    );
}
