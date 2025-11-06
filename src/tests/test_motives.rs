use crate::builder::ArrowSpaceBuilder;
use crate::energymaps::{EnergyMapsBuilder, EnergyParams};
use crate::motives::{MotiveConfig, Motives};
use crate::tests::test_data::make_gaussian_cliques;

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
    println!("Found {} motifs:", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        println!("  Motif {}: {:?}", i, m);
    }

    assert!(motifs.len() > 0);
}

#[test]
fn test_motives_basic_2() {
    crate::tests::init();

    let rows = make_gaussian_cliques(12, 0.05, 15, 10, 42);

    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 16, 10, 2.0, None) // denser intra-group
        .with_normalisation(true)
        .with_sparsity_check(false)
        .build(rows);

    let cfg = MotiveConfig {
        top_l: 12,        // >= topk; can set to 16 to fully keep neighbors
        min_triangles: 3, // slightly higher now that graph is denser
        min_clust: 0.5,   // tighter seeding
        max_motif_size: 24,
        max_sets: 100,
        jaccard_dedup: 0.4,
    };

    // rayleigh_max: In this setup, small sets had R(1S)R(1S) near 2.0 when k/topk were higher; with k=14, topk=8 and top_l=16,
    // within-group triangles increase and boundary per node decreases, allowing cohesive sets to reach R≲1.5R≲1.5.
    // Lower bounds like 0.1–0.6 are too strict for this graph density.

    let motifs = gl.spot_motives_eigen(&cfg);
    println!("Found {} motifs:", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        println!("  Motif {}: {:?}", i, m);
    }

    assert!(motifs.len() > 0);
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

    println!("Found {} motifs (energy):", motifs.len());
    for (i, m) in motifs.iter().enumerate() {
        println!("  Motif {}: {:?}", i, m);
    }

    assert!(motifs.len() > 0);
}
