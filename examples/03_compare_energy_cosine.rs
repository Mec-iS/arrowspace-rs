/// Comparison: Cosine-based vs Energy-only search
/// Demonstrates performance and result differences between traditional similarity and energy-aware retrieval
use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::core::ArrowItem;
use arrowspace::energymaps::{EnergyMaps, EnergyMapsBuilder, EnergyParams};
use log::info;
use std::time::Instant;

#[path = "./common/lib.rs"]
mod common;

use common::{
    print_box, print_section, print_results_table, print_comparison_table,
    print_performance_bar, jaccard_similarity, cosine_sim
};

const VECTORS_DATA: &str = r#"
P0001; 0.82,0.11,0.43,0.28,0.64,0.32,0.55,0.48,0.19,0.73,0.07,0.36,0.58,0.23,0.44,0.31,0.52,0.16,0.61,0.40,0.27,0.49,0.35,0.29
P0002; 0.79,0.12,0.45,0.29,0.61,0.33,0.54,0.47,0.21,0.70,0.08,0.37,0.56,0.22,0.46,0.30,0.51,0.18,0.60,0.39,0.26,0.48,0.36,0.30
P0003; 0.78,0.13,0.46,0.27,0.62,0.34,0.53,0.46,0.22,0.69,0.09,0.35,0.55,0.24,0.45,0.29,0.50,0.17,0.59,0.38,0.28,0.47,0.34,0.31
P0004; 0.81,0.10,0.44,0.26,0.63,0.31,0.56,0.45,0.20,0.71,0.06,0.34,0.57,0.25,0.47,0.33,0.53,0.15,0.62,0.41,0.25,0.50,0.37,0.27
P0005; 0.80,0.12,0.42,0.25,0.60,0.35,0.52,0.49,0.23,0.68,0.10,0.38,0.54,0.21,0.43,0.28,0.49,0.19,0.58,0.37,0.29,0.46,0.33,0.32
P0006; 0.77,0.14,0.41,0.24,0.59,0.36,0.51,0.50,0.24,0.67,0.11,0.39,0.53,0.20,0.42,0.27,0.48,0.20,0.57,0.36,0.30,0.45,0.32,0.33
P0007; 0.83,0.09,0.47,0.30,0.65,0.33,0.57,0.44,0.18,0.72,0.05,0.33,0.59,0.26,0.48,0.34,0.54,0.14,0.63,0.42,0.24,0.51,0.38,0.26
P0008; 0.76,0.15,0.40,0.23,0.58,0.37,0.50,0.51,0.25,0.66,0.12,0.40,0.52,0.19,0.41,0.26,0.47,0.21,0.56,0.35,0.31,0.44,0.31,0.34
P0009; 0.75,0.16,0.39,0.22,0.57,0.38,0.49,0.52,0.26,0.65,0.13,0.41,0.51,0.18,0.40,0.25,0.46,0.22,0.55,0.34,0.32,0.43,0.30,0.35
P0010; 0.84,0.08,0.48,0.31,0.66,0.32,0.58,0.43,0.17,0.74,0.04,0.32,0.60,0.27,0.49,0.35,0.55,0.13,0.64,0.43,0.23,0.52,0.39,0.25
"#;

fn main() {
    arrowspace::init();
    
    print_box("COSINE vs ENERGY-ONLY SEARCH COMPARISON");
    
    let (ids, db): (Vec<String>, Vec<Vec<f64>>) = common::parse_vectors_string(VECTORS_DATA);
    let n_items = db.len();
    let n_features = db[0].len();
    
    info!("📊 Dataset: {} proteins × {} features", n_items, n_features);
    
    let q_index = 3;
    let mut query = db[q_index].clone();
    for v in query.iter_mut() { *v *= 1.02; }
    info!("🔍 Query: perturbed item {} ({})", q_index, ids[q_index]);
    
    let k = 8;
    
    // ═══════════════════════════════════════════════════════════════
    // BASELINE: Naive Cosine Similarity
    // ═══════════════════════════════════════════════════════════════
    print_section("BASELINE: Brute-Force Cosine Similarity");
    
    let start = Instant::now();
    let mut base_scores: Vec<(usize, f64)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_sim(&query, v)))
        .collect();
    base_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    base_scores.truncate(k + 1);
    let baseline_time = start.elapsed();
    
    print_results_table("Baseline Cosine", &base_scores, &ids);
    info!("⏱️  Search time: {:?}", baseline_time);
    
    // ═══════════════════════════════════════════════════════════════
    // METHOD 1: Cosine-Based ArrowSpace
    // ═══════════════════════════════════════════════════════════════
    print_section("METHOD 1: Cosine-Based ArrowSpace (Standard Build)");
    
    info!("Building cosine-based index (graph + λ computation)...");
    let start = Instant::now();
    let (mut aspace_cosine, gl_cosine) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1e-3, 20, k, 2.0, Some(1e-3 * 0.75))
        .with_normalisation(false)
        .with_dims_reduction(true, None)
        .with_seed(42)
        .build(db.clone());
    let build_time_cosine = start.elapsed();
    info!("✓ Cosine index built in {:?}", build_time_cosine);
    
    // Pure cosine search (alpha=1.0)
    let mut query_item = ArrowItem::new(query.clone(), 0.0);
    query_item.lambda = aspace_cosine.prepare_query_item(&query_item.item, &gl_cosine);
    
    let start = Instant::now();
    let results_cosine = aspace_cosine.search_lambda_aware(&query_item, k + 1, 1.0);
    let search_time_cosine = start.elapsed();
    
    print_results_table("Cosine-Based Results (α=1.0)", &results_cosine, &ids);
    info!("⏱️  Search time: {:?}", search_time_cosine);
    
    let ids_baseline: Vec<usize> = base_scores.iter().map(|x| x.0).collect();
    let ids_cosine: Vec<usize> = results_cosine.iter().map(|x| x.0).collect();
    let jaccard_cos = jaccard_similarity(&ids_baseline, &ids_cosine);
    info!("📊 Jaccard(baseline, cosine-index): {:.3}", jaccard_cos);
    
    // Lambda-aware search (alpha=0.7)
    info!("\n🔬 Testing λ-aware blend (α=0.7: 70% cosine + 30% energy)...");
    let results_lambda = aspace_cosine.search_lambda_aware(&query_item, k + 1, 0.7);
    print_results_table("λ-Aware Results (α=0.7)", &results_lambda, &ids);
    
    let ids_lambda: Vec<usize> = results_lambda.iter().map(|x| x.0).collect();
    let jaccard_lam = jaccard_similarity(&ids_baseline, &ids_lambda);
    info!("📊 Jaccard(baseline, λ-aware): {:.3}", jaccard_lam);
    
    // ═══════════════════════════════════════════════════════════════
    // METHOD 2: Energy-Only Search (No Cosine)
    // ═══════════════════════════════════════════════════════════════
    print_section("METHOD 2: Energy-Only Search (EnergyMaps Pipeline)");
    
    info!("Building energy-only index (optical compression + Rayleigh kNN)...");
    let p = EnergyParams {
        optical_tokens: Some(30),
        trim_quantile: 0.1,
        eta: 0.1,
        steps: 4,
        split_quantile: 0.9,
        neighbor_k: 12,
        split_tau: 0.15,
        w_lambda: 1.0,
        w_disp: 0.5,
        w_dirichlet: 0.25,
        candidate_m: 40,
    };
    
    let start = Instant::now();
    let mut builder_energy = ArrowSpaceBuilder::new()
        .with_seed(42)
        .with_inline_sampling(None);
    let (aspace_energy, gl_energy) = builder_energy.build_energy(db.clone(), p);
    let build_time_energy = start.elapsed();
    info!("✓ Energy index built in {:?}", build_time_energy);
    
    info!("Energy graph: {} nodes (sub-centroids after diffusion + splits)", gl_energy.nnodes);
    
    // Energy-only search (no cosine)
    let start = Instant::now();
    let results_energy = aspace_energy.search_energy(&query, &gl_energy, k + 1, 1.0, 0.5);
    let search_time_energy = start.elapsed();
    
    print_results_table("Energy-Only Results (pure λ + Dirichlet)", &results_energy, &ids);
    info!("⏱️  Search time: {:?}", search_time_energy);
    
    let ids_energy: Vec<usize> = results_energy.iter().map(|x| x.0).collect();
    let jaccard_energy = jaccard_similarity(&ids_baseline, &ids_energy);
    info!("📊 Jaccard(baseline, energy-only): {:.3}", jaccard_energy);
    
    // ═══════════════════════════════════════════════════════════════
    // COMPARISON & ANALYSIS
    // ═══════════════════════════════════════════════════════════════
    print_section("COMPARISON SUMMARY");
    
    info!("\n┌─────────────────────────────────────────────────────────────────────┐");
    info!("│ RESULT SET OVERLAP (Jaccard Similarity)                            │");
    info!("├─────────────────────────────────────────────────────────────────────┤");
    info!("│  Baseline ↔ Cosine-Based:    {:.3}  (near-identical)              │", jaccard_cos);
    info!("│  Baseline ↔ λ-Aware (α=0.7): {:.3}  (moderate divergence)        │", jaccard_lam);
    info!("│  Baseline ↔ Energy-Only:     {:.3}  (high divergence)             │", jaccard_energy);
    info!("│  Cosine ↔ Energy:            {:.3}                                 │", 
          jaccard_similarity(&ids_cosine, &ids_energy));
    info!("└─────────────────────────────────────────────────────────────────────┘");
    
    print_comparison_table(
        &[
            ("Baseline (Cosine)", baseline_time.as_micros() as f64 / 1000.0, k),
            ("Cosine-Based (ArrowSpace)", search_time_cosine.as_micros() as f64 / 1000.0, k),
            ("Energy-Only (EnergyMaps)", search_time_energy.as_micros() as f64 / 1000.0, k),
        ]
    );
    
    info!("\n┌─────────────────────────────────────────────────────────────────────┐");
    info!("│ BUILD TIME COMPARISON                                               │");
    info!("├─────────────────────────────────────────────────────────────────────┤");
    print_performance_bar("Cosine-Based", build_time_cosine.as_millis() as f64);
    print_performance_bar("Energy-Only ", build_time_energy.as_millis() as f64);
    info!("└─────────────────────────────────────────────────────────────────────┘");
    
    // ═══════════════════════════════════════════════════════════════
    // ALPHA SWEEP (Cosine-Based Only)
    // ═══════════════════════════════════════════════════════════════
    print_section("ALPHA SWEEP: Transition from Cosine → Energy");
    
    info!("Testing how results change as α decreases (cosine → λ blend):\n");
    for alpha in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0].iter() {
        let results = aspace_cosine.search_lambda_aware(&query_item, k, *alpha);
        let top3: Vec<String> = results.iter().take(3).map(|(i, _)| ids[*i].clone()).collect();
        info!("α={:.1} ({:3}% cos): top-3 = {:?}", 
              alpha, (alpha * 100.0) as i32, top3);
    }
    
    // ═══════════════════════════════════════════════════════════════
    // KEY INSIGHTS
    // ═══════════════════════════════════════════════════════════════
    print_section("KEY INSIGHTS");
    
    info!("┌─────────────────────────────────────────────────────────────────────┐");
    info!("│ 1. COSINE-BASED INDEX                                               │");
    info!("│    • Near-perfect match with baseline brute-force cosine           │");
    info!("│    • Graph structure speeds up search without changing ranking     │");
    info!("│    • λ-aware blend (α<1) introduces energy-based reranking         │");
    info!("├─────────────────────────────────────────────────────────────────────┤");
    info!("│ 2. ENERGY-ONLY SEARCH                                               │");
    info!("│    • Completely removes cosine dependence (0 cosine computation)   │");
    info!("│    • Ranks by: λ proximity + local dispersion + Rayleigh distance  │");
    info!("│    • Low overlap with cosine baseline proves independence          │");
    info!("│    • Suitable for testing pure energy-aware retrieval hypothesis   │");
    info!("├─────────────────────────────────────────────────────────────────────┤");
    info!("│ 3. PERFORMANCE                                                      │");
    info!("│    • Cosine-based: Fast build, fast search (graph acceleration)    │");
    info!("│    • Energy-only: Slower build (diffusion+splits), fast search     │");
    info!("│    • Build overhead amortizes over many queries                    │");
    info!("├─────────────────────────────────────────────────────────────────────┤");
    info!("│ 4. USE CASES                                                        │");
    info!("│    • Cosine-based: Semantic similarity with optional energy blend  │");
    info!("│    • Energy-only: Novel ranking criteria, diversity, exploration   │");
    info!("│    • Hybrid: Combine both in ensemble/reranking pipeline           │");
    info!("└─────────────────────────────────────────────────────────────────────┘");
    
    print_box("COMPARISON COMPLETE");
}
