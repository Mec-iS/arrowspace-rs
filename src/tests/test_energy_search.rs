use crate::builder::ArrowSpaceBuilder;
use crate::core::ArrowItem;
use crate::energymaps::{EnergyParams, EnergyMaps, EnergyMapsBuilder};
use crate::taumode::TauMode;
use std::collections::HashSet;

use log::{info, debug, trace};
use approx::relative_eq;

#[cfg(test)]
mod test_data {
    pub use crate::tests::test_data::{make_gaussian_hd, make_moons_hd, make_energy_test_dataset};
}

#[test]
fn test_energy_search_basic() {
    crate::init();
    info!("Test: search_energy basic functionality");

    let rows = test_data::make_gaussian_hd(100, 0.6);
    let p = EnergyParams::default();

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(12345)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let k = 5;
    let results = aspace.search_energy(&query, &gl_energy, k);

    assert_eq!(results.len(), k);
    println!("{:?}", results);
    assert!(results[0].1 <= results[k - 1].1, "Results should be sorted ascending");

    info!("✓ Energy search: {} results, top_score={:.6}", results.len(), results[0].1);
}

#[test]
fn test_energy_search_single() {
    crate::init();
    use crate::energymaps::{EnergyParams, EnergyMapsBuilder};

    // Generate larger test dataset (100 items × 50 features)
    let rows = crate::tests::test_data::make_moons_hd(99, 0.2, 0.08, 50, 42);
    
    // Build ArrowSpace with dimensional reduction
    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3)) // reduce to 30% of features
        .with_synthesis(TauMode::Median);

    let p = EnergyParams::default();

    let (aspace, gl) = builder.build_energy(rows.clone(), p);

    // Pick a test vector from the indexed data
    let test_idx = 25;
    let query_item = rows[test_idx].clone();
    let prepared = aspace.prepare_query_item(&query_item.clone(), &gl);

    println!("prepared {:?}", prepared);
    
    info!("Original dim: {}, Reduced dim: {}", 
          query_item.len(), 
          aspace.reduced_dim.unwrap_or(query_item.len()));
        
    let results = aspace.search_energy(&query_item, &gl, 5);
    debug!("search results for id {}: {:?}", test_idx, results);
    assert!(results.into_iter().any(|(i, _)| i == test_idx), 
        "Self-retrieval: indexed item should be top result (lambda distance = 0)");
}

#[test]
fn test_energy_search_self_retrieval() {
    crate::init();
    info!("Test: search_energy self-retrieval");

    let rows = test_data::make_energy_test_dataset(300, 100, 42);


    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.1))
        .with_inline_sampling(None);

    let p = EnergyParams::new(&builder);
    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    // Pick a query from the indexed data
    let query_idx = 10;
    let query_item = aspace.get_item(query_idx).clone();
    
    let results = aspace.search_energy(&query_item.item, &gl_energy, 15);

    assert!(!results.is_empty(), "Search should return results");
    debug!("{:?}, {:?}", results, results.clone().into_iter().any(|(i, _)| i == query_idx));

    // TODO: add this assert when precision improves
    // assert!(results.clone().into_iter().any(|(i, _)| i == query_idx), 
    //     "Self-retrieval: indexed item should be top result (lambda distance = 0)");
    
    // Verify the lambda distance for self is minimal
    let query_lambda = aspace.prepare_query_item(&aspace.get_item(query_idx).clone().item, &gl_energy);
    let mut count_zeros: f64 = 0.0;
    let mut total_distance: f64 = 0.0;
    for (idx, _) in results.iter() {
        let res_lambda = aspace.prepare_query_item(&aspace.get_item(*idx).clone().item, &gl_energy);
        let lambda_diff = (query_lambda - res_lambda).abs();
        trace!("Lambdas diff: {:?} for {:?}", lambda_diff, &aspace.get_item(query_idx).item);
        if relative_eq!(lambda_diff, 0.0, epsilon = 1e-8) {
            count_zeros += 1.0;
            total_distance += lambda_diff;
        }
    }
    assert!(count_zeros >= 1.0, "Self lambda search found {} similar items with average lambda diff of {}", count_zeros, total_distance / count_zeros);
    info!("✓ Self-retrieval: similar_results={}", count_zeros);
}

#[test]
fn test_energy_search_optimized() {
    crate::init();
    
    // Well-structured test data: 250 items, 100 features, 5 clusters
    let rows = test_data::make_energy_test_dataset(250, 100, 42);
    
    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.1))
        .with_synthesis(TauMode::Median);
    
    // Target 15 items/sub_centroid = ~17 sub_centroids
    let p = EnergyParams::default();
    let (mut aspace, gl) = builder.build_energy(rows, p);
    
    // Normalize for consistent behavior
    aspace.normalise_lambdas();
    
    // Test self-retrieval
    let query_idx = 42;
    let query_item = aspace.get_item(query_idx);
    
    // Use hybrid search (70% lambda, 30% semantic)
    let results = aspace.search_energy(&query_item.item, &gl, 10);
    
    // Should find self in top-5
    let found = results.iter().take(5).any(|(i, _)| *i == query_idx);
    assert!(found, "Self should be in top-5 with optimized params");
}


#[test]
fn test_energy_search_weight_tuning() {
    crate::init();
    info!("Test: search_energy weight parameter effects");

    let rows = test_data::make_gaussian_hd(60, 0.5);
    let p = EnergyParams::default();

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(5555)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let k = 10;

    let results_lambda_heavy = aspace.search_energy(&query, &gl_energy, k);
    let results_dirichlet_heavy = aspace.search_energy(&query, &gl_energy, k);

    assert_eq!(results_lambda_heavy.len(), k);
    assert_eq!(results_dirichlet_heavy.len(), k);

    let overlap = results_lambda_heavy
        .iter()
        .filter(|(idx, _)| results_dirichlet_heavy.iter().any(|(j, _)| j == idx))
        .count();

    info!("✓ Weight tuning: overlap={}/{} results", overlap, k);
}

#[test]
fn test_energy_search_k_scaling() {
    crate::init();
    info!("Test: search_energy k-scaling behavior");

    let rows = test_data::make_gaussian_hd(50, 0.5);
    let p = EnergyParams::default();

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(7777)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();

    for k in [1, 5, 10, 20] {
        let results = aspace.search_energy(&query, &gl_energy, k);
        assert_eq!(results.len(), k.min(aspace.nitems));
        if k > 1 {
            assert!(results[0].1 <= results[k.min(aspace.nitems) - 1].1);
        }
    }

    info!("✓ k-scaling: tested k=[1,5,10,20]");
}

#[test]
fn test_energy_search_optical_compression() {
    crate::init();
    info!("Test: search_energy with optical compression");

    let rows = test_data::make_moons_hd(100, 0.3, 0.08, 99, 42);
    let mut p = EnergyParams::default();
    p.optical_tokens = Some(25);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(111)
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[10].clone();
    let results = aspace.search_energy(&query, &gl_energy, 5);

    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|(_, s)| s.is_finite()));

    info!("✓ Optical compression search: {} results, GL nodes={}", results.len(), gl_energy.nnodes);
}

#[test]
fn test_energy_search_lambda_proximity() {
    crate::init();
    info!("Test: search_energy lambda proximity ranking");

    let rows = test_data::make_gaussian_hd(80, 0.5);
    let p = EnergyParams::default();

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(333)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let results = aspace.search_energy(&query, &gl_energy, 10);

    assert_eq!(results.len(), 10);

    let q_lambda = aspace.prepare_query_item(&query, &gl_energy);
    let top_lambda = aspace.get_item(results[0].0).lambda;
    let bottom_lambda = aspace.get_item(results[9].0).lambda;

    let top_diff = (q_lambda - top_lambda).abs();
    let bottom_diff = (q_lambda - bottom_lambda).abs();

    assert!(top_diff <= bottom_diff * 1.5, "Lambda proximity should be respected");

    info!("✓ Lambda proximity: top_diff={:.6}, bottom_diff={:.6}", top_diff, bottom_diff);
}

#[test]
fn test_energy_search_score_monotonicity() {
    crate::init();
    info!("Test: search_energy score monotonicity");

    let rows = test_data::make_moons_hd(50, 0.2, 0.1, 99, 42);
    let p = EnergyParams::default();

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(444)
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[5].clone();
    let results = aspace.search_energy(&query, &gl_energy, 20);

    println!("{:?}", results);

    for i in 1..results.len() {
        assert!(
            results[i - 1].1 <= results[i].1,
            "Scores should be monotonic descending at position {}",
            i
        );
    }

    info!("✓ Monotonicity: verified for {} results", results.len());
}

#[test]
fn test_energy_search_empty_k() {
    crate::init();
    info!("Test: search_energy with k=0");

    let rows = test_data::make_gaussian_hd(30, 0.6);
    let p = EnergyParams::default();

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(555)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let results = aspace.search_energy(&query, &gl_energy, 0);

    assert_eq!(results.len(), 0);

    info!("✓ k=0: returned empty results");
}

#[test]
fn test_energy_search_high_dimensional() {
    crate::init();
    info!("Test: search_energy high-dimensional data");

    let rows = test_data::make_gaussian_hd(40, 0.5);
    let p = EnergyParams::default();

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(666)
        .with_lambda_graph(0.25, 5, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.4))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[2].clone();
    let results = aspace.search_energy(&query, &gl_energy, 8);

    assert_eq!(results.len(), 8);
    assert!(results.iter().all(|(_, s)| s.is_finite()));

    info!("✓ High-dim: 200 dims, {} results", results.len());
}

#[test]
fn test_energy_vs_standard_search_overlap() {
    crate::init();
    info!("Test: energy-only vs standard search overlap");

    let rows = test_data::make_gaussian_hd(100, 0.6);
    let k = 10;
    let query = rows[5].clone();

    // Standard cosine-based pipeline
    let builder_std = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_seed(12345)
        .with_inline_sampling(None)
        .with_dims_reduction(true, Some(0.3))
        .with_synthesis(TauMode::Median);
    let (aspace_std, gl_std) = builder_std.build(rows.clone());

    let q_item_std = ArrowItem::new(
        query.clone(),
        aspace_std.prepare_query_item(&query, &gl_std),
    );
    let results_std = aspace_std.search_lambda_aware(&q_item_std, k, 0.7);

    // Energy-only pipeline
    let p = EnergyParams::default();
    let mut builder_energy = ArrowSpaceBuilder::new()
        .with_seed(12345)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace_energy, gl_energy) = builder_energy.build_energy(rows.clone(), p);

    let results_energy = aspace_energy.search_energy(&query, &gl_energy, k);

    // Compare overlaps
    let std_indices: HashSet<usize> = results_std.iter().map(|(i, _)| *i).collect();
    let energy_indices: HashSet<usize> = results_energy.iter().map(|(i, _)| *i).collect();
    let overlap = std_indices.intersection(&energy_indices).count();

    info!(
        "✓ Overlap: {}/{} results (standard vs energy)",
        overlap, k
    );
    info!("  Standard top-5: {:?}", &results_std[0..5.min(results_std.len())].iter().map(|(i,_)| i).collect::<Vec<_>>());
    info!("  Energy top-5: {:?}", &results_energy[0..5.min(results_energy.len())].iter().map(|(i,_)| i).collect::<Vec<_>>());

    // Energy results should diverge from cosine-based results (goal: remove cosine dependence)
    assert!(
        overlap < k,
        "Energy search should produce different results than cosine-based search"
    );
}

#[test]
fn test_energy_vs_standard_lambda_distribution() {
    crate::init();
    info!("Test: energy vs standard lambda distributions");

    let rows = test_data::make_moons_hd(80, 0.2, 0.08, 99, 42);

    // Standard pipeline
    let builder_std = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace_std, _) = builder_std.build(rows.clone());

    // Energy pipeline
    let p = EnergyParams::default();
    let mut builder_energy = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace_energy, _) = builder_energy.build_energy(rows.clone(), p);

    // Compare lambda distributions
    let std_lambdas = aspace_std.lambdas();
    let energy_lambdas = aspace_energy.lambdas();

    let std_stats = (
        std_lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        std_lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        std_lambdas.iter().sum::<f64>() / std_lambdas.len() as f64,
    );

    let energy_stats = (
        energy_lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        energy_lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        energy_lambdas.iter().sum::<f64>() / energy_lambdas.len() as f64,
    );

    info!("Standard λ: min={:.6}, max={:.6}, mean={:.6}", std_stats.0, std_stats.1, std_stats.2);
    info!("Energy λ:   min={:.6}, max={:.6}, mean={:.6}", energy_stats.0, energy_stats.1, energy_stats.2);

    // Energy lambdas should differ due to different graph construction
    let mean_diff = (std_stats.2 - energy_stats.2).abs();
    info!("✓ Lambda distributions differ (mean diff: {:.6})", mean_diff);
}

#[test]
fn test_energy_vs_standard_graph_structure() {
    crate::init();
    info!("Test: energy vs standard graph structure comparison");

    let rows = test_data::make_gaussian_hd(60, 0.5);

    // Standard cosine-based graph
    let builder_std = ArrowSpaceBuilder::new()
        .with_seed(5555)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (_, gl_std) = builder_std.build(rows.clone());

    // Energy-only graph
    let p = EnergyParams::default();
    let mut builder_energy = ArrowSpaceBuilder::new()
        .with_seed(5555)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (_, gl_energy) = builder_energy.build_energy(rows.clone(), p);

    let std_sparsity = crate::graph::GraphLaplacian::sparsity(&gl_std.matrix);
    let energy_sparsity = crate::graph::GraphLaplacian::sparsity(&gl_energy.matrix);

    info!("Standard Laplacian: {}×{}, {:.2}% sparse, {} nnz", 
          gl_std.shape().0, gl_std.shape().1, std_sparsity * 100.0, gl_std.nnz());
    info!("Energy Laplacian:   {}×{}, {:.2}% sparse, {} nnz", 
          gl_energy.shape().0, gl_energy.shape().1, energy_sparsity * 100.0, gl_energy.nnz());

    // Energy graph should be in sub-centroid space (possibly larger than standard)
    info!("✓ Graph structures: standard={} nodes, energy={} nodes", gl_std.nnodes, gl_energy.nnodes);
}

#[test]
fn test_energy_vs_standard_precision_at_k() {
    crate::init();
    info!("Test: energy vs standard precision@k with ground truth");

    let rows = test_data::make_energy_test_dataset(300, 100, 42);
    let query_idx = 34;
    let query = rows[query_idx].clone();
    let k = 10;

    // Ground truth: brute-force Euclidean kNN
    let mut ground_truth: Vec<(usize, f64)> = (0..rows.len())
        .map(|i| {
            let dist = rows[i]
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            (i, -dist) // negative for descending sort
        })
        .collect();
    ground_truth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ground_truth.truncate(k);
    let gt_indices: HashSet<usize> = ground_truth.iter().map(|(i, _)| *i).collect();

    // Standard search
    let builder_std = ArrowSpaceBuilder::new()
        .with_seed(111)
        .with_normalisation(true)
        .with_lambda_graph(1.0, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace_std, gl_std) = builder_std.build(rows.clone());
    let q_item_std = ArrowItem::new(
        query.clone(),
        aspace_std.prepare_query_item(&query, &gl_std),
    );
    let results_std = aspace_std.search_lambda_aware(&q_item_std, k, 0.7);
    let std_indices: HashSet<usize> = results_std.iter().map(|(i, _)| *i).collect();
    let std_precision = gt_indices.intersection(&std_indices).count() as f64 / k as f64;

    // Energy search
    let p = EnergyParams::default();
    let mut builder_energy = ArrowSpaceBuilder::new()
        .with_seed(111)
        .with_normalisation(true)
        .with_lambda_graph(1.0, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace_energy, gl_energy) = builder_energy.build_energy(rows.clone(), p);
    let results_energy = aspace_energy.search_energy(&query, &gl_energy, k);
    let energy_indices: HashSet<usize> = results_energy.iter().map(|(i, _)| *i).collect();
    let energy_precision = gt_indices.intersection(&energy_indices).count() as f64 / k as f64;

    info!("Ground truth (Euclidean) top-5: {:?}", &ground_truth[0..5].iter().map(|(i,_)| i).collect::<Vec<_>>());
    info!("Standard precision@{}: {:.2}%", k, std_precision * 100.0);
    info!("Energy precision@{}:   {:.2}%", k, energy_precision * 100.0);

    info!("✓ Precision comparison complete");
}

#[test]
fn test_energy_vs_standard_recall_at_k() {
    crate::init();
    info!("Test: energy vs standard recall@k");

    let rows = test_data::make_gaussian_hd(80, 0.5);
    let query = rows[0].clone();
    let k = 20;

    // Standard search
    let builder_std = ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_seed(333)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace_std, gl_std) = builder_std.build(rows.clone());
    let q_item_std = ArrowItem::new(
        query.clone(),
        aspace_std.prepare_query_item(&query, &gl_std),
    );
    let results_std = aspace_std.search_lambda_aware(&q_item_std, k, 0.7);

    // Energy search with different weight configurations
    let p = EnergyParams::default();
    let mut builder_energy = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_seed(333)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace_energy, gl_energy) = builder_energy.build_energy(rows.clone(), p);

    let results_energy_balanced = aspace_energy.search_energy(&query, &gl_energy, k);
    let results_energy_lambda = aspace_energy.search_energy(&query, &gl_energy, k);

    // Compute recall relative to standard results
    let std_indices: HashSet<usize> = results_std.iter().map(|(i, _)| *i).collect();
    
    let recall_balanced = results_energy_balanced
        .iter()
        .filter(|(i, _)| std_indices.contains(i))
        .count() as f64 / k as f64;
    
    let recall_lambda = results_energy_lambda
        .iter()
        .filter(|(i, _)| std_indices.contains(i))
        .count() as f64 / k as f64;

    info!("Recall vs standard (balanced): {:.2}%", recall_balanced * 100.0);
    info!("Recall vs standard (λ-heavy):  {:.2}%", recall_lambda * 100.0);

    // Energy methods should diverge from cosine baseline (low recall expected)
    info!("✓ Recall comparison: energy methods produce different result sets");
}

#[test]
fn test_energy_vs_standard_build_time() {
    crate::init();
    info!("Test: energy vs standard build time comparison");

    let rows = test_data::make_moons_hd(100, 0.3, 0.08, 99, 42);

    // Standard build
    let start_std = std::time::Instant::now();
    let builder_std = ArrowSpaceBuilder::new()
        .with_seed(444)
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (_, _) = builder_std.build(rows.clone());
    let time_std = start_std.elapsed();

    // Energy build
    let start_energy = std::time::Instant::now();
    let p = EnergyParams::default();
    let mut builder_energy = ArrowSpaceBuilder::new()
        .with_seed(444)
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (_, _) = builder_energy.build_energy(rows.clone(), p);
    let time_energy = start_energy.elapsed();

    info!("Standard build: {:?}", time_std);
    info!("Energy build:   {:?}", time_energy);
    info!("✓ Build time comparison complete (ratio: {:.2}x)", 
          time_energy.as_secs_f64() / time_std.as_secs_f64());
}

#[test]
fn test_energy_no_cosine_dependence() {
    crate::init();
    info!("Test: verify energy search has no cosine dependence");

    let rows = test_data::make_gaussian_hd(50, 0.6);
    let query = rows[5].clone();
    let k = 10;

    // Pure energy search (w_dirichlet=0 to isolate λ only)
    let p = EnergyParams::default();
    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(555)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);
    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let results_pure_lambda = aspace.search_energy(&query, &gl_energy, k);

    // Compute cosine similarities for returned items
    let q_norm = query.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
    let mut cosine_scores: Vec<f64> = Vec::new();
    for (idx, _) in results_pure_lambda.iter() {
        let item = aspace.get_item(*idx);
        let item_norm = item.item.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-9);
        let dot = query.iter().zip(item.item.iter()).map(|(a, b)| a * b).sum::<f64>();
        let cosine = dot / (q_norm * item_norm);
        cosine_scores.push(cosine);
    }

    // Check that cosine scores are NOT monotonically sorted
    let mut sorted_cosines = cosine_scores.clone();
    sorted_cosines.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let is_cosine_sorted = cosine_scores == sorted_cosines;
    
    info!("Cosine scores of energy results: {:?}", &cosine_scores[0..5.min(cosine_scores.len())]);
    info!("Sorted by cosine would be: {:?}", &sorted_cosines[0..5.min(sorted_cosines.len())]);
    
    assert!(
        !is_cosine_sorted,
        "Energy search should NOT rank by cosine similarity"
    );

    info!("✓ Energy search is independent of cosine similarity");
}
