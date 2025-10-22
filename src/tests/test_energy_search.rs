// test_energy_search.rs
#![cfg(test)]

use crate::builder::ArrowSpaceBuilder;
use crate::energymaps::{EnergyParams, EnergyMaps};
use log::info;

#[cfg(test)]
mod test_data {
    pub use crate::tests::test_data::{make_gaussian_hd, make_moons_hd};
}

#[test]
fn test_energy_search_basic() {
    crate::init();
    info!("Test: search_energy_only basic functionality");

    let rows = test_data::make_gaussian_hd(100, 0.6);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(12345)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let k = 5;
    let results = aspace.search_energy_only(&query, &gl_energy, k, 1.0, 0.5);

    assert_eq!(results.len(), k);
    assert!(results[0].1 > results[k - 1].1, "Results should be sorted descending");

    info!("✓ Energy search: {} results, top_score={:.6}", results.len(), results[0].1);
}

#[test]
fn test_energy_search_self_retrieval() {
    crate::init();
    info!("Test: search_energy_only self-retrieval");

    let rows = test_data::make_moons_hd(80, 0.2, 0.08, 99, 42);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query_idx = 10;
    let query = rows[query_idx].clone();
    let results = aspace.search_energy_only(&query, &gl_energy, 1, 1.0, 0.5);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, query_idx, "Should retrieve self as top result");

    info!("✓ Self-retrieval: query_idx={}, result_idx={}", query_idx, results[0].0);
}

#[test]
fn test_energy_search_weight_tuning() {
    crate::init();
    info!("Test: search_energy_only weight parameter effects");

    let rows = test_data::make_gaussian_hd(60, 0.5);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(5555)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let k = 10;

    let results_lambda_heavy = aspace.search_energy_only(&query, &gl_energy, k, 2.0, 0.1);
    let results_dirichlet_heavy = aspace.search_energy_only(&query, &gl_energy, k, 0.1, 2.0);

    assert_eq!(results_lambda_heavy.len(), k);
    assert_eq!(results_dirichlet_heavy.len(), k);

    let overlap = results_lambda_heavy
        .iter()
        .filter(|(idx, _)| results_dirichlet_heavy.iter().any(|(j, _)| j == idx))
        .count();

    info!("✓ Weight tuning: overlap={}/{} results", overlap, k);
}

#[test]
fn test_energy_search_with_projection() {
    crate::init();
    info!("Test: search_energy_only with JL projection");

    let rows = test_data::make_moons_hd(70, 0.3, 0.05, 99, 42);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(222)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[5].clone();
    let results = aspace.search_energy_only(&query, &gl_energy, 3, 1.0, 0.5);

    assert_eq!(results.len(), 3);
    assert!(results[0].1.is_finite());

    info!("✓ Projection search: {} results, top={:.6}", results.len(), results[0].1);
}

#[test]
fn test_energy_search_k_scaling() {
    crate::init();
    info!("Test: search_energy_only k-scaling behavior");

    let rows = test_data::make_gaussian_hd(50, 0.5);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(7777)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();

    for k in [1, 5, 10, 20] {
        let results = aspace.search_energy_only(&query, &gl_energy, k, 1.0, 0.5);
        assert_eq!(results.len(), k.min(aspace.nitems));
        if k > 1 {
            assert!(results[0].1 >= results[k.min(aspace.nitems) - 1].1);
        }
    }

    info!("✓ k-scaling: tested k=[1,5,10,20]");
}

#[test]
fn test_energy_search_optical_compression() {
    crate::init();
    info!("Test: search_energy_only with optical compression");

    let rows = test_data::make_moons_hd(100, 0.3, 0.08, 99, 42);
    let mut p = EnergyParams::default();
    p.optical_tokens = Some(25);

    let builder = ArrowSpaceBuilder::new()
        .with_seed(111)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[10].clone();
    let results = aspace.search_energy_only(&query, &gl_energy,  5, 1.0, 0.5);

    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|(_, s)| s.is_finite()));

    info!("✓ Optical compression search: {} results, GL nodes={}", results.len(), gl_energy.nnodes);
}

#[test]
fn test_energy_search_lambda_proximity() {
    crate::init();
    info!("Test: search_energy_only lambda proximity ranking");

    let rows = test_data::make_gaussian_hd(80, 0.5);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(333)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let results = aspace.search_energy_only(&query, &gl_energy, 10, 1.0, 0.0);

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
    info!("Test: search_energy_only score monotonicity");

    let rows = test_data::make_moons_hd(50, 0.2, 0.1, 99, 42);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(444)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[5].clone();
    let results = aspace.search_energy_only(&query, &gl_energy, 20, 1.0, 0.5);

    for i in 1..results.len() {
        assert!(
            results[i - 1].1 >= results[i].1,
            "Scores should be monotonic descending at position {}",
            i
        );
    }

    info!("✓ Monotonicity: verified for {} results", results.len());
}

#[test]
fn test_energy_search_empty_k() {
    crate::init();
    info!("Test: search_energy_only with k=0");

    let rows = test_data::make_gaussian_hd(30, 0.6);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(555)
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[0].clone();
    let results = aspace.search_energy_only(&query, &gl_energy, 0, 1.0, 0.5);

    assert_eq!(results.len(), 0);

    info!("✓ k=0: returned empty results");
}

#[test]
fn test_energy_search_high_dimensional() {
    crate::init();
    info!("Test: search_energy_only high-dimensional data");

    let rows = test_data::make_gaussian_hd(40, 0.5);
    let p = EnergyParams::default();

    let builder = ArrowSpaceBuilder::new()
        .with_seed(666)
        .with_dims_reduction(true, Some(0.4))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows.clone(), p);

    let query = rows[2].clone();
    let results = aspace.search_energy_only(&query, &gl_energy, 8, 1.0, 0.5);

    assert_eq!(results.len(), 8);
    assert!(results.iter().all(|(_, s)| s.is_finite()));

    info!("✓ High-dim: 200 dims, {} results", results.len());
}
