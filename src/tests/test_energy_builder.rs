// test_energy_builder.rs
#![cfg(test)]

use crate::builder::ArrowSpaceBuilder;
use crate::energymaps::{EnergyMapsBuilder, EnergyParams};
use crate::graph::GraphLaplacian;
use crate::taumode::TauMode;
use log::info;
use smartcore::linalg::basic::arrays::Array;

use crate::tests::test_data::{make_gaussian_hd, make_moons_hd};

#[test]
fn test_energy_build_basic() {
    crate::tests::init();
    info!("Test: build_energy basic pipeline");

    let rows = make_gaussian_hd(100, 0.2);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(12345)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows, EnergyParams::new(&builder));

    assert!(aspace.nitems > 0);
    assert!(aspace.nfeatures == 100);
    assert!(gl_energy.nnodes > 0);
    assert!(gl_energy.nnz() > 0);
    assert!(aspace.lambdas.iter().any(|&l| l != 0.0));

    info!(
        "✓ Energy build succeeded with {} items, {} GL nodes",
        aspace.nitems, gl_energy.nnodes
    );
}

#[test]
fn test_energy_build_with_optical_compression() {
    crate::tests::init();
    info!("Test: build_energy with optical compression");

    let rows = make_moons_hd(150, 0.2, 0.1, 100, 42);
    let mut p = EnergyParams::default();
    p.optical_tokens = Some(30);
    p.trim_quantile = 0.15;

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows, p);

    assert!(aspace.nitems > 0);
    assert!(gl_energy.nnodes <= 30 * 2);
    assert!(aspace.lambdas.iter().any(|&l| l > 0.0));

    info!(
        "✓ Optical compression: {} GL nodes (target ≤ {})",
        gl_energy.nnodes, 30
    );
}

#[test]
fn test_energy_build_diffusion_splits() {
    crate::tests::init();
    info!("Test: build_energy diffusion and sub-centroid splitting");

    let rows = make_gaussian_hd(80, 0.3);
    let mut p = EnergyParams::default();
    p.steps = 6;
    p.split_quantile = 0.85;
    p.split_tau = 0.2;

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(5555)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows, p);

    assert!(aspace.n_clusters > 0);
    assert!(gl_energy.nnodes >= aspace.n_clusters);
    assert!(aspace.lambdas().iter().all(|&l| l.is_finite()));

    info!(
        "✓ Diffusion + splitting: {} clusters → {} GL nodes",
        aspace.n_clusters, gl_energy.nnodes
    );
}

#[test]
fn test_energy_laplacian_properties() {
    crate::tests::init();
    info!("Test: energy Laplacian properties (connectivity, symmetry)");

    let rows = make_moons_hd(60, 0.2, 0.1, 99, 42);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(7777)
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (_, gl_energy) = builder.build_energy(rows, EnergyParams::new(&builder));

    let sparsity = GraphLaplacian::sparsity(&gl_energy.matrix);
    assert!(sparsity > 0.0, "Laplacian should have some sparsity");

    let is_sym = gl_energy.is_symmetric(1e-6);
    assert!(is_sym, "Energy Laplacian should be symmetric");

    info!(
        "✓ Laplacian: {:.2}% sparse, symmetric={}",
        sparsity * 100.0,
        is_sym
    );
}

#[test]
fn test_energy_build_with_projection() {
    crate::tests::init();
    info!("Test: build_energy with JL projection");

    let rows = make_gaussian_hd(70, 0.4);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(222)
        .with_lambda_graph(0.5, 6, 3, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows, EnergyParams::new(&builder));

    assert!(aspace.projection_matrix.is_some());
    assert!(aspace.reduced_dim.is_some());
    assert!(aspace.reduced_dim.unwrap() < 128);
    assert!(aspace.lambdas().iter().any(|&l| l > 0.0));

    info!(
        "✓ Projection: 128 → {} dims, {} GL nodes",
        aspace.reduced_dim.unwrap(),
        gl_energy.nnodes
    );
}

#[test]
fn test_energy_build_taumode_consistency() {
    crate::tests::init();
    info!("Test: build_energy taumode consistency");

    let rows = make_moons_hd(50, 0.2, 0.08, 99, 42);

    let mut builder = ArrowSpaceBuilder::new()
        .with_synthesis(TauMode::Mean)
        .with_seed(111)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, _) = builder.build_energy(rows, EnergyParams::new(&builder));

    assert_eq!(aspace.taumode, TauMode::Mean);
    assert!(aspace.lambdas.len() == aspace.nitems);
    assert!(aspace.lambdas.iter().all(|&l| l >= 0.0 && l.is_finite()));

    let lambda_mean = aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64;
    info!(
        "✓ Taumode Mean: {} lambdas, mean={:.6}",
        aspace.lambdas.len(),
        lambda_mean
    );
}

#[test]
fn test_energy_build_custom_params() {
    crate::tests::init();
    info!("Test: build_energy with custom EnergyParams");
    let p_neighbor_k = 10;

    let rows = make_gaussian_hd(40, 0.1);
    let p = EnergyParams {
        optical_tokens: None,
        trim_quantile: 0.05,
        eta: 0.15,
        steps: 2,
        split_quantile: 0.95,
        neighbor_k: p_neighbor_k,
        split_tau: 0.1,
        w_lambda: 1.5,
        w_disp: 0.3,
        w_dirichlet: 0.15,
        candidate_m: 20,
    };

    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.001, p_neighbor_k, 5, 2.0, None)
        .with_seed(333)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    assert!(builder.normalise == false);

    let (aspace, gl_energy) = builder.build_energy(rows, p);

    assert!(gl_energy.graph_params.normalise == false);
    assert!(aspace.lambdas.iter().any(|&l| l > 0.0));

    info!(
        "✓ Custom params: k={}, normalize={}",
        gl_energy.graph_params.k, gl_energy.graph_params.normalise
    );
}

#[test]
fn test_energy_build_lambda_statistics() {
    crate::tests::init();
    info!("Test: build_energy lambda statistics");

    let rows = make_gaussian_hd(100, 0.6);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(444)
        .with_lambda_graph(0.01, 10, 5, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, _) = builder.build_energy(rows, EnergyParams::new(&builder));

    let lambdas = aspace.lambdas();
    let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;

    assert!(min >= 0.0);
    assert!(max > min);
    assert!(mean > 0.0 && mean.is_finite());

    info!(
        "✓ Lambda stats: min={:.6}, max={:.6}, mean={:.6}",
        min, max, mean
    );
}

#[test]
fn test_build_energy_dimensionality_reduction() {
    crate::tests::init();
    // Create synthetic high-dimensional dataset
    let n_items = 99;
    let n_features = 100;
    let rows = crate::tests::test_data::make_gaussian_hd(n_items, 0.6);

    // Configure builder with dimension reduction enabled
    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.001, 6, 3, 2.0, None)
        .with_dims_reduction(true, Some(0.3)) // Enable with ε=0.3
        .with_seed(42)
        .with_spectral(false)
        .with_sparsity_check(false)
        .with_synthesis(TauMode::Median);

    let energy_params = EnergyParams {
        optical_tokens: None,
        trim_quantile: 0.1,
        eta: 0.05,
        steps: 4,
        split_quantile: 0.9,
        neighbor_k: 12,
        split_tau: 0.15,
        w_lambda: 1.0,
        w_disp: 0.5,
        w_dirichlet: 0.25,
        candidate_m: 40,
    };

    // Build energy index
    let (aspace, gl_energy) = builder.build_energy(rows.clone(), energy_params);

    // Test 1: Verify dimension reduction occurred
    assert!(
        aspace.projection_matrix.is_some(),
        "Projection matrix should exist when dims_reduction is enabled"
    );

    let reduced_dim = aspace.reduced_dim.expect("Reduced dimension should be set");
    assert!(
        reduced_dim < n_features,
        "Reduced dimension {} should be less than original {}",
        reduced_dim,
        n_features
    );

    debug!("✓ Dimension reduction: {} → {}", 100, reduced_dim);

    // Test 2: Verify sub_centroids have reduced dimensions
    let sub_centroids = aspace
        .sub_centroids
        .as_ref()
        .expect("Sub_centroids should be stored");
    let (n_subcentroids, sub_features) = sub_centroids.shape();

    assert_eq!(
        sub_features, reduced_dim,
        "Sub_centroids features {} should match reduced_dim {}",
        sub_features, reduced_dim
    );

    debug!(
        "✓ Sub_centroids shape: {} × {}",
        n_subcentroids, sub_features
    );

    // Test 3: Verify graph dimensions match reduced sub_centroids
    let (graph_rows, graph_cols) = gl_energy.shape();

    assert_eq!(
        graph_rows, graph_cols,
        "Energy Laplacian should be square {} != {}",
        graph_rows, graph_cols
    );

    debug!("✓ Graph shape: {}×{}", graph_rows, graph_cols);

    // Test 4: Verify lambdas computed for all items
    assert_eq!(
        aspace.lambdas.len(),
        n_items,
        "Lambda count {} should match item count {}",
        aspace.lambdas.len(),
        n_items
    );

    assert!(
        aspace.lambdas.iter().all(|&l| l.is_finite() && l >= 0.0),
        "All lambdas should be finite and non-negative"
    );

    debug!("✓ Lambdas computed: {} values", aspace.lambdas.len());

    // Test 5: Verify centroid mapping exists and is valid
    let centroid_map = aspace
        .centroid_map
        .as_ref()
        .expect("Centroid map should exist");

    assert_eq!(
        centroid_map.len(),
        n_items,
        "Centroid map size {} should match item count {}",
        centroid_map.len(),
        n_items
    );

    assert!(
        centroid_map.iter().all(|&idx| idx < n_subcentroids),
        "All centroid indices should be < {}",
        n_subcentroids
    );

    debug!(
        "✓ Centroid mapping valid: {} items mapped",
        centroid_map.len()
    );

    // Test 6: Verify item norms computed
    let item_norms = aspace
        .item_norms
        .as_ref()
        .expect("Item norms should be computed");

    assert_eq!(
        item_norms.len(),
        n_items,
        "Norms count {} should match item count {}",
        item_norms.len(),
        n_items
    );

    assert!(
        item_norms.iter().all(|&n| n > 0.0 && n.is_finite()),
        "All norms should be positive and finite"
    );

    debug!("✓ Item norms computed: {} values", item_norms.len());

    // Test 7: Test projection consistency
    let test_item = &rows[0];
    let projected = aspace.project_query(test_item);

    assert_eq!(
        projected.len(),
        reduced_dim,
        "Projected query dimension {} should match reduced_dim {}",
        projected.len(),
        reduced_dim
    );

    debug!("✓ Query projection: {} → {}", 100, projected.len());

    // Test 8: No bounds errors during lambda computation
    // This implicitly passed if we got here without panicking
    debug!("✓ No index out of bounds errors during taumode computation");

    debug!("\n✅ All dimensionality reduction tests passed!");
}

#[test]
#[should_panic(expected = "When using build_energy, dim reduction is needed")]
fn test_build_energy_requires_dims_reduction() {
    let rows: Vec<Vec<f64>> = vec![vec![1.0; 128]; 100];

    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.001, 6, 3, 2.0, None)
        .with_dims_reduction(false, None); // Disabled

    let energy_params = EnergyParams::default();

    // Should panic
    builder.build_energy(rows, energy_params);
}
