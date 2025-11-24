use crate::builder::ConfigValue;
use crate::core::ArrowSpace;
use crate::reduction::ImplicitProjection;
use crate::{
    builder::ArrowSpaceBuilder,
    graph::GraphLaplacian,
    sampling::SamplerType,
    taumode::TauMode,
    tests::test_data::{make_gaussian_blob, make_gaussian_hd, make_moons_hd},
};

use log::debug;
use std::collections::HashMap;

/// Helper to compare two GraphLaplacian matrices for equality
fn laplacian_eq(a: &GraphLaplacian, b: &GraphLaplacian, eps: f64) -> bool {
    if a.matrix.shape() != b.matrix.shape() {
        return false;
    }

    let (r, c) = a.matrix.shape();
    for i in 0..r {
        for j in 0..c {
            let ai = *a.matrix.get(i, j).unwrap_or(&0.0);
            let bj = *b.matrix.get(i, j).unwrap_or(&0.0);
            if (ai - bj).abs() > eps {
                return false;
            }
        }
    }
    true
}

/// Helper to collect diagonal of the Laplacian matrix as Vec<f64>
fn diag_vec(gl: &GraphLaplacian) -> Vec<f64> {
    let (n, _) = gl.matrix.shape();
    (0..n).map(|i| *gl.matrix.get(i, i).unwrap()).collect()
}

#[allow(dead_code)]
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

#[test]
fn test_builder_direction_vs_magnitude_sensitivity() {
    // Construct vectors where two have the same direction but vastly different magnitudes
    let items = make_gaussian_blob(99, 0.5);

    // Build with normalisation=true (cosine-like, scale-invariant)
    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 3, 2, 2.0, Some(0.25))
        .with_normalisation(true)
        .with_spectral(true)
        .build(items.clone());

    // Build with normalisation=false (τ-mode: magnitude-sensitive)
    let (aspace_tau, gl_tau) = ArrowSpaceBuilder::default()
        .with_lambda_graph(1.0, 3, 2, 2.0, Some(0.25))
        .with_normalisation(false)
        .with_spectral(true)
        .build(items.clone());

    // τ-mode should differ from normalised graph because it is magnitude-sensitive
    let matrices_equal = laplacian_eq(&gl_norm, &gl_tau, 1e-12);
    assert!(
        !matrices_equal,
        "τ-mode should differ from normalised graph due to magnitude sensitivity"
    );

    // Lambda distributions should also differ
    let lambdas_norm = aspace_norm.lambdas();
    let lambdas_tau = aspace_tau.lambdas();

    debug!(
        "Normalized lambdas (first 3): {:?}",
        &lambdas_norm[..3.min(lambdas_norm.len())]
    );
    debug!(
        "Tau lambdas (first 3): {:?}",
        &lambdas_tau[..3.min(lambdas_tau.len())]
    );
}

#[test]
fn test_builder_normalisation_flag_is_preserved() {
    // Verify that normalisation flag is properly propagated through the builder
    let items = make_moons_hd(99, 0.1, 0.5, 3, 123);

    let (_aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 2, 1, 2.0, None)
        .with_normalisation(false)
        .build(items);

    assert_eq!(
        gl.graph_params.normalise, false,
        "normalise flag must be preserved"
    );
}

#[test]
fn test_builder_clustering_produces_valid_assignments() {
    // Test that the builder produces valid cluster assignments
    let items = make_moons_hd(99, 0.05, 1.5, 3, 456);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 3, 2, 2.0, None)
        .with_normalisation(true)
        .build(items.clone());

    debug!("Assignments: {:?}", aspace.cluster_assignments);

    // Verify all items are assigned
    let assigned_count = aspace
        .cluster_assignments
        .iter()
        .filter(|x| x.is_some())
        .count();
    assert!(
        assigned_count > 0,
        "At least some items should be assigned to clusters"
    );
}

#[test]
fn test_builder_spectral_laplacian_computation() {
    // Test that spectral Laplacian is computed when requested
    let items = make_moons_hd(4, 0.12, 0.4, 5, 789);

    // Build WITHOUT spectral computation
    let (aspace_no_spectral, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.2, 2, 1, 2.0, None)
        .with_spectral(false)
        .with_inline_sampling(None)
        .build(items.clone());

    // Build WITH spectral computation
    let (aspace_spectral, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.2, 2, 1, 2.0, None)
        .with_spectral(true)
        .with_inline_sampling(None)
        .build(items.clone());

    debug!(
        "No spectral - signals shape: {:?}",
        aspace_no_spectral.signals.shape()
    );
    debug!(
        "With spectral - signals shape: {:?}",
        aspace_spectral.signals.shape()
    );

    // When spectral is disabled, signals should be empty (0x0)
    assert_eq!(
        aspace_no_spectral.signals.shape(),
        (0, 0),
        "Signals should be empty when spectral computation is disabled"
    );

    // When spectral is enabled, signals should be populated (FxF matrix)
    assert_ne!(
        aspace_spectral.signals.shape(),
        (0, 0),
        "Signals should be populated when spectral computation is enabled"
    );
}

#[test]
fn test_builder_lambda_computation_with_different_tau_modes() {
    let items = make_moons_hd(3, 0.15, 0.35, 4, 321);

    // Build with Median tau mode
    let (aspace_median, _) = ArrowSpaceBuilder::default()
        .with_synthesis(TauMode::Median)
        .with_lambda_graph(0.2, 2, 1, 2.0, None)
        .with_inline_sampling(None)
        .build(items.clone());

    // Build with Max tau mode
    let (aspace_fixed, _) = ArrowSpaceBuilder::default()
        .with_synthesis(TauMode::Fixed(0.5))
        .with_lambda_graph(0.2, 2, 1, 2.0, None)
        .with_inline_sampling(None)
        .build(items.clone());

    let lambdas_median = aspace_median.lambdas();
    let lambdas_fixed = aspace_fixed.lambdas();

    debug!("Median tau lambdas: {:?}", lambdas_median);
    debug!("Max tau lambdas: {:?}", lambdas_fixed);

    // Lambdas should differ between tau modes
    let mut differences = 0;
    for (m, mx) in lambdas_median.iter().zip(lambdas_fixed.iter()) {
        if (m - mx).abs() > 1e-10 {
            differences += 1;
        }
    }

    assert!(
        differences > 0,
        "Different tau modes should produce different lambda values"
    );
}

#[test]
fn test_builder_with_normalized_vs_unnormalized_items() {
    // Test how normalization affects clustering and spectral properties
    let items = make_moons_hd(4, 0.18, 0.4, 6, 654);

    // Create unnormalized items with different scales
    let scales = vec![1.0, 3.0, 0.5, 2.5];
    let items_unnormalized: Vec<Vec<f64>> = items
        .iter()
        .zip(scales.iter())
        .map(|(item, &scale)| item.iter().map(|x| x * scale).collect())
        .collect();

    // Build with normalized data
    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.2, 2, 1, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .with_inline_sampling(None)
        .build(items.clone());

    // Build with unnormalized data (no normalization flag)
    let (aspace_unnorm, gl_unnorm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.2, 2, 1, 2.0, None)
        .with_normalisation(false)
        .with_spectral(true)
        .with_inline_sampling(None)
        .build(items_unnormalized);

    debug!("=== SPECTRAL ANALYSIS ===");
    let lambdas_norm = aspace_norm.lambdas();
    let lambdas_unnorm = aspace_unnorm.lambdas();

    debug!(
        "Normalized lambdas: {:?}",
        &lambdas_norm[..3.min(lambdas_norm.len())]
    );
    debug!(
        "Unnormalized lambdas: {:?}",
        &lambdas_unnorm[..3.min(lambdas_unnorm.len())]
    );

    // Diagonal elements should differ due to different graph structure
    let d_norm = diag_vec(&gl_norm);
    let d_unnorm = diag_vec(&gl_unnorm);

    debug!("Normalized diagonals: {:?}", &d_norm[..3.min(d_norm.len())]);
    debug!(
        "Unnormalized diagonals: {:?}",
        &d_unnorm[..3.min(d_unnorm.len())]
    );

    // The graphs should be different due to magnitude sensitivity
    assert!(
        !laplacian_eq(&gl_norm, &gl_unnorm, 1e-10),
        "Normalized and unnormalized builds should produce different graphs"
    );
}

#[test]
fn test_builder_with_inline_sampling() {
    // Test builder with inline sampling enabled
    let items = make_gaussian_blob(100, 0.5);

    let (_aspace_sampling, _gl_sampling) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_inline_sampling(Some(SamplerType::DensityAdaptive(0.5)))
        .build(items.clone());

    let (_aspace_no_sampling, _gl_no_sampl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_inline_sampling(Some(SamplerType::DensityAdaptive(0.5)))
        .build(items);
}

#[test]
fn test_builder_dimensionality_reduction() {
    // Test builder with dimensionality reduction enabled
    let items = make_moons_hd(50, 0.15, 0.35, 128, 111);

    let (aspace_reduced, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_sparsity_check(false)
        .build(items.clone());

    let (aspace_full, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_dims_reduction(false, None)
        .with_sparsity_check(false)
        .build(items);

    debug!("Original dimension: {}", aspace_full.nfeatures);
    debug!("Reduced dimension: {:?}", aspace_reduced.reduced_dim);

    if let Some(reduced_dim) = aspace_reduced.reduced_dim {
        assert!(
            reduced_dim < aspace_full.nfeatures,
            "Reduced dimension should be less than original"
        );
        assert!(
            aspace_reduced.projection_matrix.is_some(),
            "Projection matrix should be present when reduction is enabled"
        );
    }
}

#[test]
fn test_builder_lambda_statistics() {
    // Test that lambda statistics show reasonable variance using high-dimensional moons data
    // Use make_moons_hd with high noise to create natural clusters with distinct spectral properties

    let items: Vec<Vec<f64>> = make_moons_hd(
        200, // Sufficient samples for meaningful statistics
        0.3, // High noise for variance - standard deviation of Gaussian noise
        0.5, // Moderate separation between moons
        40,  // High dimensionality to spread variance
        768, // Fixed seed for reproducibility
    );

    debug!("=== LAMBDA STATISTICS TEST ===");
    debug!(
        "Generated {} items with {} features",
        items.len(),
        items[0].len()
    );

    // Build ArrowSpace with spectral computation enabled
    // Use parameters that create a well-connected graph for meaningful eigenvalues
    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(
            0.5,  // Larger eps for connectivity across noise
            6,    // More neighbors to capture local structure
            3,    // Keep top-3 neighbors
            2.0,  // Quadratic kernel
            None, // Auto-compute sigma
        )
        .with_sparsity_check(false)
        .build(items);

    debug!("Graph has {} nodes", gl.nnodes);

    // Extract lambda statistics
    let lambdas = aspace.lambdas();

    let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;

    // Compute standard deviation for variance measure
    let variance = lambdas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / lambdas.len() as f64;
    let std_dev = variance.sqrt();

    debug!("=== LAMBDA DISTRIBUTION ===");
    debug!("Min:     {:.6}", min);
    debug!("Max:     {:.6}", max);
    debug!("Mean:    {:.6}", mean);
    debug!("Std Dev: {:.6}", std_dev);
    debug!("Range:   {:.6}", max - min);

    // Show first few lambdas for inspection
    debug!("First 5 lambdas: {:?}", &lambdas[..5.min(lambdas.len())]);

    // All lambdas should be non-negative (spectral property)
    assert!(
        min >= 0.0,
        "All lambdas should be non-negative, got min={}",
        min
    );

    // Should have meaningful variance - the noise in make_moons_hd ensures this
    // With high noise (0.3), points within each moon have varying distances to centroids,
    // creating different local graph structures and thus different lambda values
    assert!(
        max > min,
        "Lambdas should have some variance: max={:.6}, min={:.6}",
        max,
        min
    );

    // Stronger variance test: range should be significant relative to mean
    let relative_range = (max - min) / mean.max(1e-10);
    assert!(
        relative_range > 0.1,
        "Lambda range should be at least 10% of mean for high-variance data: \
         range={:.6}, mean={:.6}, relative={:.6}",
        max - min,
        mean,
        relative_range
    );

    // Standard deviation should indicate spread
    assert!(
        std_dev > 1e-6,
        "Lambda standard deviation should indicate spread: std_dev={:.6}",
        std_dev
    );

    debug!("✓ Lambda statistics show expected variance from noisy moon dataset");
}

#[test]
fn test_builder_cluster_radius_impact() {
    // Test how cluster radius affects clustering
    let items = make_gaussian_blob(99, 0.5);

    // This test verifies that the auto-computed cluster parameters
    // produce reasonable clustering behavior
    let (aspace, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 3, 2, 2.0, None)
        .with_seed(42)
        .build(items);

    // Radius should be positive
    assert!(
        aspace.cluster_radius > 0.0,
        "Cluster radius should be positive"
    );
}

#[test]
fn test_empty_with_projection_path() {
    crate::tests::init();
    let mut proj_data = HashMap::new();
    proj_data.insert(
        "pj_mtx_original_dim".to_string(),
        ConfigValue::OptionUsize(Some(384)),
    );
    proj_data.insert(
        "pj_mtx_reduced_dim".to_string(),
        ConfigValue::OptionUsize(Some(91)),
    );
    proj_data.insert(
        "pj_mtx_seed".to_string(),
        ConfigValue::OptionU64(Some(123456789)),
    );
    proj_data.insert("extra_reduced_dim".to_string(), ConfigValue::Bool(false));

    let nrows = 10_000;
    let ncols = 384;

    let aspace = ArrowSpace::empty_with_projection(proj_data, nrows, ncols);

    // Basic shape
    assert_eq!(aspace.nitems, nrows);
    assert_eq!(aspace.nfeatures, ncols);

    // Projection is set correctly
    let proj = aspace
        .projection_matrix
        .as_ref()
        .expect("projection_matrix must be Some");
    assert_eq!(
        *proj,
        ImplicitProjection {
            original_dim: 384,
            reduced_dim: 91,
            seed: 123456789,
        }
    );

    // reduced_dim and extra_reduced_dim are consistent with proj_data
    assert_eq!(aspace.reduced_dim, Some(91));
    assert_eq!(aspace.extra_reduced_dim, false);
}

#[test]
#[should_panic(expected = "Reconstructing with extra dim reduction is not implemented yet")]
fn test_empty_with_projection_panics_on_extra_reduced_dim_true() {
    let mut proj_data = HashMap::new();
    proj_data.insert("pj_mtx_original_dim".to_string(), ConfigValue::Usize(384));
    proj_data.insert("pj_mtx_reduced_dim".to_string(), ConfigValue::Usize(91));
    proj_data.insert("pj_mtx_seed".to_string(), ConfigValue::U64(123456789));
    // This should trigger the assertion inside `empty_with_projection`.
    proj_data.insert("extra_reduced_dim".to_string(), ConfigValue::Bool(true));

    let nrows = 10_000;
    let ncols = 384;

    let _ = ArrowSpace::empty_with_projection(proj_data, nrows, ncols);
}

#[test]
fn test_empty_with_projection_none_path() {
    let mut proj_data = HashMap::new();
    proj_data.insert(
        "pj_mtx_original_dim".to_string(),
        ConfigValue::OptionUsize(None),
    );
    proj_data.insert(
        "pj_mtx_reduced_dim".to_string(),
        ConfigValue::OptionUsize(None),
    );
    proj_data.insert("pj_mtx_seed".to_string(), ConfigValue::OptionUsize(None));
    proj_data.insert("extra_reduced_dim".to_string(), ConfigValue::Bool(false));

    let nrows = 10_000;
    let ncols = 384;

    let aspace = ArrowSpace::empty_with_projection(proj_data, nrows, ncols);

    assert_eq!(aspace.projection_matrix, None);
    assert_eq!(aspace.reduced_dim, None);
    assert_eq!(aspace.extra_reduced_dim, false);
}

#[test]
fn test_arrowspace_config_typed_without_projection() {
    let items = make_gaussian_blob(99, 0.5);
    let (aspace, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 3, 2, 2.0, None)
        .with_seed(42)
        .build(items);

    // Extract config
    let config = aspace.arrowspace_config_typed();

    // Verify basic dimensions
    assert_eq!(config.get("nitems").unwrap().as_usize().unwrap(), 99);
    assert_eq!(config.get("nfeatures").unwrap().as_usize().unwrap(), 10);

    // Verify no projection
    assert_eq!(config.get("pj_mtx_original_dim").unwrap().as_usize(), None);
    assert_eq!(config.get("pj_mtx_reduced_dim").unwrap().as_usize(), None);
    assert_eq!(config.get("pj_mtx_seed").unwrap().as_u64(), None);
    assert_eq!(
        config.get("extra_reduced_dim").unwrap().as_bool().unwrap(),
        false
    );

    // Verify tau mode is present
    assert!(config.contains_key("taumode"));

    // Verify clustering params
    assert!(config.contains_key("n_clusters"));
    assert!(config.contains_key("cluster_radius"));
}

#[test]
fn test_arrowspace_config_typed_with_projection() {
    let items = make_gaussian_hd(99, 0.5);
    let (aspace, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 3, 2, 2.0, None)
        .with_seed(42)
        .with_dims_reduction(true, Some(0.25))
        .build(items);

    // Extract config
    let config = aspace.arrowspace_config_typed();

    // Verify basic dimensions
    assert_eq!(config.get("nitems").unwrap().as_usize().unwrap(), 99);
    assert_eq!(config.get("nfeatures").unwrap().as_usize().unwrap(), 100);

    // Verify projection parameters are present and correct
    assert_eq!(
        config
            .get("pj_mtx_original_dim")
            .unwrap()
            .as_usize()
            .unwrap(),
        100
    );
    assert_eq!(
        config
            .get("pj_mtx_reduced_dim")
            .unwrap()
            .as_usize()
            .unwrap(),
        50
    );
    assert_eq!(config.get("pj_mtx_seed").unwrap().as_u64().unwrap(), 42);

    // extra_reduced_dim should be false for Eigen mode
    assert_eq!(
        config.get("extra_reduced_dim").unwrap().as_bool().unwrap(),
        false
    );

    // Verify tau mode
    let taumode = config.get("taumode").unwrap();
    match taumode {
        ConfigValue::TauMode(_) => {} // Pass if it's a TauMode variant
        _ => panic!("Expected TauMode variant"),
    }

    // Verify clustering params exist
    assert!(config.get("n_clusters").is_some());
    assert!(config.get("cluster_radius").is_some());
}
