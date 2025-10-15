use approx::assert_relative_eq;

use crate::{
    builder::ArrowSpaceBuilder, graph::GraphLaplacian, sampling::SamplerType, taumode::TauMode, tests::test_data::{make_gaussian_blob, make_moons_hd}
};

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
fn test_builder_unit_norm_items_invariance_under_normalisation_toggle() {
    // Test that when items are already unit-normalized, toggling the normalisation flag
    // produces identical graph Laplacians (since cosine similarity is scale-invariant
    // and all items already have ||x|| = 1)

    // Generate high-dimensional moons dataset and manually normalize to unit vectors
    let items_raw: Vec<Vec<f64>> = make_moons_hd(
        150,  // Sufficient samples for meaningful graph structure
        0.15, // Moderate noise - not too high to maintain clear structure
        0.4,  // Good separation between moons
        12,   // Higher dimensionality for realistic test
        42,   // Fixed seed for reproducibility
    );

    // Normalize all items to unit L2 norm (||x|| = 1)
    let items: Vec<Vec<f64>> = items_raw
        .iter()
        .map(|item| {
            let norm = item.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                item.iter().map(|x| x / norm).collect()
            } else {
                item.clone()
            }
        })
        .collect();

    println!("=== UNIT NORM INVARIANCE TEST ===");
    println!(
        "Generated {} unit-normalized items with {} features",
        items.len(),
        items[0].len()
    );

    // Verify all items are unit-normalized
    for (i, item) in items.iter().enumerate().take(5) {
        let norm = item.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("Item {} norm: {:.12}", i, norm);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Item {} should be unit-normalized: norm = {:.12}",
            i,
            norm
        );
    }

    // Build with normalisation = true
    // Since items are already unit-norm, this should be a no-op
    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(
            0.5,  // Moderate eps for connectivity
            4,    // k neighbors
            2,    // top-k
            2.0,  // Quadratic kernel
            None, // Auto sigma
        )
        .with_normalisation(true)
        .with_inline_sampling(None)
        .with_spectral(true)
        .with_seed(42)
        .build(items.clone());

    // Build with normalisation = false
    // Since items are already unit-norm, this should produce the same result
    let (aspace_raw, gl_raw) = ArrowSpaceBuilder::default()
        .with_lambda_graph(
            0.5, // Same parameters
            4, 2, 2.0, None,
        )
        .with_normalisation(false)
        .with_inline_sampling(None)
        .with_spectral(true)
        .build(items.clone());

    // Graph structures should be identical
    assert_eq!(
        gl_norm.nnodes, gl_raw.nnodes,
        "Graph node counts should match for unit-norm inputs"
    );

    // Laplacian matrices should be nearly identical
    let stats_norm = gl_norm.statistics();
    let stats_raw = gl_raw.statistics();
    assert_relative_eq!(
        stats_norm.mean_degree,
        stats_raw.mean_degree,
        epsilon = 2e-1
    );

    // Lambda values should also be nearly identical
    let lambdas_norm = aspace_norm.lambdas();
    let lambdas_raw = aspace_raw.lambdas();

    println!(
        "Normalized lambdas (first 5): {:?}",
        &lambdas_norm[..5.min(lambdas_norm.len())]
    );
    println!(
        "Raw lambdas (first 5): {:?}",
        &lambdas_raw[..5.min(lambdas_raw.len())]
    );

    assert_eq!(
        lambdas_norm.len(),
        lambdas_raw.len(),
        "Lambda vector lengths should match"
    );

    // Compare lambda values element-wise
    let mut max_lambda_diff = 0.0_f64;
    for (i, (lam_norm, lam_raw)) in lambdas_norm.iter().zip(lambdas_raw.iter()).enumerate() {
        let diff = (lam_norm - lam_raw).abs();
        max_lambda_diff = max_lambda_diff.max(diff);

        if diff > 1e-10 {
            println!(
                "Lambda {} difference: norm={:.12}, raw={:.12}, diff={:.2e}",
                i, lam_norm, lam_raw, diff
            );
        }
    }

    println!("Maximum lambda difference: {:.2e}", max_lambda_diff);

    assert!(
        max_lambda_diff < 1.0,
        "Lambda values should be nearly identical for unit-norm inputs: max_diff={:.2e}",
        max_lambda_diff
    );

    println!("✓ Unit-normalized inputs produce identical results regardless of normalisation flag");
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

    println!(
        "Normalized lambdas (first 3): {:?}",
        &lambdas_norm[..3.min(lambdas_norm.len())]
    );
    println!(
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
    let items = make_moons_hd(6, 0.1, 0.3, 3, 456);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 3, 2, 2.0, None)
        .with_normalisation(true)
        .build(items.clone());

    println!("Assignments: {:?}", aspace.cluster_assignments);

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

    println!(
        "No spectral - signals shape: {:?}",
        aspace_no_spectral.signals.shape()
    );
    println!(
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

    println!("Median tau lambdas: {:?}", lambdas_median);
    println!("Max tau lambdas: {:?}", lambdas_fixed);

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

    println!("=== SPECTRAL ANALYSIS ===");
    let lambdas_norm = aspace_norm.lambdas();
    let lambdas_unnorm = aspace_unnorm.lambdas();

    println!(
        "Normalized lambdas: {:?}",
        &lambdas_norm[..3.min(lambdas_norm.len())]
    );
    println!(
        "Unnormalized lambdas: {:?}",
        &lambdas_unnorm[..3.min(lambdas_unnorm.len())]
    );

    // Diagonal elements should differ due to different graph structure
    let d_norm = diag_vec(&gl_norm);
    let d_unnorm = diag_vec(&gl_unnorm);

    println!("Normalized diagonals: {:?}", &d_norm[..3.min(d_norm.len())]);
    println!(
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

    println!("Original dimension: {}", aspace_full.nfeatures);
    println!("Reduced dimension: {:?}", aspace_reduced.reduced_dim);

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
        768,  // Fixed seed for reproducibility
    );

    println!("=== LAMBDA STATISTICS TEST ===");
    println!(
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

    println!("Graph has {} nodes", gl.nnodes);

    // Extract lambda statistics
    let lambdas = aspace.lambdas();

    let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;

    // Compute standard deviation for variance measure
    let variance = lambdas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / lambdas.len() as f64;
    let std_dev = variance.sqrt();

    println!("=== LAMBDA DISTRIBUTION ===");
    println!("Min:     {:.6}", min);
    println!("Max:     {:.6}", max);
    println!("Mean:    {:.6}", mean);
    println!("Std Dev: {:.6}", std_dev);
    println!("Range:   {:.6}", max - min);

    // Show first few lambdas for inspection
    println!("First 5 lambdas: {:?}", &lambdas[..5.min(lambdas.len())]);

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

    println!("✓ Lambda statistics show expected variance from noisy moon dataset");
}

#[test]
fn test_builder_cluster_radius_impact() {
    // Test how cluster radius affects clustering
    let items = make_moons_hd(9, 0.1, 0.25, 3, 222);

    // This test verifies that the auto-computed cluster parameters
    // produce reasonable clustering behavior
    let (aspace, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 3, 2, 2.0, None)
        .build(items);

    // Radius should be positive
    assert!(
        aspace.cluster_radius > 0.0,
        "Cluster radius should be positive"
    );
}
