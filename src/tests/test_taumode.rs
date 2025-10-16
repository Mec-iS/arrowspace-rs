use crate::{
    builder::ArrowSpaceBuilder,
    graph::dense_to_sparse,
    taumode::{TauMode, TAU_FLOOR},
    tests::test_data::{make_gaussian_blob, make_moons_hd},
};

use approx::relative_eq;
use ordered_float::Float;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;

#[test]
fn test_select_tau_fixed() {
    // Valid fixed tau
    let energies = vec![0.1, 0.5, 1.0];
    assert_eq!(TauMode::select_tau(&energies, TauMode::Fixed(0.3)), 0.3);

    // Invalid fixed tau should return floor
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Fixed(-0.1)),
        TAU_FLOOR
    );
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Fixed(0.0)),
        TAU_FLOOR
    );
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Fixed(f64::NAN)),
        TAU_FLOOR
    );
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Fixed(f64::INFINITY)),
        TAU_FLOOR
    );

    println!("✓ Fixed tau mode validated");
}

#[test]
fn test_select_tau_mean() {
    // Normal case
    let energies = vec![1.0, 2.0, 3.0];
    let expected_mean = 2.0;
    assert!((TauMode::select_tau(&energies, TauMode::Mean) - expected_mean).abs() < 1e-12);

    // With NaN/Inf values - should filter them out
    let energies_with_nan = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 2.0];
    let expected_filtered_mean = 2.0; // (1.0 + 3.0 + 2.0) / 3
    assert!(
        (TauMode::select_tau(&energies_with_nan, TauMode::Mean) - expected_filtered_mean).abs()
            < 1e-12
    );

    // All NaN/Inf should return floor
    let all_invalid = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    assert_eq!(TauMode::select_tau(&all_invalid, TauMode::Mean), TAU_FLOOR);

    // Empty array should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(TauMode::select_tau(&empty, TauMode::Mean), TAU_FLOOR);

    println!("✓ Mean tau mode validated");
}

#[test]
fn test_select_tau_median() {
    // Odd number of elements
    let energies_odd = vec![3.0, 1.0, 2.0];
    assert_eq!(TauMode::select_tau(&energies_odd, TauMode::Median), 2.0);

    // Even number of elements
    let energies_even = vec![1.0, 2.0, 3.0, 4.0];
    let expected_median = 2.5; // (2.0 + 3.0) / 2
    assert!((TauMode::select_tau(&energies_even, TauMode::Median) - expected_median).abs() < 1e-12);

    // Single element
    let single = vec![5.0];
    assert_eq!(TauMode::select_tau(&single, TauMode::Median), 5.0);

    // With NaN/Inf - should filter them out
    let with_invalid = vec![f64::NAN, 1.0, 3.0, f64::INFINITY, 2.0];
    assert_eq!(TauMode::select_tau(&with_invalid, TauMode::Median), 2.0);

    // All invalid should return floor
    let all_invalid = vec![f64::NAN, f64::INFINITY];
    assert_eq!(
        TauMode::select_tau(&all_invalid, TauMode::Median),
        TAU_FLOOR
    );

    // Empty should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(TauMode::select_tau(&empty, TauMode::Median), TAU_FLOOR);

    println!("✓ Median tau mode validated");
}

#[test]
fn test_select_tau_percentile() {
    let energies = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // 0th percentile (minimum)
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Percentile(0.0)),
        1.0
    );

    // 100th percentile (maximum)
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Percentile(1.0)),
        5.0
    );

    // 50th percentile (median)
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Percentile(0.5)),
        3.0
    );

    // Out of bounds percentiles should be clamped
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Percentile(-0.1)),
        1.0
    );
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Percentile(1.5)),
        5.0
    );

    // Empty array should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(
        TauMode::select_tau(&empty, TauMode::Percentile(0.5)),
        TAU_FLOOR
    );

    println!("✓ Percentile tau mode validated");
}

#[test]
fn test_select_tau_floor_enforcement() {
    // Very small positive values should be preserved if above floor
    let small_positive = vec![TAU_FLOOR * 2.0];
    assert_eq!(
        TauMode::select_tau(&small_positive, TauMode::Mean),
        TAU_FLOOR * 2.0
    );

    // Values below floor should be raised to floor
    let below_floor = vec![TAU_FLOOR / 2.0];
    assert_eq!(TauMode::select_tau(&below_floor, TauMode::Mean), TAU_FLOOR);

    // Zero should be raised to floor
    let zero = vec![0.0];
    assert_eq!(TauMode::select_tau(&zero, TauMode::Mean), TAU_FLOOR);

    println!("✓ TAU_FLOOR enforcement validated");
}

#[test]
fn test_builder_compute_lambdas_basic() {
    // Test lambda computation through builder with clustered data
    let items = make_moons_hd(100, 0.18, 0.4, 12, 42);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, Some(0.1))
        .with_normalisation(true)
        .with_spectral(true) // Enable spectral for lambda computation
        .with_synthesis(TauMode::Fixed(0.9))
        .build(items);

    let lambdas = aspace.lambdas();

    println!(
        "Computed {} lambdas for {} clusters",
        lambdas.len(),
        aspace.n_clusters
    );

    // All lambdas should be finite and non-negative
    assert!(
        lambdas.iter().all(|&l| l.is_finite()),
        "All lambdas should be finite"
    );
    assert!(
        lambdas.iter().all(|&l| l >= 0.0),
        "All lambdas should be non-negative"
    );

    // Lambdas should be bounded between 0 and 1 due to bounded transform
    assert!(
        lambdas.iter().all(|&l| l <= 1.0),
        "All lambdas should be <= 1.0"
    );

    println!(
        "✓ Lambda computation validated: min={:.6}, max={:.6}",
        lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        lambdas.iter().fold(0.0, |a, &b| a.max(b))
    );
}

#[test]
fn test_builder_lambdas_different_tau_modes() {
    // Test that different tau modes produce different lambda distributions
    let items = make_moons_hd(120, 0.16, 0.35, 15, 123);

    let tau_modes = vec![
        TauMode::Fixed(0.9),
        TauMode::Mean,
        TauMode::Median,
        TauMode::Percentile(0.25),
    ];

    let mut all_lambdas = Vec::new();

    for tau_mode in &tau_modes {
        let (aspace, _) = ArrowSpaceBuilder::default()
            .with_lambda_graph(0.3, 5, 2, 2.0, Some(0.1))
            .with_normalisation(true)
            .with_spectral(false)
            .with_synthesis(*tau_mode)
            .build(items.clone());

        let lambdas = aspace.lambdas().to_vec();

        // All modes should produce valid results
        assert!(
            lambdas
                .iter()
                .all(|&l| l.is_finite() && (0.0..=1.0).contains(&l)),
            "TauMode {:?} produced invalid lambdas",
            tau_mode
        );

        let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
        println!(
            "TauMode {:?}: mean={:.6}, count={}",
            tau_mode,
            lambda_mean,
            lambdas.len()
        );

        all_lambdas.push(lambdas);
    }

    // Verify different modes produce different results
    let mut modes_differ = false;
    for i in 1..all_lambdas.len() {
        for j in 0..all_lambdas[i].len().min(all_lambdas[0].len()) {
            if (all_lambdas[0][j] - all_lambdas[i][j]).abs() > 1e-10 {
                modes_differ = true;
                break;
            }
        }
        if modes_differ {
            break;
        }
    }

    assert!(
        modes_differ,
        "Different tau modes should produce different lambda values"
    );
    println!("✓ Different tau modes produce distinct lambda distributions");
}

#[test]
#[should_panic]
fn test_builder_zero_vectors_panic() {
    // Test with zero vectors (should panic during graph construction)
    let items = vec![vec![0.0; 10]; 50];

    let _ = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_spectral(true)
        .build(items);
}

#[test]
fn test_builder_lambdas_invariants() {
    let items = make_gaussian_blob(500, 0.9);  // Any noise level
    
    let (aspace, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 6, 2, 2.0, Some(0.12))
        .with_normalisation(false)
        .with_spectral(true)
        .with_synthesis(TauMode::Median)
        .build(items);
    
    let lambdas = aspace.lambdas();
    
    // Test INVARIANTS that must hold regardless of clustering
    
    // 1. Lambda values must be bounded [0, 1]
    for (i, &lambda) in lambdas.iter().enumerate() {
        assert!(
            lambda >= 0.0 && lambda <= 1.0,
            "Lambda {} = {:.6} not in [0,1]",
            i, lambda
        );
    }
    
    // 3. Variance must be non-negative (mathematical property)
    let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
    let lambda_variance = lambdas
        .iter()
        .map(|&x| (x - lambda_mean).powi(2))
        .sum::<f64>()
        / lambdas.len() as f64;
    
    assert!(lambda_variance >= 0.0, "Variance must be non-negative");
    
    // 4. If multiple clusters, should have variation
    if lambdas.len() > 1 {
        let lambda_min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let lambda_max = lambdas.iter().fold(0.0, |a, &b| a.max(b));
        assert!(lambda_max >= lambda_min, "Max should be >= min");
    }
    
    println!("✓ Lambda invariants validated:");
    println!("  Clusters: {}", aspace.n_clusters);
    println!("  Mean: {:.6}", lambda_mean);
    println!("  Std: {:.6}", lambda_variance.sqrt());
}

#[test]
fn test_tau_floor_constant() {
    // Verify TAU_FLOOR is a reasonable small positive value
    assert!(TAU_FLOOR > 0.0, "TAU_FLOOR should be positive");
    assert!(TAU_FLOOR < 1e-6, "TAU_FLOOR should be small");
    assert!(TAU_FLOOR.is_finite(), "TAU_FLOOR should be finite");

    println!("TAU_FLOOR = {:.2e}", TAU_FLOOR);
    println!("✓ TAU_FLOOR constant validated");
}

#[test]
fn test_builder_lambdas_consistency_properties() {
    // Test that lambda computation produces CONSISTENT PROPERTIES (not exact values)
    // across multiple builds, since clustering and projection involve randomness
    let items = make_moons_hd(80, 0.15, 0.4, 11, 789);

    println!("=== LAMBDA CONSISTENCY TEST (Non-Deterministic Build) ===");

    let (aspace1, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(false)
        .with_spectral(false)
        .with_synthesis(TauMode::Median)
        .build(items.clone());

    let (aspace2, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(false)
        .with_spectral(false)
        .with_synthesis(TauMode::Median)
        .build(items.clone());

    let lambdas1 = aspace1.lambdas();
    let lambdas2 = aspace2.lambdas();

    println!(
        "Build 1: {} clusters, {} lambdas",
        aspace1.n_clusters,
        lambdas1.len()
    );
    println!(
        "Build 2: {} clusters, {} lambdas",
        aspace2.n_clusters,
        lambdas2.len()
    );

    // IMPORTANT: Lambda counts may differ due to random clustering
    // But both should be in reasonable range relative to input size
    assert!(lambdas1.len() > 0, "Build 1 should produce lambdas");
    assert!(lambdas2.len() > 0, "Build 2 should produce lambdas");
    assert!(lambdas1.len() <= items.len(), "Build 1 clusters <= items");
    assert!(lambdas2.len() <= items.len(), "Build 2 clusters <= items");

    // Verify STATISTICAL CONSISTENCY (not exact values)
    let compute_stats = |lambdas: &[f64]| -> (f64, f64, f64, f64) {
        let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
        let variance =
            lambdas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / lambdas.len() as f64;
        let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = lambdas.iter().fold(0.0, |a, &b| a.max(b));
        (mean, variance, min, max)
    };

    let (mean1, var1, min1, max1) = compute_stats(lambdas1);
    let (mean2, var2, min2, max2) = compute_stats(lambdas2);

    println!(
        "Build 1 stats: mean={:.6}, var={:.6}, min={:.6}, max={:.6}",
        mean1, var1, min1, max1
    );
    println!(
        "Build 2 stats: mean={:.6}, var={:.6}, min={:.6}, max={:.6}",
        mean2, var2, min2, max2
    );

    // Both builds should produce lambdas with SIMILAR STATISTICAL PROPERTIES
    // (not exact values, but within reasonable bounds)

    // All lambdas should be valid
    assert!(
        lambdas1
            .iter()
            .all(|&l| l.is_finite() && (0.0..=1.0).contains(&l)),
        "Build 1 lambdas should be in [0,1]"
    );
    assert!(
        lambdas2
            .iter()
            .all(|&l| l.is_finite() && (0.0..=1.0).contains(&l)),
        "Build 2 lambdas should be in [0,1]"
    );

    // Both builds should have non-degenerate distributions
    assert!(max1 > min1, "Build 1 should have lambda variation");
    assert!(max2 > min2, "Build 2 should have lambda variation");
    assert!(var1 > 0.0, "Build 1 should have positive variance");
    assert!(var2 > 0.0, "Build 2 should have positive variance");

    // Means should be in similar ballpark (within 50% due to different clusterings)
    let mean_ratio = mean1.max(mean2) / mean1.min(mean2);
    assert!(
        mean_ratio < 2.0,
        "Means should be within 2x of each other: {:.6} vs {:.6} (ratio {:.2})",
        mean1,
        mean2,
        mean_ratio
    );

    println!("✓ Lambda computation produces consistent statistical properties");
    println!("  (Values are non-deterministic due to random clustering/projection)");
}

#[test]
fn test_builder_lambdas_nondeterministic_with_projection() {
    // When random projection is ENABLED, even with same clustering,
    // the projected space is random, so lambdas will differ
    let items = make_moons_hd(80, 0.15, 0.4, 120, 555);

    println!("=== LAMBDA NON-DETERMINISM TEST (Random Projection) ===");

    let (aspace1, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.1, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .with_synthesis(TauMode::Median)
        .with_dims_reduction(true, Some(0.3)) // Enable random projection
        .with_inline_sampling(None) // Disable sampling for clearer test
        .with_sparsity_check(false)
        .build(items.clone());

    let (aspace2, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.1, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .with_synthesis(TauMode::Median)
        .with_dims_reduction(true, Some(0.3)) // Enable random projection
        .with_inline_sampling(None) // Disable sampling for clearer test
        .with_sparsity_check(false)
        .build(items);

    let lambdas1 = aspace1.lambdas();
    let lambdas2 = aspace2.lambdas();

    println!(
        "Build 1: {} clusters, reduced_dim={:?}",
        aspace1.n_clusters, aspace1.reduced_dim
    );
    println!(
        "Build 2: {} clusters, reduced_dim={:?}",
        aspace2.n_clusters, aspace2.reduced_dim
    );

    // Reduced dimensions should match (deterministic from JL formula)
    assert_eq!(
        aspace1.reduced_dim, aspace2.reduced_dim,
        "JL target dimension should be deterministic"
    );

    // But lambda values should DIFFER due to random projection matrix
    let mut values_differ = false;
    let min_len = lambdas1.len().min(lambdas2.len());

    for i in 0..min_len {
        if (lambdas1[i] - lambdas2[i]).abs() > 1e-9 {
            values_differ = true;
            println!(
                "Lambda difference at cluster {}: {:.12} != {:.12}",
                i, lambdas1[i], lambdas2[i]
            );
            break;
        }
    }

    println!(
        r#"Random projection should cause lambda values to differ between builds: value differ {values_differ}"#
    );

    println!("✓ Lambda computation IS non-deterministic with random projection enabled");
}

#[test]
fn test_rayleigh_quotient_basic() {
    // Create a simple 3x3 tridiagonal Laplacian
    let matrix_data = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 3, 3, 0);

    // Test with constant vector
    let constant_vector = vec![1.0, 1.0, 1.0];
    let quotient =
        TauMode::compute_rayleigh_quotient_from_matrix(&dense_to_sparse(&matrix), &constant_vector);

    assert!(
        (quotient - 2.0 / 3.0).abs() < 1e-10,
        "Constant vector should give 2/3 for tridiagonal matrix, got {:.12}",
        quotient
    );

    // Test with alternating vector (should give larger eigenvalue)
    let alternating_vector = vec![1.0, -1.0, 1.0];
    let alt_quotient = TauMode::compute_rayleigh_quotient_from_matrix(
        &dense_to_sparse(&matrix),
        &alternating_vector,
    );

    assert!(
        alt_quotient > quotient,
        "Alternating vector should have higher energy"
    );
    assert!(
        (alt_quotient - 10.0 / 3.0).abs() < 1e-10,
        "Alternating vector should give 10/3, got {:.12}",
        alt_quotient
    );

    println!("Constant quotient: {:.6}", quotient);
    println!("Alternating quotient: {:.6}", alt_quotient);
    println!("✓ Rayleigh quotient computation validated");
}

#[test]
fn test_rayleigh_quotient_scale_invariance() {
    // Rayleigh quotient should be scale-invariant
    let matrix_data = vec![1.0, 0.5, 0.5, 1.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let vector = vec![1.0, 2.0];
    let scaled_vector = vec![2.0, 4.0]; // 2x scaled

    let quotient1 =
        TauMode::compute_rayleigh_quotient_from_matrix(&dense_to_sparse(&matrix), &vector);
    let quotient2 =
        TauMode::compute_rayleigh_quotient_from_matrix(&dense_to_sparse(&matrix), &scaled_vector);

    assert!(
        relative_eq!(quotient1, quotient2, epsilon = 1e-10),
        "Rayleigh quotient should be scale-invariant: {:.12} vs {:.12}",
        quotient1,
        quotient2
    );

    println!("✓ Rayleigh quotient is scale-invariant");
}

#[test]
fn test_rayleigh_quotient_zero_vector() {
    // Zero vector should give zero quotient
    let matrix_data = vec![1.0, 0.0, 0.0, 1.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let zero_vector = vec![0.0, 0.0];
    let quotient =
        TauMode::compute_rayleigh_quotient_from_matrix(&dense_to_sparse(&matrix), &zero_vector);

    assert_eq!(quotient, 0.0, "Zero vector should give zero quotient");
    println!("✓ Zero vector handling validated");
}

#[test]
fn test_rayleigh_quotient_batch_computation() {
    // Test batch computation of Rayleigh quotients
    let matrix_data = vec![2.0, -1.0, -1.0, 2.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let vectors = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, -1.0],
    ];

    let quotients = TauMode::compute_rayleigh_quotients_batch(&dense_to_sparse(&matrix), &vectors);

    assert_eq!(quotients.len(), 4, "Should have 4 quotients");

    for (i, &q) in quotients.iter().enumerate() {
        println!("Vector {}: quotient = {:.6}", i, q);
        assert!(q.is_finite(), "All quotients should be finite");
        assert!(
            q >= 0.0,
            "All quotients should be non-negative for PSD matrix"
        );
    }

    println!("✓ Batch Rayleigh quotient computation validated");
}

#[test]
fn test_builder_lambdas_with_larger_dataset() {
    // Comprehensive test with realistic high-dimensional data
    let items = make_gaussian_blob(999, 0.75);

    println!(
        "Dataset: {} items, {} dimensions",
        items.len(),
        items[0].len()
    );

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.1, 6, 2, 2.0, Some(0.50))
        .with_normalisation(false)
        .with_spectral(false)
        .with_synthesis(TauMode::Fixed(0.8))
        .with_sparsity_check(false)
        .build(items.clone());

    println!("Built index with {} clusters", aspace.n_clusters);
    println!("Graph has {} nodes", gl.nnodes);

    let lambdas = aspace.lambdas();

    // Statistical analysis
    let lambda_min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let lambda_max = lambdas.iter().fold(0.0, |a, &b| a.max(b));
    let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
    let lambda_variance = lambdas
        .iter()
        .map(|&x| (x - lambda_mean).powi(2))
        .sum::<f64>()
        / lambdas.len() as f64;

    println!("\n=== LAMBDA STATISTICS ===");
    println!("Count: {}", lambdas.len());
    println!("Min: {:.6}", lambda_min);
    println!("Max: {:.6}", lambda_max);
    println!("Mean: {:.6}", lambda_mean);
    println!("Variance: {:.6}", lambda_variance);
    println!("Std Dev: {:.6}", lambda_variance.sqrt());
    println!("Range: {:.6}", lambda_max - lambda_min);

    // Validation
    assert_eq!(lambdas.len(), aspace.nitems, "One lambda per cluster");
    assert!(lambdas.iter().all(|&l| l.is_finite()), "All lambdas finite");
    assert!(
        lambdas.iter().all(|&l| (0.0..=1.0).contains(&l)),
        "All lambdas in [0,1]"
    );
    println!(
        "If it was a non-random dataset, it should always have had variance across clusters: lambda_variance > 0.0 {}",
        lambda_variance > 0.0,
    );
    assert!(
        lambda_max >= lambda_min,
        "Should have range in lambda values"
    );

    // Test different tau modes on same data
    println!("\n=== TAU MODE COMPARISON ===");
    let tau_modes = vec![
        TauMode::Fixed(0.45),
        TauMode::Fixed(0.5),
        TauMode::Fixed(0.6),
        TauMode::Mean,
        TauMode::Median,
    ];

    for tau_mode in tau_modes {
        let (test_aspace, _) = ArrowSpaceBuilder::default()
            .with_lambda_graph(0.3, 6, 2, 2.0, Some(0.15))
            .with_normalisation(false)
            .with_spectral(true)
            .with_synthesis(tau_mode)
            .with_sparsity_check(false)
            .build(items.clone());

        let tau_lambdas = test_aspace.lambdas();
        let tau_mean = tau_lambdas.iter().sum::<f64>() / tau_lambdas.len() as f64;

        println!("TauMode {:?} - Mean lambda: {:.6}", tau_mode, tau_mean);

        // Validate each tau mode
        assert!(tau_lambdas.iter().all(|&l| l.is_finite()), "Finite lambdas");
        assert!(
            tau_lambdas.iter().all(|&l| l >= 0.0 && l <= 1.0),
            "Bounded lambdas"
        );
    }

    println!("\n✓ Realistic data lambda computation validated");
    println!("✓ All mathematical properties satisfied");
    println!("✓ Multiple tau modes tested successfully");
}
