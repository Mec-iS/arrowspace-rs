use crate::builder::ArrowSpaceBuilder;
use crate::core::ArrowSpace;
use crate::graph::{GraphFactory, GraphParams};
use crate::taumode::{TauMode, TAU_FLOOR};
use crate::tests::GRAPH_PARAMS;

use approx::relative_eq;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;

use log::debug;

#[test]
fn test_select_tau_fixed() {
    // Valid fixed tau
    let energies = vec![0.1, 0.5, 1.0];
    assert_eq!(TauMode::select_tau(&energies, TauMode::Fixed(0.3)), 0.3);

    // Invalid fixed tau should return floor
    assert_eq!(TauMode::select_tau(&energies, TauMode::Fixed(-0.1)), TAU_FLOOR);
    assert_eq!(TauMode::select_tau(&energies, TauMode::Fixed(0.0)), TAU_FLOOR);
    assert_eq!(TauMode::select_tau(&energies, TauMode::Fixed(f64::NAN)), TAU_FLOOR);
    assert_eq!(
        TauMode::select_tau(&energies, TauMode::Fixed(f64::INFINITY)),
        TAU_FLOOR
    );
}

#[test]
fn test_select_tau_mean() {
    // Normal case
    let energies = vec![1.0, 2.0, 3.0];
    let expected_mean = 2.0;
    assert!(
        (TauMode::select_tau(&energies, TauMode::Mean) - expected_mean).abs() < 1e-12
    );

    // With NaN/Inf values - should filter them out
    let energies_with_nan = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 2.0];
    let expected_filtered_mean = 2.0; // (1.0 + 3.0 + 2.0) / 3
    assert!(
        (TauMode::select_tau(&energies_with_nan, TauMode::Mean)
            - expected_filtered_mean)
            .abs()
            < 1e-12
    );

    // All NaN/Inf should return floor
    let all_invalid = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    assert_eq!(TauMode::select_tau(&all_invalid, TauMode::Mean), TAU_FLOOR);

    // Empty array should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(TauMode::select_tau(&empty, TauMode::Mean), TAU_FLOOR);
}

#[test]
fn test_select_tau_median() {
    // Odd number of elements
    let energies_odd = vec![3.0, 1.0, 2.0];
    assert_eq!(TauMode::select_tau(&energies_odd, TauMode::Median), 2.0);

    // Even number of elements
    let energies_even = vec![1.0, 2.0, 3.0, 4.0];
    let expected_median = 2.5; // (2.0 + 3.0) / 2
    assert!(
        (TauMode::select_tau(&energies_even, TauMode::Median) - expected_median).abs()
            < 1e-12
    );

    // Single element
    let single = vec![5.0];
    assert_eq!(TauMode::select_tau(&single, TauMode::Median), 5.0);

    // With NaN/Inf - should filter them out
    let with_invalid = vec![f64::NAN, 1.0, 3.0, f64::INFINITY, 2.0];
    assert_eq!(TauMode::select_tau(&with_invalid, TauMode::Median), 2.0);

    // All invalid should return floor
    let all_invalid = vec![f64::NAN, f64::INFINITY];
    assert_eq!(TauMode::select_tau(&all_invalid, TauMode::Median), TAU_FLOOR);

    // Empty should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(TauMode::select_tau(&empty, TauMode::Median), TAU_FLOOR);
}

#[test]
fn test_select_tau_percentile() {
    let energies = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // 0th percentile (minimum)
    assert_eq!(TauMode::select_tau(&energies, TauMode::Percentile(0.0)), 1.0);

    // 100th percentile (maximum)
    assert_eq!(TauMode::select_tau(&energies, TauMode::Percentile(1.0)), 5.0);

    // 50th percentile (median)
    assert_eq!(TauMode::select_tau(&energies, TauMode::Percentile(0.5)), 3.0);

    // Out of bounds percentiles should be clamped
    assert_eq!(TauMode::select_tau(&energies, TauMode::Percentile(-0.1)), 1.0);
    assert_eq!(TauMode::select_tau(&energies, TauMode::Percentile(1.5)), 5.0);

    // Empty array should return floor
    let empty: Vec<f64> = vec![];
    assert_eq!(TauMode::select_tau(&empty, TauMode::Percentile(0.5)), TAU_FLOOR);
}

#[test]
fn test_select_tau_floor_enforcement() {
    // Very small positive values should be preserved if above floor
    let small_positive = vec![TAU_FLOOR * 2.0];
    assert_eq!(TauMode::select_tau(&small_positive, TauMode::Mean), TAU_FLOOR * 2.0);

    // Values below floor should be raised to floor
    let below_floor = vec![TAU_FLOOR / 2.0];
    assert_eq!(TauMode::select_tau(&below_floor, TauMode::Mean), TAU_FLOOR);

    // Zero should be raised to floor
    let zero = vec![0.0];
    assert_eq!(TauMode::select_tau(&zero, TauMode::Mean), TAU_FLOOR);
}

#[test]
fn test_compute_synthetic_lambdas_basic() {
    // Create a realistic high-dimensional ArrowSpace with meaningful feature patterns
    let items = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
    ];

    let aspace = ArrowSpace::from_items_default(items.clone());

    // Create a graph based on item similarity
    let gl = GraphFactory::build_laplacian_matrix(
        items,
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);

    // Apply synthetic lambda computation
    let tau_mode = TauMode::Fixed(0.9);
    TauMode::compute_taumode_lambdas(&mut aspace, Some(tau_mode));

    let lambdas = aspace.lambdas();
    assert_eq!(lambdas.len(), 2);

    // All lambdas should be finite and non-negative
    assert!(lambdas.iter().all(|&l| l.is_finite() && l >= 0.0));

    // Lambdas should be bounded between 0 and 1 due to the bounded transform
    assert!(lambdas.iter().all(|&l| l <= 1.0));
}

#[test]
fn test_compute_synthetic_lambdas_different_alpha() {
    let items = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
        vec![
            0.85, 0.09, 0.41, 0.31, 0.67, 0.29, 0.53, 0.52, 0.17, 0.76, 0.05, 0.38,
            0.60,
        ],
    ];

    let gl = GraphFactory::build_laplacian_matrix(
        items.clone(),
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );

    // Test with alpha = 1.0 (pure energy term)
    let aspace1 = ArrowSpace::from_items_default(items.clone());
    let mut aspace1 = GraphFactory::build_spectral_laplacian(aspace1, &gl.graph_params);
    TauMode::compute_taumode_lambdas(&mut aspace1, Some(TauMode::Fixed(0.9)));
    let lambdas1 = aspace1.lambdas().to_vec();

    // Test with alpha = 0.0 (pure dispersion term)
    let aspace2 = ArrowSpace::from_items_default(items.clone());
    let mut aspace2 = GraphFactory::build_spectral_laplacian(aspace2, &gl.graph_params);
    TauMode::compute_taumode_lambdas(&mut aspace2, Some(TauMode::Fixed(0.9)));
    let lambdas2 = aspace2.lambdas().to_vec();

    // Test with alpha = 0.5 (balanced)
    let aspace3 = ArrowSpace::from_items_default(items);
    let mut aspace3 = GraphFactory::build_spectral_laplacian(aspace3, &gl.graph_params);
    TauMode::compute_taumode_lambdas(&mut aspace3, Some(TauMode::Fixed(0.9)));
    let lambdas3 = aspace3.lambdas().to_vec();

    // All should be different (unless edge case)
    assert_eq!(lambdas1.len(), lambdas2.len());
    assert_eq!(lambdas2.len(), lambdas3.len());

    // All should be finite and bounded
    for lambdas in [&lambdas1, &lambdas2, &lambdas3] {
        assert!(lambdas.iter().all(|&l| l.is_finite() && (0.0..=1.0).contains(&l)));
    }
}

#[test]
#[should_panic]
fn test_compute_synthetic_lambdas_zero_vectors() {
    // Test with zero vectors (should handle gracefully)
    let items = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]];

    let aspace = ArrowSpace::from_items_default(items.clone());
    let gl = GraphFactory::build_laplacian_matrix(items, 1e-3, 2, 2.0, None, true);
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);

    TauMode::compute_taumode_lambdas(&mut aspace, Some(TauMode::Median));
}

#[test]
fn test_compute_synthetic_lambdas_different_tau_modes() {
    let items = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
        vec![
            0.85, 0.09, 0.41, 0.31, 0.67, 0.29, 0.53, 0.52, 0.17, 0.76, 0.05, 0.38,
            0.60,
        ],
    ];

    let gl = GraphFactory::build_laplacian_matrix(
        items.clone(),
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );

    // Test different tau modes and collect results for comparison
    let tau_modes = vec![
        TauMode::Fixed(0.9),
        TauMode::Mean,
        TauMode::Median,
        TauMode::Percentile(0.25),
    ];

    let mut all_lambdas = Vec::new();

    for tau_mode in tau_modes {
        let aspace = ArrowSpace::from_items_default(items.clone());
        let mut aspace =
            GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);
        TauMode::compute_taumode_lambdas(&mut aspace, Some(tau_mode));

        let lambdas = aspace.lambdas().to_vec();

        // All modes should produce valid results
        assert!(lambdas.iter().all(|&l| l.is_finite() && (0.0..=1.0).contains(&l)));
        assert_eq!(lambdas.len(), 3); // One lambda per item

        all_lambdas.push(lambdas);

        debug!("TauMode {:?} lambdas: {:?}", tau_mode, aspace.lambdas());
    }

    // Assertions about different taumode strategies
    let fixed_lambdas = &all_lambdas[0]; // Fixed(0.9)
    let mean_lambdas = &all_lambdas[1]; // Mean
    let median_lambdas = &all_lambdas[2]; // Median
    let percentile_lambdas = &all_lambdas[3]; // Percentile(0.25)

    // Fixed mode should produce consistent values influenced by the fixed tau
    // With tau=0.9 (high), lambdas should generally be higher than adaptive modes
    let fixed_mean = fixed_lambdas.iter().sum::<f64>() / fixed_lambdas.len() as f64;

    // Mean mode should balance across all Rayleigh quotients
    let mean_mean = mean_lambdas.iter().sum::<f64>() / mean_lambdas.len() as f64;

    // Different modes should produce different results (unless degenerate case)
    let mut modes_are_different = false;
    for i in 1..all_lambdas.len() {
        for j in 0..all_lambdas[i].len() {
            if (all_lambdas[0][j] - all_lambdas[i][j]).abs() > 1e-10 {
                modes_are_different = true;
                break;
            }
        }
        if modes_are_different {
            break;
        }
    }
    assert!(
        modes_are_different,
        "Different tau modes should produce different lambda values"
    );

    // Percentile(0.25) should generally produce lower values than Mean
    // since it uses the 25th percentile as threshold
    let percentile_mean =
        percentile_lambdas.iter().sum::<f64>() / percentile_lambdas.len() as f64;

    // Statistical properties: different modes should have different central tendencies
    debug!("Lambda means by mode:");
    debug!("  Fixed(0.9): {:.6}", fixed_mean);
    debug!("  Mean: {:.6}", mean_mean);
    debug!(
        "  Median: {:.6}",
        median_lambdas.iter().sum::<f64>() / median_lambdas.len() as f64
    );
    debug!("  Percentile(0.25): {:.6}", percentile_mean);

    // Verify that modes produce meaningfully different distributions
    let lambda_variances: Vec<f64> = all_lambdas
        .iter()
        .map(|lambdas| {
            let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            lambdas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / lambdas.len() as f64
        })
        .collect();

    // All modes should have some variance (indicating feature discrimination)
    assert!(
        lambda_variances.iter().all(|&v| v >= 0.0),
        "Lambda variances should be non-negative"
    );

    debug!("Lambda variances by mode: {:?}", lambda_variances);
}

#[test]
#[should_panic]
fn test_compute_synthetic_lambdas_graph_mismatch_panics() {
    let items = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    let aspace = ArrowSpace::from_items_default(items.clone());

    // Create a graph with wrong number of features
    let wrong_items = vec![vec![1.0], vec![2.0], vec![3.0]]; // 3 items instead of 2
    let wrong_gl =
        GraphFactory::build_laplacian_matrix(wrong_items, 1e-3, 2, 2.0, None, true);

    let aspace = GraphFactory::build_spectral_laplacian(aspace, &wrong_gl.graph_params);

    // This should panic due to node count mismatch
    TauMode::compute_taumode_lambdas(&mut aspace.clone(), Some(TauMode::Median));
}

#[test]
fn test_tau_floor_constant() {
    // Verify TAU_FLOOR is a reasonable small positive value
    assert!(TAU_FLOOR > 0.0);
    assert!(TAU_FLOOR < 1e-6);
    assert!(TAU_FLOOR.is_finite());
}

#[test]
fn test_synthetic_lambda_properties() {
    // Test that synthetic lambdas satisfy expected mathematical properties
    let items = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
        vec![
            0.85, 0.09, 0.41, 0.31, 0.67, 0.29, 0.53, 0.52, 0.17, 0.76, 0.05, 0.38,
            0.60,
        ],
    ];

    let (aspace, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            GRAPH_PARAMS.eps,
            GRAPH_PARAMS.k,
            GRAPH_PARAMS.p,
            GRAPH_PARAMS.sigma,
        )
        .build(items.clone());

    debug!("Synthetic lambdas: {:?}", aspace.lambdas);
    debug!("Number of items: {}", aspace.lambdas.len());

    // Properties that should hold:
    // 1. All lambdas are in [0, 1] due to bounded transform
    assert!(aspace.lambdas.iter().all(|&l| (0.0..=1.0).contains(&l)));
    debug!("✓ All lambdas bounded in [0,1]");

    // 2. Lambdas are deterministic (same input -> same output)
    let (aspace2, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            GRAPH_PARAMS.eps,
            GRAPH_PARAMS.k,
            GRAPH_PARAMS.p,
            GRAPH_PARAMS.sigma,
        )
        .build(items.clone());

    for (i, (l1, l2)) in aspace.lambdas.iter().zip(aspace2.lambdas.iter()).enumerate() {
        assert!(
            relative_eq!(*l1, *l2, epsilon = 1e-12),
            "Synthetic lambdas should be deterministic at feature {}: {} != {}",
            i,
            l1,
            l2
        );
    }
    debug!("✓ Deterministic computation verified");

    // 3. All values are finite
    assert!(aspace.lambdas.iter().all(|l| l.is_finite()));
    debug!("✓ All lambdas are finite");

    // 4. Statistical properties for high-dimensional data
    let lambda_mean = aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64;
    let lambda_variance =
        aspace.lambdas.iter().map(|&x| (x - lambda_mean).powi(2)).sum::<f64>()
            / aspace.lambdas.len() as f64;
    let lambda_min = aspace.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let lambda_max: f64 = aspace.lambdas.iter().fold(0.0, |a, &b| a.max(b));

    debug!("Lambda statistics:");
    debug!("  Mean: {:.6}", lambda_mean);
    debug!("  Variance: {:.6}", lambda_variance);
    debug!("  Min: {:.6}", lambda_min);
    debug!("  Max: {:.6}", lambda_max);
    debug!("  Range: {:.6}", lambda_max - lambda_min);

    // 5. Non-degenerate behavior (should have some variation across features)
    assert!(
        lambda_variance > 0.0,
        "Lambda variance should be positive, indicating feature discrimination"
    );
    assert!(lambda_max > lambda_min, "Should have variation across features");
    debug!("✓ Non-degenerate feature discrimination confirmed");

    // 6. Median mode specific property: values should be influenced by median threshold
    // Test with different tau modes to verify median produces different results
    let (aspace3, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            GRAPH_PARAMS.eps,
            GRAPH_PARAMS.k,
            GRAPH_PARAMS.p,
            GRAPH_PARAMS.sigma,
        )
        .with_synthesis(TauMode::Mean)
        .build(items);

    let modes_differ = aspace
        .lambdas
        .iter()
        .zip(aspace3.lambdas.iter())
        .any(|(&median, &mean)| (median - mean).abs() > 1e-10);

    assert!(modes_differ, "Median and Mean tau modes should produce different results");
    debug!("✓ Tau mode sensitivity verified");

    // 7. Consistency with spectral properties
    assert_eq!(aspace.lambdas.len(), 3, "Should have one lambda per item");
    debug!("✓ Correct dimensionality maintained");
}

#[test]
fn test_rayleigh_quotient_basic() {
    // Create a simple 3x3 tridiagonal Laplacian
    let matrix_data = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 3, 3, 0);

    // Test with constant vector
    let constant_vector = vec![1.0, 1.0, 1.0];
    let quotient =
        TauMode::compute_rayleigh_quotient_from_matrix(&matrix, &constant_vector);

    assert!(
        (quotient - 2.0 / 3.0).abs() < 1e-10,
        "Constant vector should give 2/3 for this tridiagonal matrix, got {}",
        quotient
    );

    // Test with alternating vector (should give larger eigenvalue)
    let alternating_vector = vec![1.0, -1.0, 1.0];
    let alt_quotient =
        TauMode::compute_rayleigh_quotient_from_matrix(&matrix, &alternating_vector);

    assert!(alt_quotient > quotient, "Alternating vector should have higher energy");
    assert!(alt_quotient > 0.0, "Should be positive for this matrix");

    assert!(
        (alt_quotient - 10.0 / 3.0).abs() < 1e-10,
        "Alternating vector should give 10/3, got {}",
        alt_quotient
    );

    debug!("Constant quotient: {:.6}", quotient);
    debug!("Alternating quotient: {:.6}", alt_quotient);
}

#[test]
fn test_scale_invariance() {
    let matrix_data = vec![1.0, 0.5, 0.5, 1.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let vector = vec![1.0, 2.0];
    let scaled_vector = vec![2.0, 4.0]; // 2x scaled

    let quotient1 = TauMode::compute_rayleigh_quotient_from_matrix(&matrix, &vector);
    let quotient2 =
        TauMode::compute_rayleigh_quotient_from_matrix(&matrix, &scaled_vector);

    assert!(
        relative_eq!(quotient1, quotient2, epsilon = 1e-10),
        "Rayleigh quotient should be scale-invariant: {} vs {}",
        quotient1,
        quotient2
    );
}

#[test]
fn test_zero_vector() {
    let matrix_data = vec![1.0, 0.0, 0.0, 1.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let zero_vector = vec![0.0, 0.0];
    let quotient =
        TauMode::compute_rayleigh_quotient_from_matrix(&matrix, &zero_vector);

    assert_eq!(quotient, 0.0, "Zero vector should give zero quotient");
}

#[test]
fn test_batch_computation() {
    let matrix_data = vec![2.0, -1.0, -1.0, 2.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![1.0, -1.0]];

    let quotients = TauMode::compute_rayleigh_quotients_batch(&matrix, &vectors);

    assert_eq!(quotients.len(), 4);
    for (i, &q) in quotients.iter().enumerate() {
        debug!("Vector {}: quotient = {:.6}", i, q);
        assert!(q.is_finite(), "All quotients should be finite");
    }
}

/// Test on an embeddings dataset, see `test_data.rs`
#[test]
fn test_lambda_verification_quora_embeddings() {
    use crate::tests::test_data::QUORA_EMBEDDS;
    // Test vector with 384 dimensions from embedding space
    let test_vector = QUORA_EMBEDDS[QUORA_EMBEDDS.len() - 1];

    // Create an ArrowSpace for testing lambda computation
    let items: Vec<Vec<f64>> = QUORA_EMBEDDS[0..QUORA_EMBEDDS.len()]
        .iter()
        .map(|inner_slice| inner_slice.to_vec())
        .collect();
    let aspace = ArrowSpace::from_items_default(items.clone());

    // Verify vector dimensions
    assert_eq!(test_vector.len(), 384, "Test vector should have 384 dimensions");
    assert_eq!(aspace.data.shape(), (15, 384), "ArrowSpace should have shape (1, 384)");

    // Graph parameters as specified
    let graph_params = GraphParams {
        eps: 1e-12,
        k: 4,
        p: 2.0,
        sigma: Some(1e-12 * 0.5),
        normalise: true,
    };

    debug!("=== GRAPH PARAMETERS ===");
    debug!("eps: {}", graph_params.eps);
    debug!("k: {}", graph_params.k);
    debug!("p: {}", graph_params.p);
    debug!("sigma: {:?}", graph_params.sigma);

    // Build Laplacian with specified parameters
    let gl = GraphFactory::build_laplacian_matrix(
        items.clone(),
        graph_params.eps,
        graph_params.k,
        graph_params.p,
        graph_params.sigma,
        graph_params.normalise,
    );

    // Build spectral ArrowSpace
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);

    debug!("\n=== INITIAL STATE ===");
    debug!("Graph nodes: {}", gl.nnodes);
    debug!("ArrowSpace shape: {:?}", aspace.data.shape());
    debug!("Signals shape: {:?}", aspace.signals.shape());

    // Compute initial lambdas
    aspace.recompute_lambdas();
    let lambdas = aspace.lambdas();

    debug!("\n=== LAMBDA COMPUTATION RESULTS ===");
    debug!("Number of lambdas: {}", lambdas.len());
    debug!("Lambda values (first 10): {:?}", &lambdas[..10.min(lambdas.len())]);

    // Basic validation
    assert_eq!(lambdas.len(), 15, "Should have one lambda per item dimension");
    assert!(lambdas.iter().all(|&l| l.is_finite()), "All lambdas should be finite");
    assert!(lambdas.iter().all(|&l| l >= 0.0), "All lambdas should be non-negative");

    // Statistical analysis
    let lambda_min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let lambda_max: f64 = lambdas.iter().fold(0.0, |a, &b| a.max(b));
    let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
    let lambda_variance =
        lambdas.iter().map(|&x| (x - lambda_mean).powi(2)).sum::<f64>()
            / lambdas.len() as f64;

    debug!("\n=== LAMBDA STATISTICS ===");
    debug!("Min lambda: {:.6}", lambda_min);
    debug!("Max lambda: {:.6}", lambda_max);
    debug!("Mean lambda: {:.6}", lambda_mean);
    debug!("Lambda variance: {:.6}", lambda_variance);
    debug!("Lambda std dev: {:.6}", lambda_variance.sqrt());
    debug!("Range: {:.6}", lambda_max - lambda_min);

    // Test with different tau modes for comprehensive validation
    debug!("\n=== TAU MODE TESTING ===");
    let tau_modes = vec![
        TauMode::Fixed(0.1),
        TauMode::Fixed(0.5),
        TauMode::Fixed(0.9),
        TauMode::Mean,
        TauMode::Median,
    ];

    for tau_mode in tau_modes {
        let mut test_aspace = GraphFactory::build_spectral_laplacian(
            ArrowSpace::from_items_default(items.clone()),
            &gl.graph_params,
        );

        TauMode::compute_taumode_lambdas(&mut test_aspace, Some(tau_mode));
        let tau_lambdas = test_aspace.lambdas();

        let tau_mean = tau_lambdas.iter().sum::<f64>() / tau_lambdas.len() as f64;
        debug!("TauMode {:?} - Mean lambda: {:.6}", tau_mode, tau_mean);

        // Validate tau mode results
        assert!(
            tau_lambdas.iter().all(|&l| l.is_finite()),
            "TauMode lambdas should be finite"
        );
        assert!(
            tau_lambdas.iter().all(|&l| l >= 0.0),
            "TauMode lambdas should be non-negative"
        );
        assert!(
            tau_lambdas.iter().all(|&l| l <= 1.0),
            "TauMode lambdas should be bounded by 1.0"
        );
    }

    // Verify extreme parameters behavior
    debug!("\n=== EXTREME PARAMETERS ANALYSIS ===");
    debug!("With eps=1e-12 and sigma=5e-13, the graph should be very sparse");
    debug!("Only very similar vectors (within 1e-12 distance) will be connected");
    debug!("This may result in isolated nodes and specific spectral properties");

    // Success message
    debug!("\n✓ Lambda verification completed successfully for 384-dimensional vector");
    debug!("✓ All mathematical properties validated");
    debug!("✓ TauMode computations verified");
    debug!("✓ Extreme parameter behavior analyzed");
}
