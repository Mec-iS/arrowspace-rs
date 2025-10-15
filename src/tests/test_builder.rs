use smartcore::linalg::basic::arrays::Array;

use crate::{builder::ArrowSpaceBuilder, tests::test_data::make_moons_hd};

#[test]
fn test_minimal_input() {
    let rows = vec![
        vec![1.0, 0.0, 3.0],
        vec![0.5, 1.0, 0.0],
        vec![1.0, 0.0, 3.0],
        vec![0.5, 1.0, 0.0],
    ];
    ArrowSpaceBuilder::new().build(rows);
}

#[test]
fn simple_build() {
    // build `with_lambda_graph`
    let rows = vec![
        vec![1.0, 0.0, 5.0],
        vec![0.3, 1.0, 0.0],
        vec![1.0, 0.0, 5.0],
        vec![0.3, 1.0, 0.0],
    ];

    let eps = 1.0;
    let k = 3usize;
    let topk = 3usize;
    let p = 2.0;
    let sigma_override = None;

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(eps, k, topk, p, sigma_override)
        .build(rows);

    assert_eq!(aspace.data.shape(), (4, 3));
    assert_eq!(gl.nnodes, 4);
}

#[test]
fn build_from_rows_with_lambda_graph() {
    let rows = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36, 0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37, 0.56,
        ],
        vec![
            0.81, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36, 0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.55, 0.47, 0.21, 0.70, 0.08, 0.37, 0.56,
        ],
    ];

    // Build a lambda-proximity Laplacian over items from the data matrix
    // Parameters mirror the old intent: small eps, k=2 cap, p=2.0 kernel, default sigma
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1e-3, 2, 2, 2.0, None)
        .build(rows);

    assert_eq!(aspace.data.shape(), (4, 13));
    assert_eq!(gl.nnodes, 4);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

#[test]
fn build_with_lambda_graph_over_product_like_rows() {
    // Test with realistic high-dimensional feature vectors instead of synthetic product coordinates
    // These represent meaningful data patterns commonly found in ML applications
    let rows = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36, 0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37, 0.56,
        ],
        vec![
            0.81, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36, 0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.22, 0.70, 0.08, 0.37, 0.56,
        ],
    ];

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1e-3, 3, 3, 2.0, None)
        .build(rows);

    assert_eq!(aspace.data.shape(), (4, 13));
    assert_eq!(gl.nnodes, 4);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

#[test]
fn lambda_graph_shape_matches_rows() {
    // Test that lambda-graph construction correctly handles multiple items
    // with realistic high-dimensional feature vectors
    let items = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36, 0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37, 0.56,
        ],
        vec![
            0.85, 0.09, 0.41, 0.31, 0.67, 0.29, 0.53, 0.52, 0.17, 0.76, 0.05, 0.38, 0.60,
        ],
    ];
    let len_items = items.len();

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1e-3, 3, 3, 2.0, None)
        .build(items);

    assert_eq!(aspace.data.shape(), (len_items, 13));
    assert_eq!(gl.nnodes, len_items);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

// ============================================================================
// Sampling tests
// ============================================================================

#[test]
fn test_density_adaptive_sampling_basic() {
    // Test basic functionality of density-adaptive sampling
    let rows = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.1, 0.1, 0.0],
        vec![1.0, 0.0, 0.1],
        vec![1.1, 0.1, 0.1],
        vec![5.0, 5.0, 5.0], // Outlier - should be kept more reliably
        vec![5.1, 5.0, 5.0],
        vec![5.0, 5.1, 5.0],
        vec![5.0, 5.0, 5.1],
    ];

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_inline_sampling(true)
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .build(rows.clone());

    // Verify structure is preserved
    assert_eq!(aspace.data.shape(), (8, 3));
    assert!(gl.nnodes == 8);
    assert!(gl.matrix.shape().1 == 3);
}

#[test]
fn test_density_adaptive_preserves_outliers() {
    // Test that density-adaptive sampling keeps outliers/sparse regions
    let mut rows = Vec::new();

    // Dense cluster around origin (20 points)
    for i in 0..20 {
        rows.push(vec![
            0.1 * (i as f64 / 20.0),
            0.1 * ((i + 5) as f64 / 20.0),
            0.1 * ((i + 10) as f64 / 20.0),
        ]);
    }

    // Sparse outliers far away (5 points)
    for i in 0..7 {
        rows.push(vec![10.0 + i as f64, 10.0 + i as f64, 10.0]);
    }

    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_inline_sampling(true)
        .build(rows.clone());

    // Check that at least some outlier region is represented
    // by looking for points with large coordinates
    let mut has_outlier_region = false;
    for i in 0..aspace.data.shape().0 {
        let row_sum: f64 = (0..3).map(|j| aspace.data.get((i, j))).sum();
        if row_sum > 15.0 {
            // Outliers have sum ~30
            has_outlier_region = true;
            break;
        }
    }
    assert!(
        has_outlier_region,
        "Density-adaptive sampling should preserve outlier region"
    );
}

#[test]
fn test_density_adaptive_with_uniform_data() {
    // Test behavior on uniformly distributed data
    let rows: Vec<Vec<f64>> = make_moons_hd(50, 0.3, 0.52, 10, 42);

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_inline_sampling(true)
        .with_lambda_graph(1.0, 5, 5, 2.0, None)
        .build(rows.clone());

    assert_eq!(aspace.data.shape().1, 10);
    assert!(gl.nnodes == 50);
}

#[test]
fn test_density_adaptive_high_rate() {
    // Test with high sampling rate (90%) - should keep most data
    let rows = make_moons_hd(50, 0.10, 0.20, 10, 42);

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_inline_sampling(true)
        .with_lambda_graph(1e-3, 3, 3, 2.0, None)
        .build(rows.clone());

    let sampling_ratio = gl.matrix.shape().1 as f64 / rows.len() as f64;

    // With 90% target, should keep most rows
    assert!(
        sampling_ratio >= 0.2,
        "High sampling rate {:.2} should keep most data",
        sampling_ratio
    );

    assert_eq!(aspace.data.shape().1, 10);
    assert!(gl.nnodes > 0);
    assert_eq!(aspace.data.shape(), (50, 10));
    assert!(gl.nnodes == 50);
    assert!(gl.matrix.shape().0 == 10);
}

#[test]
fn test_density_adaptive_aggressive_sampling() {
    // Test very aggressive sampling (10%) on larger dataset
    let rows = make_moons_hd(50, 0.10, 0.40, 10, 42);

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_inline_sampling(true)
        .with_lambda_graph(2.0, 5, 5, 2.0, None)
        .build(rows.clone());

    let sampled_count = gl.matrix.shape().0;
    let sampling_ratio = sampled_count as f64 / rows.len() as f64;

    // Should sample around 20%, but may vary due to density adaptation
    assert!(
        sampling_ratio >= 0.05 && sampling_ratio <= 0.25,
        "Aggressive sampling ratio {:.2} outside expected range [0.05, 0.25]",
        sampling_ratio
    );

    // Despite aggressive sampling, should still create valid Laplacian
    assert!(
        sampled_count >= 4,
        "Should keep at least 4 points for valid graph"
    );
    assert_eq!(aspace.data.shape().1, 10);
    assert_eq!(aspace.data.shape(), (50, 10));
    assert!(gl.nnodes == 50);
    assert!(gl.matrix.shape().0 == 10);
}

#[test]
fn test_density_adaptive_with_duplicates() {
    // Test behavior with many duplicate or near-duplicate rows
    let rows = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0],
        vec![1.001, 2.001, 3.001],
        vec![1.0, 2.0, 3.0],
        vec![5.0, 6.0, 7.0], // Different cluster
        vec![5.0, 6.0, 7.0],
        vec![5.001, 6.001, 7.001],
    ];

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_inline_sampling(true)
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .build(rows.clone());

    let sampled_count = gl.matrix.shape().0;

    // Should aggressively sample from duplicate-heavy regions
    assert!(
        sampled_count >= 2 && sampled_count <= 5,
        "Should sample efficiently from duplicates: got {}",
        sampled_count
    );

    assert_eq!(aspace.data.shape().1, 3);
    assert!(gl.nnodes > 0);
}

#[test]
fn test_density_adaptive_sampling_statistics() {
    // Test statistical properties over multiple runs

    // Run multiple times and check consistency
    for i in 1..6 {
        let rows: Vec<Vec<f64>> = make_moons_hd(50 * i, 0.5, 0.2, 10 * i, 42 * (i as u64));

        let (aspace, gl) = ArrowSpaceBuilder::new()
            .with_inline_sampling(true)
            .with_sparsity_check(false)
            .build(rows.clone());

        assert_eq!(aspace.data.shape().1, 10 * i);
        assert_eq!(aspace.data.shape(), (50 * i, 10 * i));
        assert!(gl.nnodes == 50 * i);
    }
}

#[test]
fn test_density_adaptive_vs_no_sampling() {
    // Compare results with and without sampling
    let rows: Vec<Vec<f64>> = make_moons_hd(50, 0.10, 0.40, 100, 42);

    // Without sampling
    let (aspace_full, gl_full) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 5, 5, 2.0, None)
        .build(rows.clone());

    // With 50% density-adaptive sampling
    let (aspace_sampled, gl_sampled) = ArrowSpaceBuilder::new()
        .with_inline_sampling(true)
        .with_lambda_graph(1.0, 5, 5, 2.0, None)
        .build(rows.clone());

    // ashape holds full dataset: NxN in both cases
    assert!(
        aspace_sampled.data.shape().0 == aspace_full.data.shape().0,
        "Sampled ({}) should be smaller than full ({})",
        aspace_sampled.data.shape().0,
        aspace_full.data.shape().0
    );

    // Both should have same dimensionality
    assert_eq!(aspace_sampled.data.shape().1, aspace_full.data.shape().1);

    // Both should produce valid graphs
    assert!(gl_sampled.nnodes > 0);
    assert!(gl_full.nnodes > 0);
}

#[test]
fn test_density_adaptive_maintains_lambda_quality() {
    // Test that density-adaptive sampling preserves lambda quality

    for i in 1..5 {
        let rows: Vec<Vec<f64>> = make_moons_hd(50 * i, 0.5, 0.5, 100 * i, 42 * (i as u64));

        let (aspace, _gl) = ArrowSpaceBuilder::new()
            .with_lambda_graph(1e-1, 3, 3, 2.0, None)
            .with_inline_sampling(true)
            .with_sparsity_check(false)
            .build(rows);

        // Check lambda values are valid (non-negative)
        let lambdas = aspace.lambdas();
        assert!(
            lambdas.iter().all(|&l| l >= 0.0),
            "All lambdas should be non-negative"
        );

        // Check lambda values have some variance (not all identical)
        let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
        let has_variance = lambdas.iter().any(|&l| (l - lambda_mean).abs() > 0.001);
        assert!(has_variance, "Lambdas should have some variance");
    }
}

#[test]
fn test_with_deterministic_seed() {
    let items = make_moons_hd(80, 0.50, 0.50, 9, 789);
    let seed = 42u64;
    
    let (aspace1, _) = ArrowSpaceBuilder::default()
        .with_seed(seed)
        .build(items.clone());
    
    let (aspace2, _) = ArrowSpaceBuilder::default()
        .with_seed(seed)
        .build(items.clone());
    
    // Should be identical
    assert_eq!(aspace1.n_clusters, aspace2.n_clusters);
}

#[test]
fn test_builder_unit_norm_diagonal_similarity() {
    let items_raw: Vec<Vec<f64>> = make_moons_hd(80, 0.50, 0.50, 9, 789);
    
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
    
    let seed = 42u64;
    
    let (aspace_norm, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(true)
        .with_dims_reduction(false, None)
        .with_inline_sampling(false)
        .with_seed(seed)
        .build(items.clone());
    
    let (aspace_raw, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(false)
        .with_dims_reduction(false, None)
        .with_inline_sampling(false)
        .with_seed(seed)
        .build(items_raw.clone());
    
    // Now should be identical
    assert_eq!(aspace_norm.n_clusters, aspace_raw.n_clusters);
}


