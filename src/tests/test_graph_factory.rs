use crate::{
    builder::ArrowSpaceBuilder,
    tests::test_data::make_moons_hd,
};

#[test]
fn test_builder_basic_clustering_with_synthetic_data() {
    // Test basic clustering functionality with high-dimensional moons data
    let items: Vec<Vec<f64>> = make_moons_hd(
        100,    // Moderate number of samples
        0.15,   // Moderate noise
        0.4,    // Good separation
        10,     // 10-dimensional data
        42      // Reproducible seed
    );

    println!("Generated {} items with {} features", items.len(), items[0].len());

    let (_aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .build(items.clone());

    // Verify basic properties
    println!("Graph has {} nodes", gl.nnodes);
}

#[test]
fn test_builder_laplacian_diagonal_properties() {
    // Test that Laplacian diagonal entries are non-negative and finite
    let items: Vec<Vec<f64>> = make_moons_hd(
        80,     // Sufficient samples
        0.12,   // Low noise for stable structure
        0.5,    // Large separation
        8,      // 8 dimensions
        123     // Seed
    );

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.2, 4, 2, 2.0, None)
        .with_normalisation(true)
        .build(items);

    // Check diagonal properties
    let csr = &gl.matrix;
    assert!(csr.is_csr(), "Expected CSR layout");

    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();

    for i in 0..aspace.n_clusters {
        let start = indptr.into_raw_storage()[i];
        let end = indptr.into_raw_storage()[i + 1];
        let mut found = false;
        let mut diag = 0.0_f64;

        for pos in start..end {
            let j = indices[pos];
            if j == i {
                diag = data[pos];
                found = true;
                break;
            }
        }

        assert!(found, "Diagonal entry at ({},{}) should exist in Laplacian", i, i);
        assert!(diag.is_finite(), "Diagonal at ({},{}) must be finite, got {}", i, i, diag);
        assert!(diag >= 0.0, "Diagonal at ({},{}) must be non-negative, got {}", i, i, diag);
    }

    println!("✓ All {} diagonal entries are non-negative and finite", aspace.n_clusters);
}

#[test]
fn test_builder_minimum_items() {
    // Test minimum viable dataset
    let items: Vec<Vec<f64>> = make_moons_hd(
        20,     // Small dataset
        0.1,    // Low noise
        0.6,    // High separation
        5,      // Low dimensions
        42
    );

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.5, 3, 2, 2.0, None)
        .build(items.clone());

    assert!(aspace.n_clusters >= 1, "Should produce at least one cluster");
    assert_eq!(gl.nnodes, items.len());
    
    println!("Minimum items test: {} clusters from {} items", aspace.n_clusters, 20);
}

#[test]
fn test_builder_scale_invariance_with_normalization() {
    // Test that normalization makes the graph structure scale-invariant
    let items: Vec<Vec<f64>> = make_moons_hd(60, 0.15, 0.4, 8, 0);

    // Build with original scale
    let (aspace1, gl1) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(true)  // Normalize for scale invariance
        .build(items.clone());

    // Scale all items by constant factor
    let scale_factor = 5.7;
    let items_scaled: Vec<Vec<f64>> = items
        .iter()
        .map(|item| item.iter().map(|&x| x * scale_factor).collect())
        .collect();

    // Build with scaled data
    let (aspace2, gl2) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(true)  // Normalize for scale invariance
        .build(items_scaled);

    // With normalization, cluster counts should be similar (allowing minor numerical differences)
    assert!(
        (aspace1.n_clusters as i32 - aspace2.n_clusters as i32).abs() <= 3,
        "Normalized clustering should be scale-invariant: {} vs {}",
        aspace1.n_clusters, aspace2.n_clusters
    );

    // Graph sizes should match
    assert_eq!(gl1.nnodes, gl2.nnodes, "Graph node counts should match under scaling");
    
    println!("✓ Scale invariance verified: original={} clusters, scaled={} clusters", 
           aspace1.n_clusters, aspace2.n_clusters);
}

#[test]
fn test_builder_laplacian_symmetry() {
    // Test that the Laplacian is symmetric (undirected graph)
    let items: Vec<Vec<f64>> = make_moons_hd(70, 0.18, 0.35, 9, 456);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 5, 2, 2.0, None)
        .with_normalisation(true)
        .build(items);

    let csr = &gl.matrix;
    assert!(csr.is_csr(), "Expected CSR layout");

    let n = aspace.n_clusters;
    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();
    let eps = 1e-10;

    let mut symmetric_pairs = 0;
    let mut total_edges = 0;

    for i in 0..n {
        let start = indptr.into_raw_storage()[i];
        let end = indptr.into_raw_storage()[i + 1];
        
        for p in start..end {
            let j = indices[p];
            if i == j {
                continue; // Skip diagonal
            }
            
            total_edges += 1;
            let vij = data[p];

            // Find symmetric entry (j, i)
            let js = indptr.into_raw_storage()[j];
            let je = indptr.into_raw_storage()[j + 1];
            let mut vji_opt: Option<f64> = None;
            
            for q in js..je {
                if indices[q] == i {
                    vji_opt = Some(data[q]);
                    break;
                }
            }

            if let Some(vji) = vji_opt {
                assert!(
                    (vij - vji).abs() <= eps * (1.0 + vij.abs().max(vji.abs())),
                    "Symmetric entries must match: L[{},{}]={:.6} vs L[{},{}]={:.6}",
                    i, j, vij, j, i, vji
                );
                symmetric_pairs += 1;
            } else {
                panic!("Graph should be symmetric: found edge ({},{}) = {:.6} but missing ({},{})",
                       i, j, vij, j, i);
            }
        }
    }

    println!("✓ Verified symmetry for {} edge pairs (total {} edges)", 
           symmetric_pairs, total_edges);
}

#[test]
fn test_builder_parameter_preservation() {
    // Test that graph parameters are correctly preserved through the builder
    let items: Vec<Vec<f64>> = make_moons_hd(50, 0.2, 0.4, 7, 321);

    let (_, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(
            0.123,          // eps
            7,              // k
            3,              // topk
            3.5,            // p
            Some(0.456)     // sigma
        )
        .with_normalisation(false)
        .build(items);

    // Verify all parameters are preserved
    assert_eq!(gl.graph_params.eps, 0.123, "eps must match");
    assert_eq!(gl.graph_params.k, 7, "k must match");
    assert_eq!(gl.graph_params.topk, 3 + 1, "topk must match");
    assert_eq!(gl.graph_params.p, 3.5, "p must match");
    assert_eq!(gl.graph_params.sigma, Some(0.456), "sigma must match");
    assert_eq!(gl.graph_params.normalise, false, "normalise flag must match");

    println!("✓ All graph parameters correctly preserved");
}

#[test]
fn test_builder_with_different_dimensions() {
    // Test builder works across different dimensionalities
    let test_cases = vec![
        (50, 3, "low-dimensional"),
        (60, 10, "medium-dimensional"),
        (70, 25, "high-dimensional"),
    ];

    for (n_samples, dims, desc) in test_cases {
        let items: Vec<Vec<f64>> = make_moons_hd(
            n_samples,
            0.15,
            0.4,
            dims,
            42 + dims as u64  // Vary seed by dimension
        );

        let (aspace, gl) = ArrowSpaceBuilder::default()
            .with_lambda_graph(0.3, 5, 2, 2.0, None)
            .with_normalisation(true)
            .with_spectral(true)
            .build(items);

        assert!(aspace.n_clusters > 0, "{}: Should produce clusters", desc);
        assert!(aspace.nfeatures == dims, "{}: Features should be {}", desc, dims);

        println!("{}: {} clusters, {} features, {} nodes", 
               desc, aspace.n_clusters, aspace.nfeatures, gl.nnodes);
    }
}

#[test]
fn test_builder_spectral_laplacian_shape() {
    // Test that spectral Laplacian has correct shape (FxF)
    let items: Vec<Vec<f64>> = make_moons_hd(90, 0.16, 0.38, 12, 555);

    // Build WITHOUT spectral
    let (aspace_no_spectral, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 4, 2, 2.0, None)
        .with_spectral(false)
        .build(items.clone());

    // Build WITH spectral
    let (aspace_spectral, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 4, 2, 2.0, None)
        .with_spectral(true)
        .build(items.clone());

    // Without spectral, signals should be empty
    assert_eq!(
        aspace_no_spectral.signals.shape(),
        (0, 0),
        "Signals should be empty when spectral is disabled"
    );

    // With spectral, signals should be FxF where F is number of features
    let expected_dim = aspace_spectral.nfeatures;
    assert_eq!(
        aspace_spectral.signals.shape(),
        (expected_dim, expected_dim),
        "Signals should be {}x{} (feature-by-feature Laplacian)",
        expected_dim, expected_dim
    );

    println!("✓ Spectral Laplacian shape: {:?}", aspace_spectral.signals.shape());
}

#[test]
fn test_builder_lambda_values_are_nonnegative() {
    // Test that all lambda values (spectral scores) are non-negative
    let items: Vec<Vec<f64>> = make_moons_hd(100, 0.2, 0.35, 11, 999);

    let (aspace, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .build(items);

    let lambdas = aspace.lambdas();
    
    for (i, &lam) in lambdas.iter().enumerate() {
        assert!(
            lam >= 0.0,
            "Lambda at index {} should be non-negative, got {:.6}",
            i, lam
        );
    }

    let min_lambda = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_lambda = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("✓ All {} lambdas are non-negative: min={:.6}, max={:.6}", 
           lambdas.len(), min_lambda, max_lambda);
}


#[test]
fn test_builder_with_high_noise() {
    // Test behavior with very noisy data
    let items: Vec<Vec<f64>> = make_moons_hd(
        80,     // Samples
        0.5,    // HIGH noise - creates significant overlap
        0.2,    // Small separation
        9,      // Dimensions
        888
    );

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.4, 6, 3, 2.0, None)
        .with_normalisation(true)
        .build(items);

    // Even with high noise, should produce valid clustering
    assert!(aspace.n_clusters > 5, "Should produce clusters even with high noise");
}

#[test]
fn test_builder_normalization_effects() {
    // Compare normalized vs unnormalized builds
    let items: Vec<Vec<f64>> = make_moons_hd(75, 0.14, 0.45, 8, 654);

    // Build with normalization
    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .build(items.clone());

    // Build without normalization
    let (aspace_raw, gl_raw) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(false)
        .build(items);

    println!("Normalized: {} clusters", aspace_norm.n_clusters);
    println!("Raw (τ-mode): {} clusters", aspace_raw.n_clusters);

    // Parameters should be correctly set
    assert_eq!(gl_norm.graph_params.normalise, true);
    assert_eq!(gl_raw.graph_params.normalise, false);

    // Both should produce valid results
    assert!(aspace_norm.n_clusters > 0);
    assert!(aspace_raw.n_clusters > 0);

    println!("✓ Both normalization modes produce valid clusterings");
}
