//! # Query Tests: Lambda-aware search with random projection
//!
//! Tests the full query pipeline including:
//! 1. Building clustered index with optional random projection
//! 2. Projecting query vectors using the same transformation
//! 3. Computing query lambda values on cluster representatives
//! 4. Lambda-aware similarity search in clustered space
//! 5. Hybrid search combining semantic and spectral scoring

use crate::{builder::ArrowSpaceBuilder, core::ArrowItem, tests::test_data::make_moons_hd};

/// Helper: return test data (training + query split from same distribution)
fn create_test_data(
    n_train: usize,
    n_query: usize,
    dims: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let total = n_train + n_query;
    let all_data = make_moons_hd(total, 0.15, 0.4, dims, seed);

    let train = all_data[0..n_train].to_vec();
    let queries = all_data[n_train..].to_vec();

    (train, queries)
}

#[test]
fn test_query_without_projection() {
    // Build clustered index without dimensionality reduction
    let (data, queries) = create_test_data(100, 5, 50, 42);

    println!(
        "Building index with {} items, {} dims",
        data.len(),
        data[0].len()
    );

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, Some(0.1))
        .with_normalisation(true)
        .with_dims_reduction(false, None) // No projection
        .with_sparsity_check(false)
        .build(data);

    println!("Built index with {} clusters", aspace.n_clusters);

    // Prepare query (no projection needed)
    let query_lambda = aspace.prepare_query_item(&queries[0], &gl);
    assert!(query_lambda.is_finite(), "Query lambda should be finite");
    assert!(query_lambda >= 0.0, "Query lambda should be non-negative");

    let query_item = ArrowItem::new(queries[0].clone(), query_lambda);

    // Search in clustered space
    let results = aspace.search_lambda_aware(&query_item, 5, 0.7);
    assert_eq!(results.len(), 5, "Should return exactly 5 results");

    // Verify descending order by score
    for i in 0..results.len() - 1 {
        assert!(
            results[i].1 >= results[i + 1].1,
            "Results should be sorted by descending score"
        );
    }

    println!(
        "✓ Query without projection: found {} results",
        results.len()
    );
}

#[test]
fn test_query_with_projection_enabled() {
    // Test query projection in high-dimensional clustered space
    let (data, queries) = create_test_data(150, 5, 200, 123);

    println!("Testing projection with {}D data", data[0].len());

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 6, 2, 2.0, None)
        .with_normalisation(true)
        .with_dims_reduction(true, Some(0.3)) // Enable JL projection
        .with_sparsity_check(false)
        .build(data);

    // Verify projection was applied
    assert!(
        aspace.projection_matrix.is_some(),
        "Projection matrix should exist"
    );
    assert!(
        aspace.reduced_dim.is_some(),
        "Reduced dimension should be set"
    );

    let reduced_dim = aspace.reduced_dim.unwrap();
    println!(
        "Projection: {} → {} dimensions ({:.1}x compression)",
        aspace.nfeatures,
        reduced_dim,
        aspace.nfeatures as f64 / reduced_dim as f64
    );

    // Query must be projected to match index space
    let query_original = queries[1].clone();
    let query_projected = aspace.project_query(&query_original);

    assert_eq!(
        query_projected.len(),
        reduced_dim,
        "Projected query should match reduced dimension"
    );

    // Compute lambda on projected query
    let query_lambda = aspace.prepare_query_item(&query_original, &gl);
    assert!(query_lambda.is_finite() && query_lambda >= 0.0);

    let query_item = ArrowItem::new(query_original, query_lambda);

    // Search should work in projected clustered space
    let results = aspace.search_lambda_aware(&query_item, 10, 0.7);
    assert_eq!(results.len(), 10);

    // Verify all results are valid cluster indices
    for (idx, score) in &results {
        assert!(*idx < aspace.nitems, "Index should be valid cluster");
        assert!(score.is_finite(), "Score should be finite");
        assert!(
            *score >= -1.0 && *score <= 1.0,
            "Score should be in [-1, 1]"
        );
    }

    println!(
        "✓ Query with projection: {} → {} dims, found {} results",
        aspace.nfeatures,
        reduced_dim,
        results.len()
    );
}

#[test]
fn test_prepare_query_item_consistency() {
    // Test that prepare_query_item produces stable lambda values
    let (data, queries) = create_test_data(80, 5, 64, 456);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 4, 2, 2.0, None)
        .with_normalisation(true)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[2].clone();

    // Compute lambda multiple times - should be deterministic
    let lambda1 = aspace.prepare_query_item(&query, &gl);
    let lambda2 = aspace.prepare_query_item(&query, &gl);
    let lambda3 = aspace.prepare_query_item(&query, &gl);

    assert_eq!(
        lambda1, lambda2,
        "Lambda computation should be deterministic"
    );
    assert_eq!(
        lambda2, lambda3,
        "Lambda computation should be deterministic"
    );
    assert!(lambda1 >= 0.0, "Lambda should be non-negative");

    println!("✓ Consistent lambda: {:.6}", lambda1);
}

#[test]
fn test_search_lambda_aware_alpha_effect() {
    // Test that alpha parameter controls semantic vs spectral balance
    let (data, queries) = create_test_data(100, 5, 48, 789);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true) // Enable spectral for lambda computation
        .with_sparsity_check(false)
        .build(data);

    let query = queries[2].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    // High alpha (0.9): should favor semantic similarity
    let results_high_alpha = aspace.search_lambda_aware(&query_item, 5, 0.9);

    // Low alpha (0.1): should favor lambda (spectral) similarity
    let results_low_alpha = aspace.search_lambda_aware(&query_item, 5, 0.1);

    // Both should produce valid results
    assert_eq!(results_high_alpha.len(), 5);
    assert_eq!(results_low_alpha.len(), 5);

    println!(
        "High alpha top result: idx={}, score={:.4}",
        results_high_alpha[0].0, results_high_alpha[0].1
    );
    println!(
        "Low alpha top result: idx={}, score={:.4}",
        results_low_alpha[0].0, results_low_alpha[0].1
    );

    // Verify top result has high semantic similarity with high alpha
    let top_idx_high = results_high_alpha[0].0;
    let top_item_high = aspace.get_item(top_idx_high);
    let semantic_sim = query_item.cosine_similarity(&top_item_high.item);

    assert!(
        semantic_sim > 0.7,
        "High alpha should favor semantic match: {:.4}",
        semantic_sim
    );

    println!("✓ Alpha effect verified: semantic_sim={:.4}", semantic_sim);
}

#[test]
fn test_search_lambda_aware_hybrid() {
    // Compare regular lambda-aware search vs hybrid search
    let (data, queries) = create_test_data(90, 5, 56, 321);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[1].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    // Regular lambda-aware search
    let results_regular = aspace.search_lambda_aware(&query_item, 10, 0.7);

    // Hybrid search (combines multiple scoring strategies)
    let results_hybrid = aspace.search_lambda_aware_hybrid(&query_item, 10, 0.7);

    assert_eq!(results_regular.len(), 10);
    assert_eq!(results_hybrid.len(), 10);

    // Both should return valid cluster indices
    for (idx, score) in results_regular.iter().chain(results_hybrid.iter()) {
        assert!(*idx < aspace.nitems, "Index should be valid cluster");
        assert!(score.is_finite(), "Score should be finite");
    }

    println!(
        "✓ Hybrid search: regular={} results, hybrid={} results",
        results_regular.len(),
        results_hybrid.len()
    );
}

#[test]
#[should_panic]
fn test_query_dimension_mismatch_panics() {
    // Query with wrong dimension should panic
    let (data, queries) = create_test_data(50, 5, 64, 999);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .build(data);

    // Create query with wrong dimension (half the expected size)
    let wrong_query = queries[0][0..32].to_vec();

    let _ = aspace.prepare_query_item(&wrong_query, &gl);
}

#[test]
#[should_panic]
fn test_query_with_nan_values() {
    // Query with NaN should panic
    let (data, queries) = create_test_data(50, 5, 48, 888);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .build(data);

    let mut bad_query = queries[0].clone();
    bad_query[3] = f64::NAN;

    let _ = aspace.prepare_query_item(&bad_query, &gl);
}

#[test]
fn test_range_search_with_query_lambda() {
    // Test range-based search in clustered space
    let (data, queries) = create_test_data(80, 5, 40, 555);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[1].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    println!("query lambda {:?}", query_lambda);

    // Find all clusters within radius 0.5
    let results = aspace.range_search(&query_item, &gl, 0.1);

    println!("{:?}", aspace.lambdas);

    // Should find at least some clusters
    assert!(!results.is_empty(), "Range search should find some results");

    // Verify all results are within radius
    for (idx, dist) in &results {
        assert!(*idx < aspace.nitems, "Index should be valid cluster");
        assert!(
            *dist <= 0.5,
            "Distance should be within radius: {:.4}",
            dist
        );
        assert!(dist.is_finite(), "Distance should be finite");
    }

    println!(
        "✓ Range search found {} clusters within radius 0.5",
        results.len()
    );
}

#[test]
fn test_lambda_values_reasonable_range() {
    // Test that query lambdas are in reasonable range
    let (data, queries) = create_test_data(100, 5, 72, 777);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .with_sparsity_check(false)
        .build(data);

    // Test all queries
    for (i, query) in queries.iter().enumerate() {
        let lambda = aspace.prepare_query_item(query, &gl);

        assert!(lambda >= 0.0, "Lambda should be non-negative: query {}", i);
        assert!(lambda.is_finite(), "Lambda should be finite: query {}", i);
        assert!(
            lambda < 100.0,
            "Lambda unusually large for query {}: {:.4}",
            i,
            lambda
        );

        println!("Query {} lambda: {:.6}", i, lambda);
    }

    println!("✓ All query lambdas in reasonable range");
}

#[test]
fn test_search_returns_top_k_exactly() {
    // Test that search returns exactly k results (or all if k > num_clusters)
    let (data, queries) = create_test_data(120, 5, 60, 654);

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_sparsity_check(false)
        .build(data);

    let query = queries[2].clone();
    let query_lambda = aspace.prepare_query_item(&query, &gl);
    let query_item = ArrowItem::new(query, query_lambda);

    let num_clusters = aspace.n_clusters;
    println!("Testing k-NN with {} total clusters", num_clusters);

    // Test various k values
    for k in [1, 3, 5, 10] {
        let results = aspace.search_lambda_aware(&query_item, k, 0.1);
        assert_eq!(
            results.len(),
            k,
            "Should return exactly {} results for k={}",
            results.len(),
            k
        );
    }

    println!("✓ k-NN returns correct number of results");
}

#[test]
fn test_projection_preserves_relative_distances() {
    // Test Johnson-Lindenstrauss projection preserves relative distances
    let (data, _) = create_test_data(80, 0, 200, 42);

    println!("Testing JL projection with 200D data");

    let (aspace, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, Some(0.1))
        .with_normalisation(true)
        .with_dims_reduction(true, Some(0.3)) // 30% of original dimension
        .with_sparsity_check(false)
        .build(data);

    // Verify projection was applied
    assert!(
        aspace.projection_matrix.is_some(),
        "Projection should be enabled"
    );
    let reduced_dim = aspace.reduced_dim.unwrap();

    println!(
        "Projection: {} → {} dimensions ({:.1}x compression)",
        aspace.nfeatures,
        reduced_dim,
        aspace.nfeatures as f64 / reduced_dim as f64
    );

    // Create three queries with known relationships
    let query1_orig = vec![0.5; 200];
    let query2_orig = vec![0.51; 200]; // Very close to q1
    let query3_orig = vec![5.0; 200]; // Far from q1

    // Project all three queries
    let query1_proj = aspace.project_query(&query1_orig);
    let query2_proj = aspace.project_query(&query2_orig);
    let query3_proj = aspace.project_query(&query3_orig);

    // Verify projected dimensions
    assert_eq!(query1_proj.len(), reduced_dim);
    assert_eq!(query2_proj.len(), reduced_dim);
    assert_eq!(query3_proj.len(), reduced_dim);

    // Compute L2 distances in original space
    let dist_12_orig: f64 = query1_orig
        .iter()
        .zip(&query2_orig)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let dist_13_orig: f64 = query1_orig
        .iter()
        .zip(&query3_orig)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    // Compute L2 distances in projected space
    let dist_12_proj: f64 = query1_proj
        .iter()
        .zip(&query2_proj)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let dist_13_proj: f64 = query1_proj
        .iter()
        .zip(&query3_proj)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    println!(
        "Original space: dist(q1,q2)={:.4}, dist(q1,q3)={:.4}",
        dist_12_orig, dist_13_orig
    );
    println!(
        "Projected space: dist(q1,q2)={:.4}, dist(q1,q3)={:.4}",
        dist_12_proj, dist_13_proj
    );

    // CRITICAL: Verify relative ordering is preserved (JL property)
    assert!(
        dist_12_orig < dist_13_orig,
        "In original space, q1 should be closer to q2 than q3"
    );
    assert!(
        dist_12_proj < dist_13_proj,
        "In projected space, q1 should STILL be closer to q2 than q3 (ordering preserved)"
    );

    // Verify approximate distance preservation with 20% tolerance
    let epsilon = 0.2;
    let ratio_12 = dist_12_proj / dist_12_orig;
    let ratio_13 = dist_13_proj / dist_13_orig;

    println!(
        "Distance preservation ratios: q1-q2={:.3}, q1-q3={:.3}",
        ratio_12, ratio_13
    );

    assert!(
        ratio_12 > 1.0 - epsilon && ratio_12 < 1.0 + epsilon,
        "Distance q1-q2 not preserved: ratio {:.3} outside tolerance",
        ratio_12
    );
    assert!(
        ratio_13 > 1.0 - epsilon && ratio_13 < 1.0 + epsilon,
        "Distance q1-q3 not preserved: ratio {:.3} outside tolerance",
        ratio_13
    );

    // Verify lambda computation works on projected queries
    let lambda1 = aspace.prepare_query_item(&query1_orig, &gl);
    let lambda2 = aspace.prepare_query_item(&query2_orig, &gl);
    let lambda3 = aspace.prepare_query_item(&query3_orig, &gl);

    assert!(lambda1.is_finite() && lambda1 >= 0.0);
    assert!(lambda2.is_finite() && lambda2 >= 0.0);
    assert!(lambda3.is_finite() && lambda3 >= 0.0);

    println!(
        "Query lambdas: q1={:.6}, q2={:.6}, q3={:.6}",
        lambda1, lambda2, lambda3
    );
    println!("✓ Projection preserves relative distances and enables lambda computation");
}

#[test]
fn test_project_query_no_projection() {
    // When projection is disabled, queries should pass through unchanged
    let (data, queries) = create_test_data(50, 3, 48, 321);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_dims_reduction(false, None) // No projection
        .build(data);

    let query = queries[0].clone();
    let projected = aspace.project_query(&query);

    // Should return query unchanged
    assert_eq!(projected.len(), query.len());
    assert_eq!(projected, query);

    println!("✓ Query passes through unchanged when projection disabled");
}

#[test]
fn test_project_query_consistency() {
    // Projection should be deterministic
    let (data, queries) = create_test_data(60, 3, 100, 654);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_dims_reduction(true, Some(0.4))
        .with_sparsity_check(false)
        .build(data);

    let query = queries[1].clone();

    // Project same query multiple times
    let projected1 = aspace.project_query(&query);
    let projected2 = aspace.project_query(&query);
    let projected3 = aspace.project_query(&query);

    // Should be deterministic
    assert_eq!(projected1, projected2, "Projection should be deterministic");
    assert_eq!(projected2, projected3, "Projection should be deterministic");

    println!("✓ Projection is deterministic");
}

#[test]
fn test_project_query_linearity() {
    // Projection should be linear: project(α*x) = α*project(x)
    let (data, queries) = create_test_data(50, 3, 80, 987);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_dims_reduction(true, Some(0.35))
        .build(data);

    let query = queries[1].clone();
    let scaled_query: Vec<f64> = query.iter().map(|x| x * 2.0).collect();

    let projected = aspace.project_query(&query);
    let projected_scaled = aspace.project_query(&scaled_query);

    // Projection should be linear: project(2*x) = 2*project(x)
    for i in 0..projected.len() {
        let expected = projected[i] * 2.0;
        let actual = projected_scaled[i];
        assert!(
            (expected - actual).abs() < 1e-9,
            "Linearity violation at index {}: expected {:.6}, got {:.6}",
            i,
            expected,
            actual
        );
    }

    println!("✓ Projection is linear");
}

#[test]
fn test_project_query_zero_vector() {
    // Projection of zero should be zero
    let (data, _) = create_test_data(50, 0, 96, 111);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.2, 4, 2, 2.0, None)
        .with_dims_reduction(true, Some(0.3))
        .with_sparsity_check(false)
        .build(data);

    let query_zero = vec![0.0; 96];
    let projected = aspace.project_query(&query_zero);

    assert_eq!(projected.len(), aspace.reduced_dim.unwrap());

    for &val in &projected {
        assert!(val.abs() < 1e-8, "Expected near-zero, got {:.6}", val);
    }

    println!("✓ Zero vector projects to zero");
}

#[test]
fn test_project_query_preserves_scale_approximately() {
    // Johnson-Lindenstrauss preserves norms approximately
    let (data, queries) = create_test_data(60, 3, 128, 222);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 5, 2, 2.0, None)
        .with_dims_reduction(true, Some(0.25))
        .with_sparsity_check(false)
        .build(data);

    let query = queries[1].clone();
    let projected = aspace.project_query(&query);

    // Compute norms
    let orig_norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();
    let proj_norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();

    // JL preserves norms approximately - ratio should be within reasonable bounds
    let ratio = proj_norm / orig_norm;

    println!(
        "Norm preservation: original={:.4}, projected={:.4}, ratio={:.3}",
        orig_norm, proj_norm, ratio
    );

    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Norm ratio {:.3} out of expected range [0.5, 2.0]",
        ratio
    );

    println!("✓ Projection preserves scale approximately");
}

#[test]
fn test_project_query_different_queries_differ() {
    // Different queries should produce different projections
    let (data, queries) = create_test_data(300, 4, 68, 333);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_dims_reduction(true, Some(0.8))
        .build(data);

    let proj1 = aspace.project_query(&queries[0]);
    let proj2 = aspace.project_query(&queries[1]);
    let proj3 = aspace.project_query(&queries[2]);

    // All should have correct dimension
    let reduced_dim = aspace.reduced_dim.unwrap();
    assert_eq!(proj1.len(), reduced_dim);
    assert_eq!(proj2.len(), reduced_dim);
    assert_eq!(proj3.len(), reduced_dim);

    // Different queries should produce different projections
    assert_ne!(proj1, proj2);
    assert_ne!(proj2, proj3);
    assert_ne!(proj1, proj3);

    println!("✓ Different queries produce different projections");
}

#[test]
fn test_project_query_preserves_dot_product_sign() {
    // Projection should preserve angle relationships
    let (data, queries) = create_test_data(50, 2, 72, 444);

    let (aspace, _gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_dims_reduction(true, Some(0.35))
        .build(data);

    let query_pos = queries[0].clone();
    let query_neg: Vec<f64> = queries[0].iter().map(|&x| -x).collect();

    let proj_pos = aspace.project_query(&query_pos);
    let proj_neg = aspace.project_query(&query_neg);

    // Their projections should have negative dot product (opposite directions)
    let dot: f64 = proj_pos.iter().zip(&proj_neg).map(|(a, b)| a * b).sum();

    assert!(
        dot < 0.0,
        "Projection should preserve opposite directions: dot={:.6}",
        dot
    );

    println!("✓ Projection preserves opposite directions (dot product sign)");
}
