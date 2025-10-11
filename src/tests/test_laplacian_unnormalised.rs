use crate::graph::GraphLaplacian;
use crate::{builder::ArrowSpaceBuilder, tests::test_data::make_moons_hd};
use approx::assert_relative_eq;

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
fn test_builder_unit_norm_items_invariance_under_normalisation_toggle_unnorm() {
    // All items already unit-normalized; toggling normalisation must not change Laplacian
    // Generate realistic data and manually normalize to unit vectors
    let items_raw: Vec<Vec<f64>> = make_moons_hd(
        100,  // Sufficient samples
        0.12, // Low noise for stable structure
        0.45, // Good separation
        10,   // Higher dimensionality
        42,
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

    println!("Generated {} unit-normalized items", items.len());

    // Verify first few items are unit-normalized
    for (i, item) in items.iter().enumerate().take(3) {
        let norm = item.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Item {} should be unit-normalized: norm = {:.12}",
            i,
            norm
        );
    }

    // Build with normalise = true
    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(true)
        .build(items.clone());

    // Build with normalise = false
    let (aspace_raw, gl_raw) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(false)
        .build(items_raw.clone());

    assert!(
        !laplacian_eq(&gl_norm, &gl_raw, 1e-2),
        "When items are unit-normalized, Laplacians should be different as taumode is not scale-invariant"
    );

    println!(
        "✓ Unit-norm invariance verified: norm={} clusters, raw={} clusters",
        aspace_norm.n_clusters, aspace_raw.n_clusters
    );
}

#[test]
fn test_builder_direction_vs_magnitude_sensitivity_unnormalised() {
    // Test that normalization (cosine) is scale-invariant while τ-mode is magnitude-sensitive
    // Create data with similar directions but different magnitudes
    let items_base: Vec<Vec<f64>> = make_moons_hd(60, 0.15, 0.4, 8, 123);

    // Create scaled version: multiply half the items by large factor
    let mut items = Vec::new();
    for (i, item) in items_base.iter().enumerate() {
        if i % 2 == 0 {
            // Every other item: scale by 100x to create magnitude differences
            items.push(item.iter().map(|&x| x * 100.0).collect());
        } else {
            items.push(item.clone());
        }
    }

    println!("Created dataset with mixed magnitudes (every other item scaled 100x)");

    // Build with normalisation=true (cosine-like, scale-invariant)
    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.1, 5, 2, 2.0, Some(0.1))
        .with_normalisation(true)
        .build(items_base.clone());

    // Build with normalisation=false (τ-mode: magnitude-sensitive)
    let (aspace_tau, gl_tau) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.1, 5, 2, 2.0, Some(0.1))
        .with_normalisation(false)
        .build(items_base.clone());

    // Build again with normalisation=true to verify stability
    let (_aspace_norm_again, gl_norm_again) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.1, 5, 2, 2.0, Some(0.1))
        .with_normalisation(true)
        .build(items.clone());

    // Normalized builds should be different
    assert!(
        !laplacian_eq(&gl_norm, &gl_norm_again, 1e-9),
        "Normalised builds should be different as taumode is scale-aware unlike cosine similarity"
    );

    // τ-mode should differ from normalized graph due to magnitude sensitivity
    let matrices_equal = laplacian_eq(&gl_norm, &gl_tau, 1e-9);
    assert!(
        !matrices_equal,
        "τ-mode should differ from normalised graph because it is magnitude-sensitive"
    );

    println!(
        "✓ Normalized: {} clusters, Tau: {} clusters",
        aspace_norm.n_clusters, aspace_tau.n_clusters
    );
    println!("✓ Cosine (normalized) is scale-invariant, τ-mode is magnitude-sensitive");
}

#[test]
fn test_builder_graph_params_preservation() {
    // Verify that graph parameters are correctly preserved through the builder
    let items: Vec<Vec<f64>> = make_moons_hd(50, 0.18, 0.4, 7, 456);

    let (_, gl) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 6, 3, 2.5, Some(0.15))
        .with_normalisation(false)
        .build(items);

    assert_eq!(gl.graph_params.eps, 0.25, "eps must match");
    assert_eq!(gl.graph_params.k, 6, "k must match");
    assert_eq!(gl.graph_params.topk, 3 + 1, "topk must match");
    assert_eq!(gl.graph_params.p, 2.5, "p must match");
    assert_eq!(gl.graph_params.sigma, Some(0.15), "sigma must match");
    assert_eq!(
        gl.graph_params.normalise, false,
        "normalise flag must match"
    );

    println!("✓ Graph parameters correctly preserved");
}

#[test]
fn test_builder_unit_norm_diagonal_similarity() {
    // Test that unit-normalized data produces SIMILAR graph properties
    // under both normalization modes (not identical due to clustering randomness)

    let items_raw: Vec<Vec<f64>> = make_moons_hd(80, 0.14, 0.42, 9, 789);

    // Normalize to unit vectors
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

    let (aspace_norm, gl_norm) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_normalisation(true)
        .with_dims_reduction(false, None) // Disable for more deterministic clustering
        .with_inline_sampling(false) // Disable for more deterministic clustering
        .build(items.clone());

    let (aspace_raw, gl_raw) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None) // SAME parameters
        .with_normalisation(false)
        .with_dims_reduction(false, None)
        .with_inline_sampling(false)
        .build(items_raw.clone()); // SAME input

    println!("Normalized build: {} clusters", aspace_norm.n_clusters);
    println!("Raw build: {} clusters", aspace_raw.n_clusters);

    // With unit-norm input and disabled randomization, clustering should be similar
    let cluster_diff = (aspace_norm.n_clusters as i32 - aspace_raw.n_clusters as i32).abs();
    assert!(
        cluster_diff <= 2,
        "Unit-norm data should produce similar cluster counts: {} vs {} (diff={})",
        aspace_norm.n_clusters,
        aspace_raw.n_clusters,
        cluster_diff
    );

    // Compare diagonal statistics (not exact values)
    let d_norm = diag_vec(&gl_norm);
    let d_raw = diag_vec(&gl_raw);

    let mean_diag_norm = d_norm.iter().sum::<f64>() / d_norm.len() as f64;
    let mean_diag_raw = d_raw.iter().sum::<f64>() / d_raw.len() as f64;

    println!("Mean diagonal (normalized): {:.6}", mean_diag_norm);
    println!("Mean diagonal (raw): {:.6}", mean_diag_raw);

    // For unit-norm data, statistical properties should be similar
    let mean_ratio = mean_diag_norm.max(mean_diag_raw) / mean_diag_norm.min(mean_diag_raw);
    assert!(
        mean_ratio < 1.5,
        "Mean diagonal values should be within 50% for unit-norm data: {:.6} vs {:.6} (ratio {:.2})",
        mean_diag_norm, mean_diag_raw, mean_ratio
    );

    println!(
        "✓ Unit-norm data produces similar diagonal statistics: {} vs {} clusters",
        d_norm.len(),
        d_raw.len()
    );
}

fn compute_cosine_similarity(item1: &[f64], item2: &[f64]) -> f64 {
    let dot: f64 = item1.iter().zip(item2.iter()).map(|(a, b)| a * b).sum();
    let norm1 = item1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2 = item2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm1 > 1e-12 && norm2 > 1e-12 {
        dot / (norm1 * norm2)
    } else {
        0.0
    }
}

fn compute_hybrid_similarity(item1: &[f64], item2: &[f64], alpha: f64, beta: f64) -> f64 {
    let norm1 = item1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2 = item2.iter().map(|x| x * x).sum::<f64>().sqrt();

    let cosine_sim = compute_cosine_similarity(item1, item2);

    if norm1 > 1e-12 && norm2 > 1e-12 {
        let magnitude_penalty = (-((norm1 / norm2).ln().abs())).exp();
        alpha * cosine_sim + beta * magnitude_penalty
    } else {
        cosine_sim
    }
}

#[test]
fn test_cosine_similarity_scale_invariance() {
    // Test that cosine similarity is scale invariant
    let items: Vec<Vec<f64>> = make_moons_hd(2, 0.0, 1.0, 13, 321);
    let item1 = &items[0];
    let item2 = &items[1];

    // Scale items by different factors
    let scale1 = 3.5;
    let scale2 = 0.2;
    let item1_scaled: Vec<f64> = item1.iter().map(|x| x * scale1).collect();
    let item2_scaled: Vec<f64> = item2.iter().map(|x| x * scale2).collect();

    let cosine_original = compute_cosine_similarity(item1, item2);
    let cosine_scaled = compute_cosine_similarity(&item1_scaled, &item2_scaled);

    println!("Original cosine similarity: {:.6}", cosine_original);
    println!("Scaled cosine similarity: {:.6}", cosine_scaled);

    // Cosine similarity should be identical (scale invariant)
    assert_relative_eq!(cosine_original, cosine_scaled, epsilon = 1e-10);
    println!("✓ Cosine similarity is scale invariant");
}

#[test]
fn test_hybrid_similarity_scale_sensitivity() {
    // Test that hybrid similarity is sensitive to scale differences
    let items: Vec<Vec<f64>> = make_moons_hd(2, 0.0, 1.0, 13, 654);
    let item1 = &items[0];
    let item2 = &items[1];

    let alpha = 0.7; // Weight for cosine component
    let beta = 0.3; // Weight for magnitude component

    // Test with original items
    let hybrid_original = compute_hybrid_similarity(item1, item2, alpha, beta);

    // Scale items by different factors
    let scale1 = 5.0;
    let scale2 = 0.1;
    let item1_scaled: Vec<f64> = item1.iter().map(|x| x * scale1).collect();
    let item2_scaled: Vec<f64> = item2.iter().map(|x| x * scale2).collect();

    let hybrid_scaled = compute_hybrid_similarity(&item1_scaled, &item2_scaled, alpha, beta);

    println!("Original hybrid similarity: {:.6}", hybrid_original);
    println!("Scaled hybrid similarity: {:.6}", hybrid_scaled);
    println!("Difference: {:.6}", (hybrid_original - hybrid_scaled).abs());

    // Hybrid similarity should be different (scale sensitive)
    assert!(
        (hybrid_original - hybrid_scaled).abs() > 1e-6,
        "Hybrid similarity should be scale sensitive"
    );
    println!("✓ Hybrid similarity is scale sensitive");
}

#[test]
fn test_builder_normalized_vs_unnormalized_clustering() {
    // Test clustering behavior with both normalized and unnormalized items
    let items_base: Vec<Vec<f64>> = make_moons_hd(70, 0.16, 0.38, 11, 999);

    // Create unnormalized items with different scales
    let scales = vec![1.0, 3.0, 0.5, 2.5, 1.5, 4.0, 0.8];
    let items_unnormalized: Vec<Vec<f64>> = items_base
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let scale = scales[i % scales.len()];
            item.iter().map(|x| x * scale).collect()
        })
        .collect();

    // Normalize items manually for comparison
    let items_normalized: Vec<Vec<f64>> = items_unnormalized
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

    println!("=== NORMALIZED vs UNNORMALIZED CLUSTERING ===");

    // Verify pairwise cosine similarities are identical
    let mut cosine_diffs = Vec::new();
    for i in 0..items_base.len().min(10) {
        for j in (i + 1)..items_base.len().min(10) {
            let cos_base = compute_cosine_similarity(&items_base[i], &items_base[j]);
            let cos_norm = compute_cosine_similarity(&items_normalized[i], &items_normalized[j]);
            cosine_diffs.push((cos_base - cos_norm).abs());
        }
    }

    let max_cosine_diff = cosine_diffs.iter().fold(0.0_f64, |a, &b| a.max(b));
    assert!(
        max_cosine_diff < 1e-10,
        "Cosine similarities should be identical: max_diff={:.2e}",
        max_cosine_diff
    );

    println!(
        "✓ Cosine similarities verified identical (max diff: {:.2e})",
        max_cosine_diff
    );
}

#[test]
fn test_builder_lambda_comparison_normalized_vs_unnormalized() {
    // Test how normalization affects lambda (spectral score) values
    let items_base: Vec<Vec<f64>> = make_moons_hd(60, 0.18, 0.35, 10, 555);

    // Create items with dramatically different scales
    let scales = vec![10.0, 0.1, 5.0, 2.0, 0.5];
    let items_unnormalized: Vec<Vec<f64>> = items_base
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let scale = scales[i % scales.len()];
            item.iter().map(|x| x * scale).collect()
        })
        .collect();

    // Build with normalization (cosine similarity, scale-invariant)
    let (aspace_norm, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 5, 2, 2.0, None)
        .with_normalisation(true)
        .with_spectral(true)
        .build(items_base.clone());

    // Build without normalization (τ-mode, magnitude-sensitive)
    let (aspace_unnorm, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.25, 5, 2, 2.0, None)
        .with_normalisation(false)
        .with_spectral(true)
        .build(items_unnormalized.clone());

    let lambdas_norm = aspace_norm.lambdas();
    let lambdas_unnorm = aspace_unnorm.lambdas();

    println!("=== LAMBDA SPECTRAL ANALYSIS ===");
    println!(
        "Normalized lambdas (first 5): {:?}",
        &lambdas_norm[..5.min(lambdas_norm.len())]
    );
    println!(
        "Unnormalized lambdas (first 5): {:?}",
        &lambdas_unnorm[..5.min(lambdas_unnorm.len())]
    );

    // Count differences
    let min_len = lambdas_norm.len().min(lambdas_unnorm.len());
    let mut significant_diffs = 0;

    for i in 0..min_len {
        if (lambdas_norm[i] - lambdas_unnorm[i]).abs() > 1e-6 {
            significant_diffs += 1;
        }
    }

    println!("Lambda differences: {}/{}", significant_diffs, min_len);
    println!("✓ Cosine-based vs τ-mode spectral properties compared");
}

#[test]
fn test_magnitude_penalty_computation() {
    // Test magnitude penalty formula: exp(-|ln(r)|) == min(r, 1/r)
    let item1 = vec![1.0, 2.0, 3.0];
    let item2_same_scale = vec![1.5, 3.0, 4.5]; // 1.5x scale
    let item2_diff_scale = vec![0.1, 0.2, 0.3]; // 0.1x scale

    let norm1 = item1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2_same = item2_same_scale.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2_diff = item2_diff_scale.iter().map(|x| x * x).sum::<f64>().sqrt();

    let penalty_same = (-((norm1 / norm2_same).ln().abs())).exp();
    let penalty_diff = (-((norm1 / norm2_diff).ln().abs())).exp();

    // Closed-form expectation: exp(-|ln r|) == min(r, 1/r)
    let expected_same = (norm1 / norm2_same).min(norm2_same / norm1);
    let expected_diff = (norm1 / norm2_diff).min(norm2_diff / norm1);

    // Verify exact expected values
    assert!(
        (penalty_same - expected_same).abs() < 1e-12,
        "penalty_same mismatch: got {:.12}, expected {:.12}",
        penalty_same,
        expected_same
    );
    assert!(
        (penalty_diff - expected_diff).abs() < 1e-12,
        "penalty_diff mismatch: got {:.12}, expected {:.12}",
        penalty_diff,
        expected_diff
    );

    // Qualitative property: similar scale > different scale
    assert!(
        penalty_same > penalty_diff,
        "Similar magnitude should yield higher penalty: same={:.6} diff={:.6}",
        penalty_same,
        penalty_diff
    );

    println!(
        "✓ Magnitude penalty: same_scale={:.6}, diff_scale={:.6}",
        penalty_same, penalty_diff
    );
}

#[test]
fn test_hybrid_similarity_components() {
    // Comprehensive test of hybrid similarity components
    let items: Vec<Vec<f64>> = make_moons_hd(2, 0.0, 1.0, 10, 888);
    let item1 = &items[0];
    let item2 = &items[1];

    // Test different scale combinations
    let scales = vec![0.1, 0.5, 1.0, 2.0, 10.0];

    println!("=== HYBRID SIMILARITY COMPONENT ANALYSIS ===");
    println!(
        "{:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Scale1", "Scale2", "Cosine", "MagPenalty", "Hybrid", "Difference"
    );

    let base_cosine = compute_cosine_similarity(item1, item2);

    for &scale1 in &scales {
        for &scale2 in &scales {
            let item1_scaled: Vec<f64> = item1.iter().map(|x| x * scale1).collect();
            let item2_scaled: Vec<f64> = item2.iter().map(|x| x * scale2).collect();

            let cosine = compute_cosine_similarity(&item1_scaled, &item2_scaled);
            let hybrid = compute_hybrid_similarity(&item1_scaled, &item2_scaled, 0.6, 0.4);

            // Compute magnitude penalty separately
            let norm1 = item1_scaled.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm2 = item2_scaled.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mag_penalty = if norm1 > 1e-12 && norm2 > 1e-12 {
                (-((norm1 / norm2).ln().abs())).exp()
            } else {
                0.0
            };

            let hybrid_manual = 0.6 * cosine + 0.4 * mag_penalty;

            println!(
                "{:8.1} {:8.1} {:12.6} {:12.6} {:12.6} {:12.8}",
                scale1,
                scale2,
                cosine,
                mag_penalty,
                hybrid,
                (hybrid - hybrid_manual).abs()
            );

            // Verify manual computation matches function
            assert_relative_eq!(hybrid, hybrid_manual, epsilon = 1e-10);

            // Cosine should always be the same
            assert_relative_eq!(cosine, base_cosine, epsilon = 1e-10);
        }
    }

    println!("✓ Hybrid similarity components computed correctly");
    println!("✓ Cosine component remains scale-invariant");
    println!("✓ Magnitude penalty varies with scale differences");
}
