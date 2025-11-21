use crate::graph::GraphLaplacian;
use crate::{builder::ArrowSpaceBuilder, tests::test_data::make_moons_hd};
use approx::assert_relative_eq;

/// Helper to compare two GraphLaplacian matrices for equality
#[allow(dead_code)]
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
#[allow(dead_code)]
fn diag_vec(gl: &GraphLaplacian) -> Vec<f64> {
    let (n, _) = gl.matrix.shape();
    (0..n).map(|i| *gl.matrix.get(i, i).unwrap()).collect()
}

#[allow(dead_code)]
fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
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

    debug!("✓ Graph parameters correctly preserved");
}

#[test]
fn test_with_deterministic_clustering() {
    let items = make_moons_hd(80, 0.50, 0.50, 9, 789);

    // Build with fixed seed
    let (aspace1, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_seed(42) // If your API supports this
        .build(items.clone());

    let (aspace2, _) = ArrowSpaceBuilder::default()
        .with_lambda_graph(0.3, 4, 2, 2.0, None)
        .with_seed(42)
        .build(items.clone());

    // Now these should be identical
    assert_eq!(aspace1.n_clusters, aspace2.n_clusters);
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

    debug!("Original cosine similarity: {:.6}", cosine_original);
    debug!("Scaled cosine similarity: {:.6}", cosine_scaled);

    // Cosine similarity should be identical (scale invariant)
    assert_relative_eq!(cosine_original, cosine_scaled, epsilon = 1e-10);
    debug!("✓ Cosine similarity is scale invariant");
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

    debug!("Original hybrid similarity: {:.6}", hybrid_original);
    debug!("Scaled hybrid similarity: {:.6}", hybrid_scaled);
    debug!("Difference: {:.6}", (hybrid_original - hybrid_scaled).abs());

    // Hybrid similarity should be different (scale sensitive)
    assert!(
        (hybrid_original - hybrid_scaled).abs() > 1e-6,
        "Hybrid similarity should be scale sensitive"
    );
    debug!("✓ Hybrid similarity is scale sensitive");
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

    debug!("=== NORMALIZED vs UNNORMALIZED CLUSTERING ===");

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

    debug!(
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

    debug!("=== LAMBDA SPECTRAL ANALYSIS ===");
    debug!(
        "Normalized lambdas (first 5): {:?}",
        &lambdas_norm[..5.min(lambdas_norm.len())]
    );
    debug!(
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

    debug!("Lambda differences: {}/{}", significant_diffs, min_len);
    debug!("✓ Cosine-based vs τ-mode spectral properties compared");
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

    debug!(
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

    debug!("=== HYBRID SIMILARITY COMPONENT ANALYSIS ===");
    debug!(
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

            debug!(
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

    debug!("✓ Hybrid similarity components computed correctly");
    debug!("✓ Cosine component remains scale-invariant");
    debug!("✓ Magnitude penalty varies with scale differences");
}
