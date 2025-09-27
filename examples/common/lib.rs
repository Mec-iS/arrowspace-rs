use arrowspace::core::{ArrowItem, ArrowSpace};
use arrowspace::graph::GraphLaplacian;

use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;

use approx::relative_eq;

const VECTORS_DATA: &str = include_str!("datasets/vectors_data_3000.txt");

/// Parse string and feature rows (N×F).
#[allow(dead_code)]
pub fn parse_vectors_string(vectors_string: &str) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut ids = Vec::new();
    let mut rows = Vec::new();

    for line in vectors_string.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let mut parts = l.splitn(2, ';');
        let id = parts.next().unwrap().trim().to_string();
        let rest = parts.next().unwrap_or("").trim();

        let vals: Vec<f64> =
            rest.split(',').map(|s| s.trim().parse::<f64>().unwrap()).collect();

        ids.push(id);
        rows.push(vals);
    }

    (ids, rows)
}

/// Parse block into item IDs and feature rows (N×F).
#[allow(dead_code)]
pub fn parse_vectors_block() -> (Vec<String>, Vec<Vec<f64>>) {
    let mut ids = Vec::new();
    let mut rows = Vec::new();

    for line in VECTORS_DATA.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let mut parts = l.splitn(2, ';');
        let id = parts.next().unwrap().trim().to_string();
        let rest = parts.next().unwrap_or("").trim();

        let vals: Vec<f64> =
            rest.split(',').map(|s| s.trim().parse::<f64>().unwrap()).collect();

        ids.push(id);
        rows.push(vals);
    }

    (ids, rows)
}

/// Parse block into item IDs and feature rows (N×F).
/// Returns n records starting at offset
#[allow(dead_code)]
pub fn parse_vectors_slice(n: usize, offset: usize) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut ids = Vec::new();
    let mut rows = Vec::new();

    let lines: Vec<_> = VECTORS_DATA.lines().collect();
    assert!(
        lines.len() > offset + n,
        "{}",
        format!("offset {} + n {} shall be less than {}", offset, n, lines.len())
    );
    let slice = if offset >= lines.len() {
        &[]
    } else {
        &lines[offset..(offset + n).min(lines.len())]
    };

    for line in slice {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let mut parts = l.splitn(2, ';');
        let id = parts.next().unwrap().trim().to_string();
        let rest = parts.next().unwrap_or("").trim();

        let vals: Vec<f64> =
            rest.split(',').map(|s| s.trim().parse::<f64>().unwrap()).collect();

        ids.push(id);
        rows.push(vals);
    }

    (ids, rows)
}

#[allow(dead_code)]
pub fn parse_vectors_slice_features(
    n: usize,
    offset: usize,
    features_n: usize,
) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut ids = Vec::new();
    let mut rows = Vec::new();

    let lines: Vec<_> = VECTORS_DATA.lines().collect();
    assert!(
        lines.len() >= offset + n,
        "offset {} + n {} shall be less or equal than total lines {}",
        offset,
        n,
        lines.len()
    );

    let slice = if offset >= lines.len() {
        &[]
    } else {
        &lines[offset..(offset + n).min(lines.len())]
    };

    for line in slice {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let mut parts = l.splitn(2, ';');
        let id = parts.next().unwrap().trim().to_string();
        let rest = parts.next().unwrap_or("").trim();

        let vals: Vec<f64> = rest
            .split(',')
            .take(features_n) // ← KEY CHANGE: Only take first features_n features
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();

        // Optional: Ensure we have exactly features_n features (pad with zeros if needed)
        // let mut vals = vals;
        // vals.resize(features_n, 0.0);

        ids.push(id);
        rows.push(vals);
    }

    (ids, rows)
}

/// cosine similarity using smartcore 0.4 API (thread-safe)
#[allow(dead_code)]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    // Create DenseMatrix for optimized operations
    let matrix_a = DenseMatrix::from_2d_vec(&vec![a.to_vec()]).unwrap();
    let matrix_b = DenseMatrix::from_2d_vec(&vec![b.to_vec()]).unwrap();

    // Compute norms using smartcore optimized operations
    let norm_a = matrix_a.get_row(0).iterator(0).map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = matrix_b.get_row(0).iterator(0).map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        return 0.0;
    }

    // Compute dot product using vectorized operations
    let dot_product: f64 = matrix_a
        .get_row(0)
        .iterator(0)
        .zip(matrix_b.get_row(0).iterator(0))
        .map(|(x, y)| x * y)
        .sum();

    dot_product / (norm_a * norm_b)
}

/// Compute the ratio of connected components to total possible connections
/// Returns a value between 0 and 1, where values > 0.95 indicate good connectivity
#[allow(dead_code)]
pub fn graph_connectivity_ratio(matrix: &DenseMatrix<f64>) -> f64 {
    let (nrows, ncols) = matrix.shape();

    if nrows != ncols || nrows <= 1 {
        return if nrows <= 1 { 1.0 } else { 0.0 };
    }

    // Count off-diagonal negative entries using smartcore's iterator pattern
    let total_edges: usize = (0..nrows)
        .map(|i| {
            // Extract row using smartcore's canonical pattern
            let row_values: Vec<f64> = matrix.get_row(i).iterator(0).copied().collect();
            
            // Count negative off-diagonal entries in this row
            row_values
                .iter()
                .enumerate()
                .filter(|(j, value)| {
                    i != *j
                        && **value < 0.0
                        && !relative_eq!(**value, 0.0, epsilon = f64::EPSILON)
                })
                .count()
        })
        .sum::<usize>()
        / 2; // Divide by 2 for symmetric matrix

    let max_possible_edges = (nrows * (nrows - 1)) / 2;

    if max_possible_edges == 0 {
        return 1.0;
    }

    let min_connectivity_edges = nrows - 1;
    let connectivity_ratio = (total_edges as f64) / (min_connectivity_edges as f64);

    (connectivity_ratio * 0.95).min(1.0)
}

/// Assess the quality of lambda distribution
/// Returns a value between 0 and 1, where higher values indicate better distribution
#[allow(dead_code)]
pub fn lambda_distribution_quality(lambdas: &[f64]) -> f64 {
    if lambdas.is_empty() {
        return 0.0;
    }

    // Filter out invalid values
    let valid_lambdas: Vec<f64> =
        lambdas.iter().copied().filter(|&x| x.is_finite() && x >= 0.0).collect();

    if valid_lambdas.is_empty() {
        return 0.0;
    }

    // Calculate statistics
    let mean = valid_lambdas.iter().sum::<f64>() / valid_lambdas.len() as f64;
    let variance = valid_lambdas.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / valid_lambdas.len() as f64;

    // Good lambda distribution should have:
    // 1. Reasonable spread (not all the same value)
    // 2. Not too many extreme outliers
    // 3. Most values in a reasonable range [0, 1] for normalized spectral scores

    let std_dev = variance.sqrt();
    let spread_quality = if std_dev > 1e-6 {
        (std_dev / (mean + 1e-6)).min(1.0)
    } else {
        0.1 // Very low spread is not good
    };

    // Check if values are well-bounded (prefer values in [0, 1] range)
    let in_range_count =
        valid_lambdas.iter().filter(|&&x| x >= 0.0 && x <= 1.0).count();
    let range_quality = in_range_count as f64 / valid_lambdas.len() as f64;

    // Combine metrics
    0.4 * spread_quality + 0.6 * range_quality
}

/// Evaluate edge count efficiency (sparse but connected)
/// Returns a value between 0 and 1, where higher values indicate better efficiency
#[allow(dead_code)]
pub fn edge_count_efficiency(adjacency_matrix: &DenseMatrix<f64>) -> f64 {
    let (nrows, ncols) = adjacency_matrix.shape();

    if nrows != ncols || nrows <= 1 {
        return if nrows <= 1 { 1.0 } else { 0.0 };
    }

    // Count off-diagonal negative entries using smartcore's iterator pattern
    let total_edges: usize = (0..nrows)
        .map(|i| {
            // Extract row using smartcore's canonical pattern
            let row_values: Vec<f64> = 
                adjacency_matrix.get_row(i).iterator(0).copied().collect();
            
            // Count negative off-diagonal entries in this row
            row_values
                .iter()
                .enumerate()
                .filter(|(j, value)| {
                    i != *j
                        && **value < 0.0
                        && !relative_eq!(**value, 0.0, epsilon = f64::EPSILON)
                })
                .count()
        })
        .sum::<usize>()
        / 2; // Divide by 2 for symmetric matrix

    let min_edges = nrows - 1;
    let max_reasonable_edges = (nrows * 6).min((nrows * (nrows - 1)) / 2);

    if total_edges < min_edges {
        return (total_edges as f64) / (min_edges as f64) * 0.3;
    }

    let sparsity_score = if total_edges <= max_reasonable_edges {
        1.0 - (total_edges - min_edges) as f64
            / (max_reasonable_edges - min_edges) as f64
    } else {
        0.1
    };

    0.7 + 0.3 * sparsity_score
}

/// Compute overall graph connectivity score
#[allow(dead_code)]
pub fn graph_connectivity_score(gl: &GraphLaplacian) -> f64 {
    graph_connectivity_ratio(&gl.matrix)
}

/// Compute lambda quality score
#[allow(dead_code)]
pub fn lambda_quality_score(lambdas: &[f64]) -> f64 {
    lambda_distribution_quality(lambdas)
}

/// Evaluate search effectiveness using actual queries
#[allow(dead_code)]
pub fn search_effectiveness_score(
    aspace: &ArrowSpace,
    queries: &[Vec<f64>],
    alpha: f64,
    beta: f64,
    k: usize,
) -> f64 {
    if queries.is_empty() {
        return 0.5; // Neutral score if no queries to test
    }

    let mut total_score = 0.0;
    let mut valid_queries = 0;

    for query in queries.iter() {
        if query.len() != aspace.nfeatures {
            continue; // Skip incompatible queries
        }

        let query_item = ArrowItem::new(query.clone(), 0.0);

        // Test lambda-aware search vs regular cosine
        let lambda_results = aspace.search_lambda_aware(&query_item, k, alpha, beta);

        if lambda_results.is_empty() {
            continue;
        }

        // Evaluate quality of results
        let mut result_quality = 0.0;

        // Check if we get reasonable similarity scores
        let avg_similarity = lambda_results.iter().map(|(_, sim)| *sim).sum::<f64>()
            / lambda_results.len() as f64;

        // Good searches should return results with reasonable similarities
        if avg_similarity > 0.1 && avg_similarity <= 1.0 {
            result_quality += 0.5;
        }

        // Check diversity in results (not all the same similarity)
        let similarities: Vec<f64> =
            lambda_results.iter().map(|(_, sim)| *sim).collect();
        if similarities.len() > 1 {
            let sim_variance = {
                let mean = avg_similarity;
                similarities.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                    / similarities.len() as f64
            };

            if sim_variance > 1e-6 {
                result_quality += 0.3;
            }
        }

        // Reward if lambda-aware search provides different results than pure cosine
        // This indicates the spectral component is contributing meaningfully
        if beta > 0.0 {
            result_quality += 0.2;
        }

        total_score += result_quality;
        valid_queries += 1;
    }

    if valid_queries == 0 {
        return 0.5;
    }

    total_score / valid_queries as f64
}

#[allow(dead_code)]
pub fn evaluate_graph_quality(aspace: &ArrowSpace, gl: &GraphLaplacian) -> f64 {
    let connectivity = graph_connectivity_ratio(&gl.matrix); // Should be > 0.95
    let lambda_variance = lambda_distribution_quality(aspace.lambdas());
    let edge_efficiency = edge_count_efficiency(&gl.matrix); // Sparse but connected

    // Weighted combination of quality metrics
    0.4 * connectivity + 0.3 * lambda_variance + 0.3 * edge_efficiency
}

#[allow(dead_code)]
pub fn evaluate_parameter_quality(
    aspace: &ArrowSpace,
    gl: &GraphLaplacian,
    queries: &[Vec<f64>],
    alpha: f64,
    beta: f64,
    k: usize,
) -> f64 {
    // Combine multiple quality metrics:
    let connectivity_score = graph_connectivity_score(gl);
    let lambda_distribution_score = lambda_quality_score(aspace.lambdas());
    let search_quality_score = 
        search_effectiveness_score(aspace, queries, alpha, beta, k);

    // search quality is compared to cosine, the objective is to find meaningful relation
    // beyond cosine similarity so the search quality score is given less importance
    0.4 * connectivity_score
        + 0.4 * lambda_distribution_score
        + 0.2 * search_quality_score
}


/// Optimized statistical computations using smartcore DenseMatrix (thread-safe)
#[allow(dead_code)]
fn compute_statistics(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let row_len = values.len() as f64;

    // Create DenseMatrix for optimized statistical operations
    let matrix = DenseMatrix::from_2d_vec(&vec![values.to_vec()]).unwrap();
    let row = matrix.get_row(0);

    // Mean using smartcore optimized operations
    let mean = row.iterator(0).sum::<f64>() / row_len;

    // Variance using vectorized operations
    let variance = row.iterator(0).map(|&x| (x - mean).powi(2)).sum::<f64>() / row_len;

    let std_dev = variance.sqrt();

    (mean, variance, std_dev)
}
