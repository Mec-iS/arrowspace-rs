use crate::reduction::{ImplicitProjection, compute_jl_dimension, project_matrix};
use smartcore::linalg::basic::{
    arrays::{Array, Array2},
    matrix::DenseMatrix,
};

// ============================================================================
// ImplicitProjection Tests
// ============================================================================

#[test]
fn test_implicit_projection_creates() {
    let proj = ImplicitProjection::new(100, 10, Some(42));
    assert_eq!(proj.original_dim, 100);
    assert_eq!(proj.reduced_dim, 10);
    assert!(proj.seed > 0);
}

#[test]
fn test_implicit_projection_dimensions() {
    let proj = ImplicitProjection::new(50, 8, Some(42));
    let query = vec![0.5; 50];

    let projected = proj.project(&query);

    assert_eq!(projected.len(), 8);
    assert!(projected.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_implicit_projection_deterministic() {
    // Same seed should produce same projection
    let proj = ImplicitProjection::new(30, 5, Some(42));
    let query = vec![1.0; 30];

    let result1 = proj.project(&query);
    let result2 = proj.project(&query);

    assert_eq!(result1, result2);
}

#[test]
fn test_implicit_projection_different_seeds() {
    // Different instances should have different seeds
    let proj1 = ImplicitProjection::new(20, 5, None);
    let proj2 = ImplicitProjection::new(20, 5, None);

    // Seeds should be different (probabilistically)
    assert_ne!(proj1.seed, proj2.seed);

    let query = vec![1.0; 20];
    let result1 = proj1.project(&query);
    let result2 = proj2.project(&query);

    // Results should differ due to different seeds
    assert_ne!(result1, result2);
}

#[test]
fn test_implicit_projection_zero_vector() {
    let proj = ImplicitProjection::new(40, 10, Some(42));
    let query = vec![0.0; 40];

    let projected = proj.project(&query);

    assert_eq!(projected.len(), 10);
    // All should be near-zero
    assert!(projected.iter().all(|&x| x.abs() < 1e-10));
}

#[test]
fn test_implicit_projection_linearity() {
    let proj = ImplicitProjection::new(25, 6, Some(42));

    let query = vec![1.0; 25];
    let scaled_query: Vec<f64> = query.iter().map(|x| x * 2.0).collect();

    let proj1 = proj.project(&query);
    let proj2 = proj.project(&scaled_query);

    // Projection is linear: project(2x) = 2*project(x)
    for i in 0..proj1.len() {
        let expected = proj1[i] * 2.0;
        let actual = proj2[i];
        assert!(
            (expected - actual).abs() < 1e-9,
            "Linearity violation at {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_implicit_projection_preserves_scale() {
    let proj = ImplicitProjection::new(50, 15, Some(42));
    let query = vec![1.0; 50];

    let projected = proj.project(&query);

    let orig_norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();
    let proj_norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();

    let ratio = proj_norm / orig_norm;

    // JL guarantees approximate norm preservation
    assert!(ratio > 0.5 && ratio < 2.0);
}

#[test]
fn test_implicit_projection_non_trivial() {
    let proj = ImplicitProjection::new(30, 8, Some(42));
    let query = vec![1.0; 30];

    let projected = proj.project(&query);

    // Should have at least one non-zero value
    let has_nonzero = projected.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero);
}

// ============================================================================
// project_matrix Tests
// ============================================================================

#[test]
fn test_project_matrix_dimensions() {
    let data = vec![1.0; 60]; // 3 rows × 20 cols
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 3, 20, 1);
    let proj = ImplicitProjection::new(20, 5, Some(42));

    let projected = project_matrix(&matrix, &proj);

    assert_eq!(projected.shape(), (3, 5));
}

#[test]
fn test_project_matrix_preserves_rows() {
    let data = vec![0.5; 100]; // 10 rows × 10 cols
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 10, 10, 1);
    let proj = ImplicitProjection::new(10, 3, Some(42));

    let projected = project_matrix(&matrix, &proj);

    assert_eq!(projected.shape().0, 10);
    assert_eq!(projected.shape().1, 3);
}

#[test]
fn test_project_matrix_zero_matrix() {
    let data = vec![0.0; 80]; // 4 rows × 20 cols
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 4, 20, 0);
    let proj = ImplicitProjection::new(20, 6, Some(42));

    let projected = project_matrix(&matrix, &proj);

    // All values should be near-zero
    for i in 0..projected.shape().0 {
        for j in 0..projected.shape().1 {
            assert!(projected.get((i, j)).abs() < 1e-10);
        }
    }
}

#[test]
fn test_project_matrix_different_rows_different_projections() {
    let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let matrix = DenseMatrix::from_iterator(data.into_iter(), 3, 4, 0);
    let proj = ImplicitProjection::new(4, 2, Some(42));

    let projected = project_matrix(&matrix, &proj);

    // Extract rows
    let row0: Vec<f64> = (0..2).map(|j| *projected.get((0, j))).collect();
    let row1: Vec<f64> = (0..2).map(|j| *projected.get((1, j))).collect();
    let row2: Vec<f64> = (0..2).map(|j| *projected.get((2, j))).collect();

    // Different input rows should produce different projections
    assert_ne!(row0, row1);
    assert_ne!(row1, row2);
}

// ============================================================================
// compute_jl_dimension Tests
// ============================================================================

#[test]
fn test_jl_dimension_basic() {
    let n = 1000;
    let epsilon = 0.1;

    let dim = compute_jl_dimension(n, epsilon);

    assert!(dim >= 32);
    assert!(dim > 0);
}

#[test]
fn test_jl_dimension_minimum_bound() {
    let n = 2;
    let epsilon = 0.9;

    let dim = compute_jl_dimension(n, epsilon);

    // For n=2, ε=0.9: 8*ln(2)/0.81 ≈ 6.9 → clamped to 32
    assert_eq!(dim, 32);
}

#[test]
fn test_jl_dimension_grows_with_n() {
    let epsilon = 0.1;

    let dim_100 = compute_jl_dimension(100, epsilon);
    let dim_10000 = compute_jl_dimension(10000, epsilon);

    assert!(dim_10000 > dim_100);
}

#[test]
fn test_jl_dimension_inversely_proportional_epsilon() {
    let n = 5000;

    let dim_01 = compute_jl_dimension(n, 0.1);
    let dim_02 = compute_jl_dimension(n, 0.2);

    assert!(dim_01 > dim_02);
}

#[test]
fn test_jl_dimension_large_dataset() {
    let n = 1_000_000;
    let epsilon = 0.1;

    let dim = compute_jl_dimension(n, epsilon);

    assert!(dim >= 10_000);
    assert!(dim < 20_000);
}

#[test]
fn test_jl_dimension_tight_epsilon() {
    let n = 10_000;
    let epsilon = 0.05;

    let dim = compute_jl_dimension(n, epsilon);

    assert!(dim >= 25_000);
}

#[test]
fn test_jl_dimension_formula_correctness() {
    let n = 1000;
    let epsilon = 0.1;

    let dim = compute_jl_dimension(n, epsilon);

    let expected = (8.0 * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
    let expected_with_min = expected.max(32);

    assert_eq!(dim, expected_with_min);
}

#[test]
fn test_jl_dimension_reasonable_range() {
    let test_cases = vec![(100, 0.2), (1000, 0.15), (10000, 0.1)];

    for (n, eps) in test_cases {
        let dim = compute_jl_dimension(n, eps);

        assert!(dim >= 32);
        assert!(dim < 100_000);
    }
}

#[test]
fn test_jl_dimension_edge_case_small_epsilon() {
    let n = 500;
    let epsilon = 0.01;

    let dim = compute_jl_dimension(n, epsilon);

    assert!(dim >= 400_000);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_pipeline_implicit_projection() {
    // Simulate full pipeline: matrix → project → verify
    let n_samples = 20;
    let orig_dim = 100;
    let reduced_dim = 15;

    // Create test data
    let data: Vec<f64> = (0..n_samples * orig_dim)
        .map(|i| (i as f64) * 0.01)
        .collect();
    let matrix = DenseMatrix::from_iterator(data.into_iter(), n_samples, orig_dim, 0);

    // Create projection
    let proj = ImplicitProjection::new(orig_dim, reduced_dim, Some(42));

    // Project matrix
    let projected = project_matrix(&matrix, &proj);

    assert_eq!(projected.shape(), (n_samples, reduced_dim));

    // Verify all values are finite
    for i in 0..projected.shape().0 {
        for j in 0..projected.shape().1 {
            assert!(projected.get((i, j)).is_finite());
        }
    }
}

#[test]
fn test_memory_efficiency() {
    // ImplicitProjection should be tiny (just 24 bytes)
    let proj = ImplicitProjection::new(1000, 100, Some(42));

    // Verify it can project without storing the matrix
    let query = vec![1.0; 1000];
    let projected = proj.project(&query);

    assert_eq!(projected.len(), 100);

    // The struct should be minimal size
    assert_eq!(std::mem::size_of::<ImplicitProjection>(), 24); // 3 usizes
}
