use crate::graph::GraphParams;
use approx::assert_abs_diff_eq;
use smartcore::linalg::basic::arrays::{Array};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::laplacian::*;

use log::debug;

// Helper function for creating test vectors with known similarities
fn create_test_vectors() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 0.0, 0.0], // Unit vector along x-axis
        vec![0.8, 0.6, 0.0], // ~53° from x-axis, cosine ≈ 0.8
        vec![0.0, 1.0, 0.0], // Unit vector along y-axis
        vec![0.0, 0.8, 0.6], // ~53° from y-axis
        vec![0.0, 0.0, 1.0], // Unit vector along z-axis
    ]
}

fn default_params() -> GraphParams {
    GraphParams { eps: 0.5, k: 3, p: 2.0, sigma: Some(0.1), normalise: false }
}

#[test]
fn test_basic_laplacian_construction() {
    let items = create_test_vectors();
    let params = default_params();

    let laplacian = build_laplacian_matrix(items.clone(), &params);

    assert_eq!(laplacian.nnodes, 5);
    assert_eq!(laplacian.matrix.shape(), (5, 5));
    assert_eq!(laplacian.graph_params, params);
}

#[test]
fn test_laplacian_mathematical_properties() {
    let items = create_test_vectors();
    let params = default_params();
    let laplacian = build_laplacian_matrix(items, &params);

    let n = laplacian.nnodes;

    // Property 1: Row sums should be zero (within numerical precision)
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| laplacian.matrix.get((i, j))).sum();
        assert_abs_diff_eq!(row_sum, 0.0, epsilon = 1e-12);
    }

    // Property 2: Matrix should be symmetric
    for i in 0..n {
        for j in 0..n {
            let l_ij = *laplacian.matrix.get((i, j));
            let l_ji = *laplacian.matrix.get((j, i));
            assert_abs_diff_eq!(l_ij, l_ji, epsilon = 1e-12);
        }
    }

    // Property 3: Diagonal entries should be non-negative (degrees)
    for i in 0..n {
        let diagonal = *laplacian.matrix.get((i, i));
        assert!(
            diagonal >= -1e-12,
            "Diagonal L[{},{}] should be non-negative, got {:.6}",
            i,
            i,
            diagonal
        );
    }

    // Property 4: Off-diagonal entries should be non-positive
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let off_diag = *laplacian.matrix.get((i, j));
                assert!(
                    off_diag <= 1e-12,
                    "Off-diagonal L[{},{}] should be non-positive, got {:.6}",
                    i,
                    j,
                    off_diag
                );
            }
        }
    }
}

#[test]
fn test_cosine_similarity_based_construction() {
    // Create vectors with known cosine similarities
    let items = vec![
        vec![1.0, 0.0],     // cos(0°) = 1.0 with itself
        vec![0.707, 0.707], // cos(45°) ≈ 0.707 with [1,0]
        vec![0.0, 1.0],     // cos(90°) = 0.0 with [1,0]
        vec![-1.0, 0.0],    // cos(180°) = -1.0 with [1,0]
    ];

    let params = GraphParams {
        eps: 2.0, // Increased to allow more connections
        k: 3,
        p: 1.0,
        sigma: Some(0.5), // Increased sigma for better discrimination
        normalise: true,
    };

    let (_, adjacency) = build_laplacian_matrix_with_adjacency(&items, &params);
}

#[test]
fn test_eps_parameter_constraint() {
    let items = vec![
        vec![1.0, 0.0],
        vec![0.5, 0.0], // Distance = 1 - 0.5 = 0.5
        vec![0.0, 1.0], // Distance = 1 - 0 = 1.0
    ];

    // Test with restrictive eps
    let restrictive_params = GraphParams {
        eps: 0.3, // Only allow very close connections
        k: 10,
        p: 1.0,
        sigma: Some(0.1),
        normalise: true,
    };

    let (_, adjacency_restrictive) =
        build_laplacian_matrix_with_adjacency(&items, &restrictive_params);

    // Test with permissive eps
    let permissive_params = GraphParams {
        eps: 1.5, // Allow most connections
        k: 10,
        p: 1.0,
        sigma: Some(0.1),
        normalise: true,
    };

    let (_, adjacency_permissive) =
        build_laplacian_matrix_with_adjacency(&items, &permissive_params);

    // Count non-zero adjacency entries
    let count_restrictive = count_nonzero_adjacency(&adjacency_restrictive);
    let count_permissive = count_nonzero_adjacency(&adjacency_permissive);

    assert!(
        count_permissive >= count_restrictive,
        "Larger eps should allow more connections: {} >= {}",
        count_permissive,
        count_restrictive
    );
}

#[test]
fn test_k_parameter_constraint() {
    let items = create_test_vectors();

    let small_k_params = GraphParams {
        eps: 1.0,
        k: 1, // Only 1 neighbor per node
        p: 1.0,
        sigma: Some(0.1),
        normalise: true,
    };

    let large_k_params = GraphParams {
        eps: 1.0,
        k: 4, // Up to 4 neighbors per node
        p: 1.0,
        sigma: Some(0.1),
        normalise: true,
    };

    let (_, adj_small_k) =
        build_laplacian_matrix_with_adjacency(&items, &small_k_params);
    let (_, adj_large_k) =
        build_laplacian_matrix_with_adjacency(&items, &large_k_params);

    let connections_small_k = count_nonzero_adjacency(&adj_small_k);
    let connections_large_k = count_nonzero_adjacency(&adj_large_k);

    assert!(
        connections_large_k >= connections_small_k,
        "Larger k should allow more connections: {} >= {}",
        connections_large_k,
        connections_small_k
    );
}

#[test]
#[should_panic(expected = "items should be at least of shape (2,2): (1,1)")]
fn test_insufficient_data_panics() {
    let insufficient_items = vec![vec![1.0]]; // Only one item
    let params = default_params();
    build_laplacian_matrix(insufficient_items, &params);
}

#[test]
fn test_numerical_stability() {
    // Test with very small values that might cause numerical issues
    let small_values = vec![vec![1e-10, 2e-10], vec![3e-10, 1e-10], vec![2e-10, 3e-10]];

    let params =
        GraphParams { eps: 1.0, k: 2, p: 2.0, sigma: Some(1e-8), normalise: false };

    let laplacian = build_laplacian_matrix(small_values, &params);

    // Should produce finite values
    for i in 0..3 {
        for j in 0..3 {
            let val = *laplacian.matrix.get((i, j));
            assert!(
                val.is_finite(),
                "Matrix entry [{},{}] should be finite, got {}",
                i,
                j,
                val
            );
        }
    }
}

#[test]
fn test_performance_with_larger_dataset() {
    // Test with larger dataset to ensure algorithm completes reasonably
    let large_items: Vec<Vec<f64>> = crate::tests::test_data::QUORA_EMBEDDS
        .iter()
        .map(|inner_slice| inner_slice.to_vec())
        .collect();

    let params =
        GraphParams { eps: 0.8, k: 10, p: 2.0, sigma: Some(0.1), normalise: false };

    let start = std::time::Instant::now();
    let laplacian = build_laplacian_matrix(large_items, &params);
    let duration = start.elapsed();

    assert_eq!(laplacian.nnodes, 15);
    debug!("Large dataset (15 items) processed in {:?}", duration);

    // Sanity check - should complete in reasonable time
    assert!(duration.as_secs() < 5, "Should complete within 5 seconds");
}

// Helper function to count non-zero adjacency entries
fn count_nonzero_adjacency(adjacency: &DenseMatrix<f64>) -> usize {
    let (rows, cols) = adjacency.shape();
    let mut count = 0;
    for i in 0..rows {
        for j in 0..cols {
            if i != j && adjacency.get((i, j)).abs() > 1e-15 {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn test_arrowspace_integration_pattern() {
    // Simulate the usage pattern from ArrowSpace protein example
    let protein_like_data = vec![
        vec![0.82, 0.11, 0.43, 0.28, 0.64],
        vec![0.79, 0.12, 0.45, 0.29, 0.61],
        vec![0.78, 0.13, 0.46, 0.27, 0.62],
        vec![0.81, 0.10, 0.44, 0.26, 0.63],
    ];

    let arrowspace_params = GraphParams {
        eps: 1e-2, // Similar to the original example
        k: 6,
        p: 2.0,
        sigma: None, // Use default
        normalise: true,
    };

    let laplacian = build_laplacian_matrix(protein_like_data, &arrowspace_params);

    // Verify it produces the expected structure for lambda computation
    assert_eq!(laplacian.nnodes, 4);
    assert_eq!(laplacian.matrix.shape(), (4, 4));

    // Should be ready for eigenvalue computation in lambda synthesis
    for i in 0..4 {
        let row_sum: f64 = (0..4).map(|j| laplacian.matrix.get((i, j))).sum();
        assert!(
            (row_sum).abs() < 1e-10,
            "Invalid Laplacian structure for lambda computation"
        );
    }
}

#[test]
fn test_optimized_dense_matrix_laplacian() {
    let items = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.8, 0.2, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.8, 0.2],
    ];

    let params =
        GraphParams { eps: 0.8, k: 2, p: 2.0, sigma: Some(0.1), normalise: false };

    let laplacian = build_laplacian_matrix(items, &params);

    // Verify structure
    assert_eq!(laplacian.nnodes, 4);
    assert_eq!(laplacian.matrix.shape().0, 4);
    assert_eq!(laplacian.matrix.shape().1, 4);

    // Verify Laplacian properties
    // 1. Row sums should be zero
    for i in 0..4 {
        let row_sum: f64 = (0..4).map(|j| laplacian.matrix.get((i, j))).sum();
        assert!(
            row_sum.abs() < 1e-10,
            "Row {} sum should be ~0, got {:.2e}",
            i,
            row_sum
        );
    }

    // 2. Matrix should be symmetric
    for i in 0..4 {
        for j in 0..4 {
            let diff =
                (laplacian.matrix.get((i, j)) - laplacian.matrix.get((j, i))).abs();
            assert!(diff < 1e-10, "Matrix should be symmetric at ({},{})", i, j);
        }
    }

    // 3. Diagonal entries should be non-negative
    for i in 0..4 {
        let diagonal: f64 = *laplacian.matrix.get((i, i));
        assert!(
            diagonal >= -1e-10,
            "Diagonal L[{},{}] should be non-negative, got {:.6}",
            i,
            i,
            diagonal
        );
    }

    debug!("Optimized DenseMatrix Laplacian test passed");
}

#[test]
fn test_with_adjacency_output() {
    let items = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];

    let params =
        GraphParams { eps: 0.5, k: 2, p: 1.0, sigma: Some(0.2), normalise: false };

    let (laplacian, adjacency) = build_laplacian_matrix_with_adjacency(&items, &params);

    // Verify adjacency has zero diagonal
    for i in 0..3 {
        assert_eq!(*adjacency.get((i, i)), 0.0);
    }

    // Verify Laplacian = Degree - Adjacency
    for i in 0..3 {
        let degree: f64 = (0..3).map(|j| adjacency.get((i, j))).sum();
        assert!((laplacian.matrix.get((i, i)) - degree).abs() < 1e-10);

        for j in 0..3 {
            if i != j {
                let expected = -adjacency.get((i, j));
                assert!((laplacian.matrix.get((i, j)) - expected).abs() < 1e-10);
            }
        }
    }

    debug!("Adjacency + Laplacian test passed");
}
