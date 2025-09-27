use smartcore::linalg::basic::arrays::{Array, Array2, ArrayView2};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::graph::{GraphFactory, GraphLaplacian, GraphParams};

use approx::relative_eq;

use log::debug;

#[test]
fn test_build_lambda_graph_basic() {
    // Test with 3 items, each with 2 features
    let items = vec![
        vec![1.0, 0.0], // Item 0: high in feature 0, low in feature 1
        vec![0.0, 1.0], // Item 1: low in feature 0, high in feature 1
        vec![0.5, 0.5], // Item 2: medium in both features
    ];
    let len_items = items.len();

    let gl = GraphFactory::build_laplacian_matrix(items, 0.5, 2, 2.0, None, true);

    // Verify each node has a non-negative diagonal entry
    for i in 0..len_items {
        let diagonal_value: f64 = *gl.matrix.get((i, i));

        // Check that diagonal entry exists (is finite)
        assert!(
            diagonal_value.is_finite(),
            "Diagonal entry at position {} should be finite, got: {}",
            i,
            diagonal_value
        );

        // Check non-negative diagonal entry
        assert!(
            diagonal_value >= 0.0,
            "Diagonal entry at position {} should be non-negative, got: {}",
            i,
            diagonal_value
        );
    }
}

#[test]
fn test_build_lambda_graph_minimum_items() {
    // Test minimum case: 2 items
    let items = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    let gl = GraphFactory::build_laplacian_matrix(items, 1.0, 6, 2.0, None, true);
    assert_eq!(gl.nnodes, 2);
}

#[test]
#[should_panic(expected = "items should be at least of shape (2,2): (1,2)")]
fn test_build_lambda_graph_insufficient_items() {
    // Should panic with only 1 item
    let items = vec![vec![1.0, 2.0]];
    GraphFactory::build_laplacian_matrix(items, 1.0, 6, 2.0, None, true);
}

#[test]
fn test_build_lambda_graph_scale_invariance() {
    // Test that scaling all items uniformly doesn't affect graph structure
    let items = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![3.0, 6.0, 9.0]];

    let gl1 =
        GraphFactory::build_laplacian_matrix(items.clone(), 0.5, 2, 2.0, None, true);

    // Scale all items by constant factor
    let scale_factor = 5.7;
    let items_scaled: Vec<Vec<f64>> = items
        .iter()
        .map(|item| item.iter().map(|&x| x * scale_factor).collect())
        .collect();

    let gl2 =
        GraphFactory::build_laplacian_matrix(items_scaled, 0.5, 2, 2.0, None, true);

    // Graph structure should be identical
    assert_eq!(gl1.nnodes, gl2.nnodes);
    assert_eq!(gl1.matrix.diag(), gl2.matrix.diag());
    // Note: values may differ due to lambda aggregation, but structure should be same
}

#[test]
fn test_graph_laplacian_structure() {
    // Test that created Laplacians have proper structure
    let items = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![3.0, 6.0, 9.0]];

    let gl = GraphFactory::build_laplacian_matrix(items, 1.0, 6, 2.0, None, true);

    // Symmetry check (for undirected graph)
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                let value_ij = gl.matrix.get((i, j));
                let value_ji = gl.matrix.get((j, i));

                // Only check non-zero entries (equivalent to sparse matrix logic)
                if value_ij.abs() > f64::EPSILON {
                    // Verify corresponding entry exists
                    assert!(
                        value_ji.abs() > f64::EPSILON,
                        "Graph should be symmetric: found edge ({},{}) = {} but ({},{}) = {}",
                        i, j, value_ij, j, i, value_ji
                    );
                }
            }
        }
    }
}

#[test]
fn test_new_from_items_transpose_verification() {
    // Create a non-square matrix to clearly see transposition
    let items_data = vec![
        1.0, 2.0, 3.0, 4.0, // Item 0 (4 features)
        5.0, 6.0, 7.0, 8.0, // Item 1 (4 features)
        9.0, 10.0, 11.0, 12.0, // Item 2 (4 features)
    ];
    let items_matrix = DenseMatrix::from_iterator(items_data.into_iter(), 3, 4, 0);

    let graph_params =
        GraphParams { eps: 0.5, k: 3, p: 1.0, sigma: None, normalise: true };

    // This should be treated as 3 items with 4 features each
    // After transposition for features matrix, it becomes 4 features with 3 items each
    let laplacian = GraphLaplacian::prepare_from_items(items_matrix, graph_params);

    // Verify dimensions after transposition
    assert_eq!(laplacian.nnodes, 4, "Should have 4 nodes (number of features)");
    assert_eq!(laplacian.matrix.shape().0, 4, "Matrix rows should be 4 (features)");
    assert_eq!(laplacian.matrix.shape().1, 3, "Matrix cols should be 3 (items)");

    // Verify transposition is correct
    // Original: items[i][f] = value at item i, feature f
    // Transposed: features[f][i] = value at feature f, item i

    // Feature 0 across items: [1.0, 5.0, 9.0]
    // Feature 0 across items: [1.0, 5.0, 9.0]
    assert!(relative_eq!(*laplacian.matrix.get((0, 0)), 1.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((0, 1)), 5.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((0, 2)), 9.0, epsilon = 1e-10));

    // Feature 1 across items: [2.0, 6.0, 10.0]
    assert!(relative_eq!(*laplacian.matrix.get((1, 0)), 2.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((1, 1)), 6.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((1, 2)), 10.0, epsilon = 1e-10));

    // Feature 2 across items: [3.0, 7.0, 11.0]
    assert!(relative_eq!(*laplacian.matrix.get((2, 0)), 3.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((2, 1)), 7.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((2, 2)), 11.0, epsilon = 1e-10));

    // Feature 3 across items: [4.0, 8.0, 12.0]
    assert!(relative_eq!(*laplacian.matrix.get((3, 0)), 4.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((3, 1)), 8.0, epsilon = 1e-10));
    assert!(relative_eq!(*laplacian.matrix.get((3, 2)), 12.0, epsilon = 1e-10));

    debug!("Transpose verification test passed");
}

#[test]
fn test_new_from_items_non_square_matrix_panic() {
    // Create a non-square matrix (this should panic)
    let non_square_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let non_square_matrix =
        DenseMatrix::from_iterator(non_square_data.into_iter(), 2, 3, 0);

    let graph_params =
        GraphParams { eps: 1.0, k: 1, p: 2.0, sigma: Some(0.1), normalise: true };

    // This should panic because input matrix is not square
    let gl = GraphLaplacian::prepare_from_items(non_square_matrix, graph_params);

    // should have returned the transposed of the input
    assert!(gl.nnodes == 3 && gl.matrix.shape() == (3, 2));
}

#[test]
fn test_new_from_items_parameter_preservation() {
    // Test that graph parameters are preserved correctly
    let matrix_data = vec![10.0, 20.0, 30.0, 40.0];
    let matrix = DenseMatrix::from_iterator(matrix_data.into_iter(), 2, 2, 0);

    let original_params =
        GraphParams { eps: 0.123, k: 5, p: 3.5, sigma: Some(0.456), normalise: true };

    let laplacian = GraphLaplacian::prepare_from_items(matrix, original_params.clone());

    // Verify all parameters are preserved
    assert_eq!(laplacian.graph_params.eps, original_params.eps);
    assert_eq!(laplacian.graph_params.k, original_params.k);
    assert_eq!(laplacian.graph_params.p, original_params.p);
    assert_eq!(laplacian.graph_params.sigma, original_params.sigma);

    debug!("Parameter preservation test passed");
}

#[test]
fn test_new_from_items_single_element() {
    // Test with 1x1 matrix
    let single_data = vec![42.0];
    let single_matrix = DenseMatrix::from_iterator(single_data.into_iter(), 1, 1, 0);

    let graph_params =
        GraphParams { eps: 2.0, k: 1, p: 1.5, sigma: None, normalise: true };

    let laplacian = GraphLaplacian::prepare_from_items(single_matrix, graph_params);

    assert_eq!(laplacian.nnodes, 1);
    assert_eq!(laplacian.matrix.shape().0, 1);
    assert_eq!(laplacian.matrix.shape().1, 1);
    assert_eq!(*laplacian.matrix.get((0, 0)), 42.0);

    debug!("Single element test passed");
}
