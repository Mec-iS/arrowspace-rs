use smartcore::linalg::basic::arrays::Array;

use crate::builder::ArrowSpaceBuilder;

#[test]
fn test_minimal_input() {
    let rows = vec![vec![1.0, 0.0, 3.0], vec![0.5, 1.0, 0.0]];
    ArrowSpaceBuilder::new().build(rows);
}

#[test]
fn simple_build() {
    // build `with_lambda_graph`
    let rows = vec![vec![1.0, 0.0, 5.0], vec![0.3, 1.0, 0.0]];

    let eps = 1.0;
    let k = 3usize;
    let p = 2.0;
    let sigma_override = None;

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(eps, k, p, sigma_override)
        .build(rows);

    assert_eq!(aspace.data.shape(), (2, 3));
    assert_eq!(gl.nnodes, 2);
}

#[test]
fn build_from_rows_with_lambda_graph() {
    let rows = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
    ];

    // Build a lambda-proximity Laplacian over items from the data matrix
    // Parameters mirror the old intent: small eps, k=2 cap, p=2.0 kernel, default sigma
    let (aspace, gl) =
        ArrowSpaceBuilder::new().with_lambda_graph(1e-3, 2, 2.0, None).build(rows);

    assert_eq!(aspace.data.shape(), (2, 13));
    assert_eq!(gl.nnodes, 2);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

#[test]
fn build_with_lambda_graph_over_product_like_rows() {
    // Test with realistic high-dimensional feature vectors instead of synthetic product coordinates
    // These represent meaningful data patterns commonly found in ML applications
    let rows = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
    ];

    let (aspace, gl) =
        ArrowSpaceBuilder::new().with_lambda_graph(1e-3, 3, 2.0, None).build(rows);

    assert_eq!(aspace.data.shape(), (2, 13));
    assert_eq!(gl.nnodes, 2);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

#[test]
fn lambda_graph_shape_matches_rows() {
    // Test that lambda-graph construction correctly handles multiple items
    // with realistic high-dimensional feature vectors
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
    let len_items = items.len();

    let (aspace, gl) =
        ArrowSpaceBuilder::new().with_lambda_graph(1e-3, 3, 2.0, None).build(items);

    assert_eq!(aspace.data.shape(), (len_items, 13));
    assert_eq!(gl.nnodes, len_items);
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}
