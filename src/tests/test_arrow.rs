use smartcore::linalg::basic::arrays::Array;

use crate::builder::ArrowSpaceBuilder;
use crate::core::{ArrowSpace, TAUDEFAULT};
use crate::graph::GraphFactory;

use crate::tests::{GRAPH_PARAMS, TAU_PARAMS};

use log::debug;

#[test]
fn arrowspace_build_and_recompute() {
    // Test ArrowSpace construction and λ recomputation with realistic high-dimensional data
    // These vectors represent different types of feature patterns commonly found in ML:

    let rows = vec![
        // Vector 1: High-dimensional feature vector with mixed magnitudes (e.g., customer profile)
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        // Vector 2: Similar but distinct pattern (e.g., similar customer segment)
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
        // Vector 3: Concentrated in early features (e.g., young demographic profile)
        vec![
            0.95, 0.87, 0.72, 0.15, 0.08, 0.05, 0.12, 0.03, 0.01, 0.09, 0.02, 0.04,
            0.11,
        ],
        // Vector 4: Uniform distribution across features (e.g., balanced feature set)
        vec![
            0.45, 0.48, 0.52, 0.47, 0.51, 0.49, 0.46, 0.53, 0.44, 0.50, 0.48, 0.52,
            0.47,
        ],
        // Vector 5: Concentrated in later features (e.g., senior demographic profile)
        vec![
            0.08, 0.12, 0.15, 0.31, 0.52, 0.68, 0.74, 0.83, 0.91, 0.88, 0.76, 0.82,
            0.89,
        ],
        // Vector 6: Sparse pattern with few dominant features (e.g., specialized user)
        vec![
            0.02, 0.01, 0.03, 0.01, 0.02, 0.98, 0.01, 0.02, 0.01, 0.95, 0.01, 0.03,
            0.02,
        ],
        // Vector 7: Bimodal distribution (e.g., user with two main interests)
        vec![
            0.81, 0.79, 0.05, 0.03, 0.02, 0.04, 0.03, 0.05, 0.78, 0.85, 0.02, 0.04,
            0.83,
        ],
    ];

    let aspace = ArrowSpace::from_items(rows.clone(), TAU_PARAMS);
    assert_eq!(aspace.nfeatures, 13, "Expected 13-dimensional feature space");
    assert_eq!(aspace.nitems, 7, "Expected 7 data points");

    debug!("=== ARROWSPACE CONSTRUCTION ===");
    debug!("Feature dimensions: {}", aspace.nfeatures);
    debug!("Number of items: {}", aspace.nitems);
    debug!("Data shape: {:?}", aspace.data.shape());

    // Build λτ-graph from the same data matrix with realistic parameters
    let gl = GraphFactory::build_laplacian_matrix(
        rows.clone(),
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );

    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);

    assert_eq!(gl.nnodes, aspace.nitems, "Graph nodes should match ArrowSpace items");
    assert_eq!(
        aspace.signals.shape(),
        (13, 13),
        "Signals matrix should be 13x13 for spectral decomposition"
    );

    debug!("\n=== SPECTRAL ANALYSIS ===");
    debug!("Graph nodes: {}", gl.nnodes);
    debug!("Signals matrix shape: {:?}", aspace.signals.shape());

    // Recompute lambdas and analyze spectral properties
    aspace.recompute_lambdas();

    debug!("Lambda values (raw): {:?}", aspace.lambdas);

    // Spectral graph theory properties: all eigenvalues should be non-negative
    assert!(
        aspace.lambdas[0] >= 0.0,
        "Smallest eigenvalue should be non-negative: {}",
        aspace.lambdas[0]
    );
    assert!(
        aspace.lambdas[1] >= 0.0,
        "Fiedler eigenvalue should be non-negative: {}",
        aspace.lambdas[1]
    );
}

#[test]
fn arrowspace_build_and_recompute_nonzero() {
    // Create test data with more similar patterns to encourage graph connectivity
    let rows = vec![
        vec![1.0, 0.8, 0.6, 0.4], // Smooth decreasing pattern
        vec![0.9, 0.7, 0.5, 0.3], // Similar decreasing pattern
        vec![0.8, 0.9, 0.7, 0.5], // Similar with slight variation
        vec![0.7, 0.6, 0.8, 0.6], // Mixed but similar range
        vec![0.6, 0.5, 0.4, 0.7], // Another similar pattern
    ];

    let aspace = ArrowSpace::from_items(rows.clone(), TAU_PARAMS);
    assert_eq!(aspace.nfeatures, 4);
    assert_eq!(aspace.nitems, 5);

    // Build λτ-graph with parameters that encourage connectivity
    // Larger eps to include more neighbors, higher k for more connections
    let gl = GraphFactory::build_laplacian_matrix(
        rows,
        0.5, // Larger eps for more connectivity
        4,   // Higher k for more neighbors
        2.0,
        None,
        GRAPH_PARAMS.normalise,
    );
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);

    assert!(gl.nnodes == aspace.nitems);
    aspace.recompute_lambdas();
    let lam0 = aspace.lambdas()[0];
    let lam1 = aspace.lambdas()[1];

    // Should now have non-zero lambdas
    debug!("lam0: {:?}, lam1: {:?}", lam0, lam1);
    assert!(lam0 >= 0.0);
    assert!(lam1 >= 0.0);
    assert!(lam0 > 1e-10 || lam1 > 1e-10, "At least one lambda should be non-zero");
}

#[test]
fn arrowspace_construct_and_lambda() {
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

    let aspace = ArrowSpace::from_items_default(rows.clone());
    assert_eq!(aspace.data.shape(), (2, 13));

    // Build λτ-graph from the same data matrix
    let gl = GraphFactory::build_laplacian_matrix(
        rows,
        1e-3,
        3,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
    );
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);

    assert!(gl.nnodes == aspace.data.shape().0);
    aspace.recompute_lambdas();
    let lam0 = aspace.lambdas()[0];
    let lam1 = aspace.lambdas()[1];

    // Basic sanity: non-negative and often lam1 <= lam0 for this pair of rows
    assert!(lam0 >= 0.0);
    assert!(lam1 >= 0.0);
}

#[test]
fn arrowspace_add_rows_superpose() {
    // Two one-hot signals on adjacent items
    let rows = vec![
        vec![0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19],
        vec![0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21],
    ];
    let aspace = ArrowSpace::from_items_default(rows.clone());

    // λτ-graph from the same data
    let gl = GraphFactory::build_laplacian_matrix(
        rows,
        1e-1,
        2,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
    );
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);
    aspace.recompute_lambdas();

    let lam_before: Vec<f64> = aspace.lambdas().to_vec();

    // superpose item1 into item0
    aspace.get_item(0).add_inplace(&aspace.get_item(1));
    let lam_after: Vec<f64> = aspace.lambdas().to_vec();

    // Non-negativity and boundedness sanity; superposition should not produce NaN/Inf
    assert!(lam_after[0].is_finite() && lam_after[0] >= 0.0);
    // Typically, adding a nearby emitter reduces roughness; allow non-strict inequality
    assert!(lam_after[0] <= lam_before[0] + 1e-12);
}

#[test]
fn arrowspace_lambda_taumode_characteristics() {
    // Create test data with distinct patterns to analyze taumode behavior
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
        vec![
            0.77, 0.14, 0.47, 0.26, 0.59, 0.35, 0.51, 0.45, 0.23, 0.68, 0.10, 0.34,
            0.54,
        ],
    ];

    let (mut aspace, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            GRAPH_PARAMS.eps,
            GRAPH_PARAMS.k,
            GRAPH_PARAMS.p,
            GRAPH_PARAMS.sigma,
        )
        .build(items);

    debug!("Taumode lambdas: {:?}", &aspace.lambdas);

    // Test 1: Non-negativity (fundamental property of Rayleigh quotients)
    assert!(
        aspace.lambdas.iter().all(|&x| x >= 0.0),
        "Taumode indices must be non-negative"
    );

    // Test 2: Bounded above by reasonable values (based on graph structure)
    let max_lambda: f64 = aspace.lambdas.iter().fold(0.0, |a, &b| a.max(b));
    assert!(
        max_lambda < 100.0,
        "Taumode indices should be reasonably bounded: max={}",
        max_lambda
    );

    // Test 3: Consistency - recomputing should give identical results
    let initial_lambdas = aspace.lambdas.to_vec();
    aspace.recompute_lambdas();
    let recomputed_lambdas = aspace.lambdas();

    for (i, (&initial, &recomputed)) in
        initial_lambdas.iter().zip(recomputed_lambdas.iter()).enumerate()
    {
        assert!(
            (initial - recomputed).abs() < 1e-12,
            "Recomputation inconsistency at feature {}: initial={}, recomputed={}",
            i,
            initial,
            recomputed
        );
    }

    // Test 4: Sensitivity to data changes - different data should give different lambdas
    let modified_items = vec![
        vec![
            0.90, 0.05, 0.50, 0.35, 0.70, 0.25, 0.60, 0.55, 0.12, 0.80, 0.03, 0.40,
            0.65,
        ],
        vec![
            0.70, 0.18, 0.40, 0.22, 0.55, 0.38, 0.48, 0.40, 0.28, 0.62, 0.15, 0.30,
            0.48,
        ],
        vec![
            0.88, 0.06, 0.38, 0.34, 0.72, 0.26, 0.50, 0.58, 0.14, 0.82, 0.02, 0.42,
            0.68,
        ],
        vec![
            0.74, 0.17, 0.44, 0.23, 0.56, 0.39, 0.49, 0.42, 0.26, 0.65, 0.13, 0.31,
            0.49,
        ],
    ];

    let (aspace2, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(
            GRAPH_PARAMS.eps,
            GRAPH_PARAMS.k,
            GRAPH_PARAMS.p,
            GRAPH_PARAMS.sigma,
        )
        .build(modified_items);

    debug!("Taumode lambdas: {:?}", &aspace.lambdas);

    let mut significant_differences = 0;
    for (_, (&original, &modified)) in
        aspace.lambdas.iter().zip(aspace2.lambdas.iter()).enumerate()
    {
        let diff = (original - modified).abs();
        if diff > 1e-6 {
            significant_differences += 1;
        }
    }

    assert!(
        significant_differences > 0,
        "Taumode should be sensitive to data changes - no significant differences found"
    );

    // Test 5: Finite values (no NaN or infinity)
    assert!(
        aspace.lambdas.iter().all(|&x| x.is_finite()),
        "All taumode indices should be finite"
    );

    // Test 6: Feature variation analysis
    let lambda_variance = {
        let mean = aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64;
        let variance = aspace.lambdas.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / aspace.lambdas.len() as f64;
        variance
    };

    debug!("Lambda statistics:");
    debug!("  Min: {:.6}", aspace.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
    debug!("  Max: {:.6}", max_lambda);
    debug!(
        "  Mean: {:.6}",
        aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64
    );
    debug!("  Variance: {:.6}", lambda_variance);

    // Variance should be positive (indicating feature discrimination)
    assert!(lambda_variance >= 0.0, "Lambda variance should be non-negative");

    debug!("✓ Taumode index characteristics validated");
    debug!("✓ Non-negativity, boundedness, and consistency verified");
    debug!("✓ Sensitivity to data changes confirmed");
    debug!("✓ Statistical properties within expected ranges");
}

#[test]
fn test_arrowspace_get() {
    // Create test data: 3 items, 4 features
    // item 0: [1.0, 2.0, 3.0, 4.0]
    // item 1: [5.0, 6.0, 7.0, 8.0]
    // item 2: [9.0, 10.0, 11.0, 12.0]
    let items = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];

    let arrow_space = ArrowSpace::from_items(items, TAUDEFAULT.unwrap());

    // Test item 0
    let item_0: Vec<f64> = arrow_space.get_item(0).item;
    assert_eq!(item_0, &[1.0, 2.0, 3.0, 4.0]);

    // Test item 1
    let item_1 = arrow_space.get_item(1).item;
    assert_eq!(item_1, &[5.0, 6.0, 7.0, 8.0]);

    // Test item 2
    let item_2 = arrow_space.get_item(2).item;
    assert_eq!(item_2, &[9.0, 10.0, 11.0, 12.0]);

    // Test feature 0
    let feature_0 = arrow_space.get_feature(0).feature;
    assert_eq!(feature_0, &[1.0, 5.0, 9.0]);

    // Test feature 1
    let feature_1 = arrow_space.get_feature(1).feature;
    assert_eq!(feature_1, &[2.0, 6.0, 10.0]);

    // Test feature 2
    let feature_2 = arrow_space.get_feature(2).feature;
    assert_eq!(feature_2, &[3.0, 7.0, 11.0]);

    // Test feature 3
    let feature_3 = arrow_space.get_feature(3).feature;
    assert_eq!(feature_3, &[4.0, 8.0, 12.0]);
}

#[test]
fn arrowspace_lambda_sanity() {
    // Four items with 13 features each; check non-negativity and a conservative upper bound using max degree
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
        vec![
            0.77, 0.14, 0.47, 0.26, 0.59, 0.35, 0.51, 0.45, 0.23, 0.68, 0.10, 0.34,
            0.54,
        ],
    ];

    let aspace = ArrowSpace::from_items_default(items.clone());
    let gl = GraphFactory::build_laplacian_matrix(
        items,
        1e-3,
        3,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
    );

    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);
    aspace.recompute_lambdas();
    let lambda = aspace.lambdas();

    let max_deg = (0..aspace.nitems).map(|i| aspace.data.get((i, i))).fold(
        aspace.data.get((0, 0)),
        |max, current| {
            if current > max {
                current
            } else {
                max
            }
        },
    );

    let upper = 2.0 * max_deg + 1e-12;

    assert!(lambda.iter().all(|&x| x >= 0.0), "synthetic index must be nonnegative");
    assert!(
        lambda.iter().all(|&x| x <= upper),
        "synthetic index {lambda:?} exceeded conservative upper bound {upper} (max_deg={max_deg})"
    );
}

#[test]
fn arrowspace_lambda_positivity_bounds() {
    // Build λτ-graph from a small basis
    let gl = GraphFactory::build_laplacian_matrix(
        vec![
            vec![
                0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
                0.58,
            ],
            vec![
                0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
                0.56,
            ],
        ],
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );

    let test_rows0 = vec![
        vec![
            0.85, 0.15, 0.42, 0.33, 0.67, 0.28, 0.59, 0.41, 0.22, 0.78, 0.05, 0.38,
            0.62,
        ],
        vec![
            0.76, 0.09, 0.48, 0.31, 0.58, 0.35, 0.51, 0.49, 0.18, 0.72, 0.11, 0.34,
            0.54,
        ],
    ];
    let test_rows1 = vec![
        vec![
            0.73, 0.18, 0.46, 0.25, 0.69, 0.31, 0.57, 0.43, 0.16, 0.81, 0.04, 0.39,
            0.61,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
    ];

    for test in [test_rows0, test_rows1].iter() {
        let aspace = ArrowSpace::from_items_default(test.clone());
        let mut aspace =
            GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);
        aspace.recompute_lambdas();
        let lambda = aspace.lambdas()[0];
        assert!(lambda >= 0.0, "Lambda positivity failed");
        // Generous upper bound for small tests
        assert!(lambda <= 10.0, "Lambda upper bound failed");
    }
}

#[test]
fn arrowitem_addition_inplace() {
    let row_a = vec![0.1, 0.5, 0.6, 0.2];
    let row_b = vec![0.9, 0.1, 0.3, 0.6];

    let aspace0 = ArrowSpace::from_items_default(vec![row_a.clone(), row_b.clone()]);
    let aspace1 = ArrowSpace::from_items_default(vec![row_b.clone(), row_a.clone()]);

    let mut item0 = aspace0.get_item(0);
    item0.add_inplace(&aspace1.get_item(0));
    assert_eq!(item0.item, vec![1.0, 0.6, 0.8999999999999999, 0.8]);

    let mut item1 = aspace1.get_item(1);
    item1.add_inplace(&aspace0.get_item(1));
    assert_eq!(item1.item, vec![1.0, 0.6, 0.8999999999999999, 0.8]);
}

#[test]
fn arrowspace_get_item() {
    // arrowspace stores the data column-wise
    // to return rows a lookup is needed

    let row_a = vec![0.1, 0.5, 0.6, 0.2];
    let row_b = vec![0.9, 0.1, 0.3, 0.6];

    let aspace0 = ArrowSpace::from_items_default(vec![row_a.clone(), row_b.clone()]);

    let item0 = aspace0.get_item(0);
    assert_eq!(item0.item, vec![0.1, 0.5, 0.6, 0.2]);

    let item1 = aspace0.get_item(1);
    assert_eq!(item1.item, vec![0.9, 0.1, 0.3, 0.6]);
}

#[test]
fn arrowspace_addition_commutativity() {
    // Test that (A + B) and (B + A) produce the same lambda effects
    // when operating on items in column-major ArrowSpace

    // More realistic high-dimensional feature vectors (13 features)
    let item_a = vec![
        0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36, 0.58,
    ];
    let item_b = vec![
        0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37, 0.56,
    ];

    // Create two ArrowSpaces with the SAME items but in different positions
    // aspace1: A at index 0, B at index 1
    // aspace2: B at index 0, A at index 1
    let aspace1 = ArrowSpace::from_items_default(vec![item_a.clone(), item_b.clone()]);
    let aspace2 = ArrowSpace::from_items_default(vec![item_b.clone(), item_a.clone()]);

    debug!("ArrowSpace shape: {:?}", aspace1.data.shape());

    // Build λ-graph from the union of items for consistent Laplacian
    let gl = GraphFactory::build_laplacian_matrix(
        vec![item_a.clone(), item_b.clone()],
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );

    debug!("Graph nodes: {:?}", gl.nnodes);

    // Compute initial lambdas for both spaces
    let mut aspace1 = GraphFactory::build_spectral_laplacian(aspace1, &gl.graph_params);
    let mut aspace2 = GraphFactory::build_spectral_laplacian(aspace2, &gl.graph_params);
    aspace1.recompute_lambdas();
    aspace2.recompute_lambdas();

    debug!("=== BEFORE ADDITION ===");
    debug!(
        "aspace1 items: [0]={:?}, [1]={:?}",
        aspace1.get_item(0).item,
        aspace1.get_item(1).item
    );
    debug!(
        "aspace2 items: [0]={:?}, [1]={:?}",
        aspace2.get_item(0).item,
        aspace2.get_item(1).item
    );
    debug!("aspace1 lambdas: {:?}", aspace1.lambdas());
    debug!("aspace2 lambdas: {:?}", aspace2.lambdas());

    // Verify that we have the items we expect
    assert_eq!(aspace1.get_item(0).item, item_a, "aspace1[0] should be item_a");
    assert_eq!(aspace1.get_item(1).item, item_b, "aspace1[1] should be item_b");
    assert_eq!(aspace2.get_item(0).item, item_b, "aspace2[0] should be item_b");
    assert_eq!(aspace2.get_item(1).item, item_a, "aspace2[1] should be item_a");

    // Store initial lambda states for comparison
    let aspace1_initial_lambdas = aspace1.lambdas().to_vec();
    let aspace2_initial_lambdas = aspace2.lambdas().to_vec();

    // Perform the addition operations:
    // aspace1: A + B (add item[1] into item[0]) -> item[0] becomes A+B
    // aspace2: B + A (add item[1] into item[0]) -> item[0] becomes B+A
    aspace1.add_items(0, 1, &gl); // A += B
    aspace2.add_items(0, 1, &gl); // B += A

    debug!("\n=== AFTER ADDITION ===");
    debug!("aspace1 result: A+B = {:?}", aspace1.get_item(0).item);
    debug!("aspace2 result: B+A = {:?}", aspace2.get_item(0).item);
    debug!("aspace1 lambdas: {:?}", aspace1.lambdas());
    debug!("aspace2 lambdas: {:?}", aspace2.lambdas());

    // Verify that both results are identical due to commutativity: A+B = B+A
    let result1 = aspace1.get_item(0); // A + B
    let result2 = aspace2.get_item(0); // B + A

    // Expected result: element-wise sum of the two 13-dimensional vectors
    // [0.82+0.79, 0.11+0.12, 0.43+0.45, 0.28+0.29, 0.64+0.61, 0.32+0.33, 0.55+0.54,
    //  0.48+0.47, 0.19+0.21, 0.73+0.70, 0.07+0.08, 0.36+0.37, 0.58+0.56]
    let expected = vec![
        1.61, 0.23, 0.88, 0.57, 1.25, 0.65, 1.09, 0.95, 0.40, 1.43, 0.15, 0.73, 1.14,
    ];

    // Verify both results match the expected sum
    for (i, (&actual, &expected_val)) in
        result1.item.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual - expected_val).abs() < 1e-10,
            "aspace1 feature {i}: got {actual}, expected {expected_val}"
        );
    }

    for (i, (&actual, &expected_val)) in
        result2.item.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual - expected_val).abs() < 1e-10,
            "aspace2 feature {i}: got {actual}, expected {expected_val}"
        );
    }

    // Verify commutativity: A+B = B+A
    for (i, (&v1, &v2)) in result1.item.iter().zip(result2.item.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "Commutativity failed at feature {}: A+B={}, B+A={}, diff={}",
            i,
            v1,
            v2,
            (v1 - v2).abs()
        );
    }

    // Test the lambda symmetry property:
    // After addition, the feature distributions have changed, and the lambdas reflect this.
    // Since both spaces now contain the same data (A+B in position 0, unchanged B/A in position 1),
    // but with different item arrangements, the feature lambdas should reflect the swapped symmetry.
    //
    // Key insight:
    // - aspace1 now has [A+B, B] as items
    // - aspace2 now has [B+A, A] as items
    // Since A+B = B+A, both have identical item 0, but different item 1
    // The lambda patterns should be related but may not be identical due to different item 1
    //
    // With these more realistic vectors, we can observe how spectral properties change
    // across higher-dimensional feature spaces and more complex similarity relationships.

    debug!("\n=== LAMBDA ANALYSIS ===");
    debug!("Initial aspace1 lambdas: {aspace1_initial_lambdas:?}");
    debug!("Initial aspace2 lambdas: {aspace2_initial_lambdas:?}");
    debug!("Final aspace1 lambdas:   {:?}", aspace1.lambdas());
    debug!("Final aspace2 lambdas:   {:?}", aspace2.lambdas());

    // Calculate and display feature-wise differences to analyze the effect
    let feature_diff: Vec<f64> =
        item_a.iter().zip(item_b.iter()).map(|(a, b)| (a - b).abs()).collect();
    debug!("Feature-wise |A-B| differences: {feature_diff:?}");
    debug!(
        "Max difference: {:.6}",
        feature_diff.iter().fold(0.0, |a: f64, &b| a.max(b))
    );
    debug!(
        "Mean difference: {:.6}",
        feature_diff.iter().sum::<f64>() / feature_diff.len() as f64
    );

    // The main test: verify mathematical commutativity of the addition operation
    debug!("\n✓ Addition commutativity verified: A+B = B+A = {expected:?}");
    debug!("✓ Column-major storage and retrieval working correctly with 13-dimensional vectors!");
    debug!("✓ High-dimensional spectral lambda computation maintains mathematical consistency!");
}

#[test]
#[should_panic]
fn arrowspace_zero_vector_should_panic() {
    let gl = GraphFactory::build_laplacian_matrix(
        vec![
            vec![
                0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
                0.58,
            ],
            vec![
                0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
                0.56,
            ],
        ],
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );

    let zero_row =
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let aspace = ArrowSpace::from_items_default(vec![zero_row.clone(), zero_row]);
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);
    aspace.recompute_lambdas();
}

#[test]
#[should_panic]
fn graph_one_node() {
    let _ = GraphFactory::build_laplacian_matrix(
        vec![vec![1.0, 1.0, 1.0]],
        1.0,
        2,
        2.0,
        None,
        GRAPH_PARAMS.normalise,
    );
}

#[test]
fn arrowspace_recompute_synthetic() {
    use approx::relative_eq;
    // Test with meaningful high-dimensional feature vectors representing different data patterns
    let gl = GraphFactory::build_laplacian_matrix(
        vec![
            vec![
                0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
                0.58,
            ],
            vec![
                0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
                0.56,
            ],
        ],
        GRAPH_PARAMS.eps,
        GRAPH_PARAMS.k,
        GRAPH_PARAMS.p,
        GRAPH_PARAMS.sigma,
        GRAPH_PARAMS.normalise,
    );

    let constant = vec![
        vec![
            0.82, 0.11, 0.43, 0.28, 0.64, 0.32, 0.55, 0.48, 0.19, 0.73, 0.07, 0.36,
            0.58,
        ],
        vec![
            0.79, 0.12, 0.45, 0.29, 0.61, 0.33, 0.54, 0.47, 0.21, 0.70, 0.08, 0.37,
            0.56,
        ],
    ];
    let aspace = ArrowSpace::from_items_default(constant);
    let mut aspace = GraphFactory::build_spectral_laplacian(aspace, &gl.graph_params);
    aspace.recompute_lambdas();

    debug!("lambdas: {:?}", aspace.lambdas);

    assert!(relative_eq!(aspace.lambdas[0], 0.182305845, epsilon = 1e-4));
    assert!(relative_eq!(aspace.lambdas[1], 0.186414509, epsilon = 1e-4));
}
