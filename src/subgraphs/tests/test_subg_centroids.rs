use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::builder::ArrowSpaceBuilder;
use crate::tests::test_data::make_gaussian_hd;

use crate::subgraphs::sg_from_centroids::{
    CentroidGraphParams, CentroidHierarchy, CentroidNode, recluster_centroids,
};

#[test]
fn test_centroid_hierarchy_basic() {
    crate::tests::init();

    let rows = make_gaussian_hd(80, 0.4);
    assert_eq!(rows.len(), 80);

    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_sparsity_check(false)
        .with_seed(42);

    let (aspace, gl_centroids) = builder.build(rows.clone());

    // At level-0, init_data is F × X0, matrix is F × F, nnodes = X0.
    assert!(gl_centroids.nnodes > 0);
    let (f0, x0) = gl_centroids.init_data.shape();
    assert_eq!(x0, 3, "init_data is FxX so X should be 3 (no. of clusters)");
    let (mf0, mf1) = gl_centroids.matrix.shape();
    assert_eq!(mf0, f0);
    assert_eq!(mf1, f0);

    let params = CentroidGraphParams {
        k: 4,
        topk: 4,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(123),
        min_centroids: 3, // allow second level
        max_depth: 2,
    };

    let hierarchy = CentroidHierarchy::from_centroid_graph(&aspace, &gl_centroids, &params);

    let root = &hierarchy.root;
    let root_gl = &root.graph.laplacian;

    // Root nnodes is number of centroids X0.
    assert_eq!(root_gl.nnodes, x0);
    // node_indices live in centroid space [0..X0).
    assert_eq!(root.graph.node_indices.len(), x0);

    // init_data is F × X0 and matrix is F × F.
    let (rf, rx) = root_gl.init_data.shape();
    assert_eq!(rf, f0);
    assert_eq!(rx, x0);
    let (rm0, rm1) = root_gl.matrix.shape();
    assert_eq!(rm0, f0);
    assert_eq!(rm1, f0);

    // parent_map is identity over centroids.
    assert_eq!(root.parent_map.len(), x0);

    assert!(!hierarchy.levels.is_empty());
    assert!(!hierarchy.levels[0].is_empty());

    let level0 = &hierarchy.levels[0];
    assert_eq!(level0.len(), 1);
    let level0_node: &CentroidNode = &level0[0];

    assert_eq!(level0_node.graph.laplacian.nnodes, root_gl.nnodes);
    assert_eq!(
        level0_node.graph.laplacian.init_data.shape(),
        root_gl.init_data.shape()
    );

    if root_gl.nnodes >= params.min_centroids && params.max_depth > 1 {
        let level1 = hierarchy.level(1);
        assert!(
            !level1.is_empty(),
            "expected at least one centroid node at depth 1"
        );

        for node in level1 {
            let gl = &node.graph.laplacian;
            let (f_l, x_l) = gl.init_data.shape();

            assert!(x_l > 0);
            assert!(f_l > 0);

            // nnodes is number of centroids at this level.
            assert_eq!(gl.nnodes, x_l);

            // Feature Laplacian must be F_l × F_l.
            let (lm0, lm1) = gl.matrix.shape();
            assert_eq!(lm0, f_l);
            assert_eq!(lm1, f_l);

            assert!(
                !node.parent_map.is_empty(),
                "non-root levels must have a non-empty parent_map"
            );
        }
    }

    let count_by_levels: usize = hierarchy.levels.iter().map(|v| v.len()).sum();
    assert_eq!(hierarchy.count_subgraphs(), count_by_levels);
    assert!(hierarchy.count_subgraphs() >= 1);
}

#[test]
fn test_centroid_hierarchy_min_centroids_cutoff() {
    crate::tests::init();

    let rows = make_gaussian_hd(10, 0.5);

    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 6, 4, 2.0, None)
        .with_sparsity_check(false)
        .with_seed(7);

    let (aspace, gl_centroids) = builder.build(rows);

    let x0 = gl_centroids.nnodes;

    let params = CentroidGraphParams {
        k: 4,
        topk: 4,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(1),
        min_centroids: x0 + 1,
        max_depth: 3,
    };

    let hierarchy = CentroidHierarchy::from_centroid_graph(&aspace, &gl_centroids, &params);

    assert!(!hierarchy.levels.is_empty());
    assert_eq!(hierarchy.levels[0].len(), 1);

    for depth in 1..hierarchy.levels.len() {
        assert!(
            hierarchy.levels[depth].is_empty(),
            "expected no nodes at depth {} when min_centroids > root size",
            depth
        );
    }

    assert_eq!(hierarchy.count_subgraphs(), 1);
}

#[test]
fn test_recluster_centroids_properties() {
    crate::tests::init();

    let data = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];
    let centroids_xf = DenseMatrix::from_2d_vec(&data).unwrap(); // X × F
    let k = 3;

    let (labels, new_centroids_xf) = recluster_centroids(&centroids_xf, k, None);

    assert_eq!(labels.len(), centroids_xf.shape().0);

    let (k_eff, d) = new_centroids_xf.shape();
    assert_eq!(k_eff, k.min(centroids_xf.shape().0));
    assert_eq!(d, centroids_xf.shape().1);

    for &cid in &labels {
        assert!(cid < k_eff, "label {} out of range 0..{}", cid, k_eff);
    }
}

#[test]
fn test_centroid_hierarchy_two_levels_root_indices() {
    crate::tests::init();

    let rows = make_gaussian_hd(120, 0.3);
    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_sparsity_check(false)
        .with_seed(99);

    let (aspace, gl_centroids) = builder.build(rows.clone());

    let params = CentroidGraphParams {
        k: 4,
        topk: 4,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(1234),
        min_centroids: 3,
        max_depth: 2,
    };

    let hierarchy = CentroidHierarchy::from_centroid_graph(&aspace, &gl_centroids, &params);

    // Expect a second level.
    let level1 = hierarchy.level(1);
    assert!(
        !level1.is_empty(),
        "expected non-empty level 1 for nested hierarchy"
    );

    // Root indices at level 0 should partition (with possible overlaps) the dataset.
    let root = &hierarchy.root;
    let total_root_items: usize = root.root_indices.iter().map(|v| v.len()).sum();
    assert!(
        total_root_items >= aspace.nitems,
        "root_indices should cover at least all items (may overlap): {}, {}",
        total_root_items,
        aspace.nitems
    );

    // For each child node in level 1, all its root_indices must be a subset of root's.
    for node in level1 {
        let child_items: usize = node.root_indices.iter().map(|v| v.len()).sum();
        assert!(child_items > 0);

        for item_list in &node.root_indices {
            for &item_idx in item_list {
                assert!(
                    item_idx < aspace.nitems,
                    "item index {} should be < nitems",
                    item_idx
                );
            }
        }
    }
}

#[test]
fn test_centroid_hierarchy_three_levels_structure() {
    crate::tests::init();

    let rows = make_gaussian_hd(200, 0.25);
    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 12, 8, 2.0, None)
        .with_sparsity_check(false)
        .with_seed(123);

    let (aspace, gl_centroids) = builder.build(rows);

    let params = CentroidGraphParams {
        k: 3,
        topk: 3,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(5),
        min_centroids: 3,
        max_depth: 3,
    };

    let hierarchy = CentroidHierarchy::from_centroid_graph(&aspace, &gl_centroids, &params);

    // We expect up to 3 levels (0,1,2). Some levels may be empty depending on data,
    // but the first two should be non-empty given min_centroids and dataset size.
    assert!(
        !hierarchy.level(0).is_empty(),
        "level 0 (root) must be non-empty"
    );
    assert!(
        !hierarchy.level(1).is_empty(),
        "level 1 should be non-empty for this configuration"
    );

    // Check invariants across all populated levels.
    for (depth, level) in hierarchy.levels.iter().enumerate() {
        for node in level {
            let gl = &node.graph.laplacian;
            let (f_l, x_l) = gl.init_data.shape();

            assert!(
                x_l > 0,
                "level {} node must have at least one centroid",
                depth
            );
            assert!(f_l > 0);

            // nnodes is X_l: number of centroids at this level.
            assert_eq!(
                gl.nnodes, x_l,
                "level {} nnodes must equal centroid count X_l",
                depth
            );

            let (lm0, lm1) = gl.matrix.shape();
            assert_eq!(
                lm0, f_l,
                "level {} feature Laplacian rows must equal feature dim",
                depth
            );
            assert_eq!(
                lm1, f_l,
                "level {} feature Laplacian cols must equal feature dim",
                depth
            );

            // root_indices consistency
            for (cid, items) in node.root_indices.iter().enumerate() {
                for &item_idx in items {
                    assert!(
                        item_idx < aspace.nitems,
                        "item index {} at depth {} centroid {} out of range",
                        item_idx,
                        depth,
                        cid
                    );
                }
            }
        }
    }
}
