use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::builder::ArrowSpaceBuilder;
use crate::tests::test_data::make_gaussian_hd;

use crate::subgraphs::sg_from_centroids::recluster_centroids;
use crate::subgraphs::{CentroidGraphParams, SubgraphsCentroid};

#[test]
fn test_centroid_subgraphs_basic() {
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
        min_centroids: 3,
        max_depth: 2,
    };

    // Use the new API to get all subgraphs as a flat list
    let subgraphs = gl_centroids.spot_subg_centroids(&aspace, &params);

    assert!(
        !subgraphs.is_empty(),
        "Should extract at least one subgraph"
    );

    // Verify all subgraphs have correct invariants
    for sg in &subgraphs {
        let (f_sg, x_sg) = sg.laplacian.init_data.shape();

        // init_data is F × X, nnodes is X
        assert_eq!(sg.laplacian.nnodes, x_sg);
        assert_eq!(sg.node_indices.len(), x_sg);

        // matrix is F × F
        let (mf_rows, mf_cols) = sg.laplacian.matrix.shape();
        assert_eq!(mf_rows, f_sg);
        assert_eq!(mf_cols, f_sg);
    }

    debug!("Extracted {} centroid subgraphs", subgraphs.len());
}

#[test]
fn test_centroid_hierarchy_advanced() {
    crate::tests::init();

    let rows = make_gaussian_hd(80, 0.4);

    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_sparsity_check(false)
        .with_seed(42);

    let (aspace, gl_centroids) = builder.build(rows);

    let params = CentroidGraphParams {
        k: 4,
        topk: 4,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(123),
        min_centroids: 3,
        max_depth: 2,
    };

    // Use the new API to get the full hierarchy
    let hierarchy = gl_centroids.build_centroid_hierarchy(&aspace, params);

    let root = &hierarchy.root;
    let root_gl = &root.graph.laplacian;

    let (f0, x0) = gl_centroids.init_data.shape();

    // Root nnodes is number of centroids X0.
    assert_eq!(root_gl.nnodes, x0);
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

    if root_gl.nnodes >= hierarchy.root.graph.laplacian.nnodes {
        let level1 = hierarchy.level(1);
        if !level1.is_empty() {
            for node in level1 {
                let gl = &node.graph.laplacian;
                let (f_l, x_l) = gl.init_data.shape();

                assert!(x_l > 0);
                assert!(f_l > 0);
                assert_eq!(gl.nnodes, x_l);

                let (lm0, lm1) = gl.matrix.shape();
                assert_eq!(lm0, f_l);
                assert_eq!(lm1, f_l);

                assert!(!node.parent_map.is_empty());
            }
        }
    }

    let count_by_levels: usize = hierarchy.levels.iter().map(|v| v.len()).sum();
    assert_eq!(hierarchy.count_subgraphs(), count_by_levels);
    assert!(hierarchy.count_subgraphs() >= 1);
}

#[test]
fn test_centroid_subgraphs_min_centroids_cutoff() {
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

    // With min_centroids > root size, should only get root level
    let subgraphs = gl_centroids.spot_subg_centroids(&aspace, &params);

    // Should have exactly 1 subgraph (the root)
    assert_eq!(
        subgraphs.len(),
        1,
        "Should only extract root when min_centroids > root size"
    );

    // Verify via hierarchy API too
    let hierarchy = gl_centroids.build_centroid_hierarchy(&aspace, params);
    assert_eq!(hierarchy.count_subgraphs(), 1);

    for depth in 1..hierarchy.levels.len() {
        assert!(
            hierarchy.levels[depth].is_empty(),
            "expected no nodes at depth {} when min_centroids > root size",
            depth
        );
    }
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
    let centroids_xf = DenseMatrix::from_2d_vec(&data).unwrap();
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
fn test_centroid_subgraphs_two_levels() {
    crate::tests::init();

    let rows = make_gaussian_hd(120, 0.3);
    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_sparsity_check(false)
        .with_seed(99);

    let (aspace, gl_centroids) = builder.build(rows);

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

    let hierarchy = gl_centroids.build_centroid_hierarchy(&aspace, params);

    // Expect a second level
    let level1 = hierarchy.level(1);
    assert!(
        !level1.is_empty(),
        "expected non-empty level 1 for nested hierarchy"
    );

    // Verify all subgraphs have valid root_indices
    let all_subgraphs = hierarchy.all_subgraphs();
    for sg in &all_subgraphs {
        let (f_sg, x_sg) = sg.laplacian.init_data.shape();
        assert_eq!(sg.laplacian.nnodes, x_sg);
        assert!(f_sg > 0);
    }

    debug!(
        "Built hierarchy with {} levels, {} total subgraphs",
        hierarchy.levels.len(),
        all_subgraphs.len()
    );
}

#[test]
fn test_centroid_subgraphs_three_levels() {
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

    let hierarchy = gl_centroids.build_centroid_hierarchy(&aspace, params);

    assert!(
        !hierarchy.level(0).is_empty(),
        "level 0 (root) must be non-empty"
    );
    assert!(
        !hierarchy.level(1).is_empty(),
        "level 1 should be non-empty for this configuration"
    );

    // Check invariants across all subgraphs
    let all_subgraphs = hierarchy.all_subgraphs();
    for (i, sg) in all_subgraphs.iter().enumerate() {
        let (f_sg, x_sg) = sg.laplacian.init_data.shape();

        assert!(x_sg > 0, "subgraph {} must have at least one centroid", i);
        assert!(f_sg > 0);

        assert_eq!(
            sg.laplacian.nnodes, x_sg,
            "subgraph {} nnodes must equal centroid count X",
            i
        );

        let (mf_rows, mf_cols) = sg.laplacian.matrix.shape();
        assert_eq!(mf_rows, f_sg);
        assert_eq!(mf_cols, f_sg);
    }

    debug!(
        "Built 3-level hierarchy with {} total subgraphs",
        all_subgraphs.len()
    );
}

#[test]
fn test_centroid_subgraphs_flat_vs_hierarchy() {
    crate::tests::init();

    let rows = make_gaussian_hd(100, 0.3);
    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_sparsity_check(false)
        .with_seed(555);

    let (aspace, gl_centroids) = builder.build(rows);

    let params = CentroidGraphParams {
        k: 4,
        topk: 4,
        eps: 0.4,
        p: 2.0,
        sigma: None,
        normalise: true,
        sparsitycheck: false,
        seed: Some(999),
        min_centroids: 3,
        max_depth: 2,
    };

    // Get subgraphs via flat API
    let flat_subgraphs = gl_centroids.spot_subg_centroids(&aspace, &params);

    // Get subgraphs via hierarchy API
    let hierarchy = gl_centroids.build_centroid_hierarchy(&aspace, params);
    let hierarchy_subgraphs = hierarchy.all_subgraphs();

    // Both should return the same subgraphs
    assert_eq!(
        flat_subgraphs.len(),
        hierarchy_subgraphs.len(),
        "Flat and hierarchy APIs should return same number of subgraphs"
    );

    assert_eq!(
        flat_subgraphs.len(),
        hierarchy.count_subgraphs(),
        "Subgraph count should match hierarchy count"
    );

    debug!(
        "Both APIs returned {} subgraphs consistently",
        flat_subgraphs.len()
    );
}
