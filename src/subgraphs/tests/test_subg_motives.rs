use crate::builder::ArrowSpaceBuilder;
use crate::tests::test_data::make_gaussian_hd;

use crate::motives::MotiveConfig;
use crate::subgraphs::{Subgraph, SubgraphConfig, Subgraphs};

#[test]
fn test_spot_subgraphs_eigen_basic() {
    crate::init();

    let rows = make_gaussian_hd(80, 0.3);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.5, 6, 3, 2.0, None)
        .with_seed(42)
        .build(rows);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 12,
            min_triangles: 2,
            min_clust: 0.3,
            max_motif_size: 20,
            max_sets: 50,
            jaccard_dedup: 0.7,
        },
        rayleigh_max: Some(0.3),
        min_size: 4,
    };

    let subgraphs = gl.spot_subgraphs_eigen(cfg);

    assert!(
        !subgraphs.is_empty(),
        "Should extract at least one subgraph"
    );

    for sg in &subgraphs {
        assert!(sg.node_indices.len() >= 4, "Min size should be enforced");
        assert_eq!(
            sg.laplacian.nnodes,
            sg.node_indices.len(),
            "Local Laplacian nodes should match subgraph size"
        );
        if let Some(r) = sg.rayleigh {
            assert!(r <= 0.3, "Rayleigh filter should be applied");
        }
    }

    println!("Extracted {} subgraphs from eigen mode", subgraphs.len());
}

#[test]
fn test_subgraph_from_parent() {
    crate::init();

    let rows = make_gaussian_hd(50, 0.4);
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.5, 8, 4, 2.0, None)
        .with_seed(123)
        .build(rows);

    let nodes = vec![0, 1, 2, 5, 10];
    let sg = Subgraph::from_parent(&gl, &nodes, None);

    assert_eq!(sg.node_indices, nodes);
    assert_eq!(sg.laplacian.nnodes, nodes.len());
    assert!(sg.laplacian.matrix.nnz() > 0, "Subgraph should have edges");
}

#[test]
fn test_subgraph_rayleigh_computation() {
    crate::init();

    let rows = make_gaussian_hd(60, 0.3);
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 6, 3, 2.0, None)
        .with_seed(99)
        .build(rows);

    let nodes = vec![0, 1, 2, 3];
    let mut sg = Subgraph::from_parent(&gl, &nodes, None);

    assert!(sg.rayleigh.is_none(), "Rayleigh should be None initially");

    sg.compute_rayleigh();

    assert!(sg.rayleigh.is_some(), "Rayleigh should be computed");
    assert!(
        sg.rayleigh.unwrap().is_finite(),
        "Rayleigh should be finite"
    );
}
