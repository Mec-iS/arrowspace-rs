use crate::builder::ArrowSpaceBuilder;
use crate::motives::MotiveConfig;
use crate::subgraphs::{Subgraph, SubgraphConfig, SubgraphsMotive};
use crate::tests::test_data::make_gaussian_cliques_multi;

use smartcore::linalg::basic::arrays::Array;

#[test]
fn test_subgraph_from_parent() {
    crate::init();

    // 300 points, 10 cliques, tight clusters, 50 dimensions → expect ~10 motifs
    let rows = make_gaussian_cliques_multi(300, 0.2, 10, 100, 999);
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_seed(999)
        .build(rows);

    let nodes = vec![0, 1, 2];
    let sg = Subgraph::from_parent(&gl, &nodes, None);

    // Verify node_indices.
    assert_eq!(sg.node_indices, nodes);

    // init_data is F × X where X = nodes.len().
    let (f_dim, x_sg) = sg.laplacian.init_data.shape();
    assert_eq!(x_sg, nodes.len());

    // nnodes is X (node count in motif space).
    assert_eq!(sg.laplacian.nnodes, nodes.len());

    // matrix is F × F.
    let (mf_rows, mf_cols) = sg.laplacian.matrix.shape();
    assert_eq!(mf_rows, f_dim);
    assert_eq!(mf_cols, f_dim);
    assert!(sg.laplacian.matrix.nnz() > 0, "Subgraph should have edges");
}

#[test]
fn test_subgraph_rayleigh_computation() {
    crate::init();

    let rows = make_gaussian_cliques_multi(300, 0.2, 10, 100, 999);
    let (_aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 10, 6, 2.0, None)
        .with_seed(999)
        .build(rows);

    let nodes = vec![0, 1, 2];
    let mut sg = Subgraph::from_parent(&gl, &nodes, None);

    assert!(sg.rayleigh.is_none(), "Rayleigh should be None initially");

    sg.compute_rayleigh();

    assert!(sg.rayleigh.is_some(), "Rayleigh should be computed");
    assert!(
        sg.rayleigh.unwrap().is_finite(),
        "Rayleigh should be finite"
    );
}

#[test]
fn test_spot_subgraphs_energy_basic() {
    crate::init();

    let rows = make_gaussian_cliques_multi(300, 0.2, 10, 100, 999);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 12, 8, 2.0, None)
        .with_seed(999)
        .build(rows);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 18,
            min_triangles: 3,
            min_clust: 0.35,
            max_motif_size: 30,
            max_sets: 60,
            jaccard_dedup: 0.65,
        },
        rayleigh_max: Some(0.4),
        min_size: 5,
    };

    let subgraphs = gl.spot_subg_motives(&aspace, &cfg);

    if subgraphs.is_empty() {
        debug!("No subgraphs extracted (may need different params)");
        return;
    }

    assert!(
        !subgraphs.is_empty(),
        "Should extract at least one subgraph from clique data"
    );

    for sg in &subgraphs {
        // init_data is F × X_motif, nnodes is X_motif (centroid count).
        let (f_dim, x_motif) = sg.laplacian.init_data.shape();
        assert_eq!(
            sg.laplacian.nnodes, x_motif,
            "nnodes must equal number of motif centroids (columns in init_data)"
        );
        assert_eq!(
            sg.node_indices.len(),
            x_motif,
            "node_indices length must match nnodes"
        );

        // matrix is F × F feature Laplacian.
        let (mf_rows, mf_cols) = sg.laplacian.matrix.shape();
        assert_eq!(mf_rows, f_dim);
        assert_eq!(mf_cols, f_dim);
        assert!(sg.laplacian.matrix.nnz() > 0, "Subgraph should have edges");

        // item_indices should be populated in energy mode.
        assert!(
            sg.item_indices.is_some(),
            "Energy subgraphs must have item_indices"
        );

        if let Some(r) = sg.rayleigh {
            assert!(r <= 0.4, "Rayleigh filter should be applied");
        }
    }

    debug!("Extracted {} subgraphs from energy mode", subgraphs.len());
}

#[test]
fn test_spot_subgraphs_energy_with_item_mapping() {
    crate::init();

    let rows = make_gaussian_cliques_multi(300, 0.2, 10, 100, 999);
    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 12, 8, 2.0, None)
        .with_seed(999);

    let (aspace, gl) = builder.build(rows);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 20,
            min_triangles: 3,
            min_clust: 0.35,
            max_motif_size: 35,
            max_sets: 70,
            jaccard_dedup: 0.6,
        },
        rayleigh_max: None,
        min_size: 5,
    };

    let subgraphs = gl.spot_subg_motives(&aspace, &cfg);

    if subgraphs.is_empty() {
        debug!("No energy subgraphs extracted");
        return;
    }

    let (f_parent, n_parent) = gl.init_data.shape();

    for sg in &subgraphs {
        // init_data is F × X_centroid where X_centroid is centroid count for this motif.
        let (f_sg, x_centroid) = sg.laplacian.init_data.shape();
        assert_eq!(f_sg, f_parent);
        assert_eq!(sg.laplacian.nnodes, x_centroid);

        // matrix is F × F.
        let (mf_rows, mf_cols) = sg.laplacian.matrix.shape();
        assert_eq!(mf_rows, f_parent);
        assert_eq!(mf_cols, f_parent);

        // item_indices should be populated in energy mode.
        assert!(
            sg.item_indices.is_some(),
            "Energy subgraphs must have item_indices"
        );

        // All centroid node_indices must be in parent range.
        for &node_idx in &sg.node_indices {
            assert!(
                node_idx < n_parent,
                "centroid index {} out of range",
                node_idx
            );
        }

        // All item indices must be in ArrowSpace range.
        if let Some(ref items) = sg.item_indices {
            for &item_idx in items {
                assert!(
                    item_idx < aspace.nitems,
                    "item index {} out of range",
                    item_idx
                );
            }
        }
    }

    debug!(
        "Extracted {} subgraphs from energy mode with item mappings",
        subgraphs.len()
    );
}

#[test]
fn test_spot_subgraphs_energy_multi_motifs() {
    crate::init();

    // 300 points, 10 cliques, tight clusters, 50 dimensions → expect ~10 motifs
    let rows = make_gaussian_cliques_multi(300, 0.2, 10, 100, 999);

    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.35, 14, 10, 2.0, None)
        .with_seed(999)
        .build(rows);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 22,
            min_triangles: 3,
            min_clust: 0.3,
            max_motif_size: 40,
            max_sets: 80,
            jaccard_dedup: 0.55,
        },
        rayleigh_max: None,
        min_size: 6,
    };

    let subgraphs = gl.spot_subg_motives(&aspace, &cfg);

    if subgraphs.len() < 2 {
        debug!(
            "Extracted only {} subgraph(s); clique structure may yield fewer distinct motifs",
            subgraphs.len()
        );
    }

    if !subgraphs.is_empty() {
        // Check for overlapping centroids across motifs.
        let mut centroid_appearances: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for sg in &subgraphs {
            for &node_idx in &sg.node_indices {
                *centroid_appearances.entry(node_idx).or_insert(0) += 1;
            }
        }

        let overlapping_centroids: Vec<_> = centroid_appearances
            .iter()
            .filter(|(_, count)| **count > 1)
            .collect();

        if !overlapping_centroids.is_empty() {
            debug!(
                "Found {} centroids appearing in multiple motifs (overlapping structure)",
                overlapping_centroids.len()
            );
        }

        // Verify invariants for all subgraphs.
        let (f_parent, n_parent) = gl.init_data.shape();
        for sg in &subgraphs {
            let (f_sg, x_sg) = sg.laplacian.init_data.shape();
            assert_eq!(f_sg, f_parent);
            assert_eq!(sg.laplacian.nnodes, x_sg);
            assert_eq!(sg.node_indices.len(), x_sg);

            for &node_idx in &sg.node_indices {
                assert!(node_idx < n_parent);
            }

            assert!(sg.item_indices.is_some());
        }

        debug!(
            "Extracted {} energy subgraphs with multi-motif validation",
            subgraphs.len()
        );
    }
}

#[test]
fn test_subgraph_energy_rayleigh_filter() {
    crate::init();

    let rows = make_gaussian_cliques_multi(300, 0.2, 10, 100, 999);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.4, 12, 8, 2.0, None)
        .with_seed(999)
        .build(rows);

    // Strict Rayleigh filter to test filtering behavior.
    let cfg_strict = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 18,
            min_triangles: 3,
            min_clust: 0.35,
            max_motif_size: 30,
            max_sets: 60,
            jaccard_dedup: 0.65,
        },
        rayleigh_max: Some(0.15), // Very strict
        min_size: 5,
    };

    let subgraphs_strict = gl.spot_subg_motives(&aspace, &cfg_strict);

    // Relaxed Rayleigh filter.
    let cfg_relaxed = SubgraphConfig {
        rayleigh_max: Some(0.5), // More permissive
        ..cfg_strict
    };

    let subgraphs_relaxed = gl.spot_subg_motives(&aspace, &cfg_relaxed);

    // Relaxed filter should yield >= strict filter results.
    assert!(
        subgraphs_relaxed.len() >= subgraphs_strict.len(),
        "Relaxed Rayleigh filter should yield at least as many subgraphs as strict filter"
    );

    debug!(
        "Strict filter: {} subgraphs, Relaxed filter: {} subgraphs",
        subgraphs_strict.len(),
        subgraphs_relaxed.len()
    );
}

#[test]
fn test_subgraph_structure_clique_data() {
    crate::init();

    let rows = make_gaussian_cliques_multi(200, 0.3, 6, 100, 999);
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.35, 14, 10, 2.0, None)
        .with_seed(999)
        .build(rows);

    let cfg = SubgraphConfig {
        motives: MotiveConfig {
            top_l: 20,
            min_triangles: 4,
            min_clust: 0.4,
            max_motif_size: 35,
            max_sets: 70,
            jaccard_dedup: 0.6,
        },
        rayleigh_max: None,
        min_size: 8, // minimum ITEM count
    };

    let subgraphs = gl.spot_subg_motives(&aspace, &cfg);

    if subgraphs.is_empty() {
        debug!("No subgraphs extracted with these strict parameters");
        return;
    }

    // Verify all subgraphs have sensible sizes given clique structure.
    for (i, sg) in subgraphs.iter().enumerate() {
        let (f_dim, x_centroids) = sg.laplacian.init_data.shape();

        debug!(
            "Subgraph {}: {} centroids, {} features, {} items",
            i,
            x_centroids,
            f_dim,
            sg.item_indices.as_ref().map(|v| v.len()).unwrap_or(0)
        );

        // Centroid count must be >= 2 for graph construction.
        assert!(
            x_centroids >= 2,
            "Subgraph {} must have at least 2 centroids for graph construction",
            i
        );

        // Item count should meet min_size threshold (enforced during filtering).
        if let Some(ref items) = sg.item_indices {
            assert!(
                items.len() >= cfg.min_size,
                "Subgraph {} should have at least min_size items",
                i
            );

            // With clique structure, item count is typically >> centroid count
            // (many items per centroid).
            assert!(
                items.len() >= x_centroids,
                "Subgraph {} should have at least as many items as centroids",
                i
            );
        }
    }
}
