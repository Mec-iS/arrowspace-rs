//! Integration tests verifying that the EigenMaps trait stages produce identical
//! results to the original ArrowSpaceBuilder::build monolithic path.
//!
//! # Test Strategy
//!
//! Each test constructs an ArrowSpace and GraphLaplacian using both:
//! 1. The original `builder.build(rows)` path (control)
//! 2. The new EigenMaps trait stages: start_clustering → eigenmaps → compute_taumode → search
//!
//! We then verify equivalence on:
//! - Clustering metadata (n_clusters, cluster_assignments, cluster_sizes)
//! - Laplacian properties (shape, nnz, sparsity, node count)
//! - Lambda values (element-wise relative equality within tolerance)
//! - Search results (same top-k indices and scores within tolerance)
//!
//! # Determinism
//!
//! Tests use `with_seed(...)` to ensure reproducible clustering and stable results
//! across both paths. Without seeding, clustering is nondeterministic due to parallel
//! processing in incremental clustering.

use approx::relative_ne;
use log::info;

use crate::builder::ArrowSpaceBuilder;
use crate::core::ArrowSpace;
use crate::eigenmaps::{ClusteredOutput, EigenMaps};
use crate::graph::GraphLaplacian;
use crate::taumode::TauMode;

use crate::tests::test_data::make_gaussian_hd;
use crate::tests::init;

/// Helper: Compare two ArrowSpace lambda vectors element-wise with tolerance.
fn assert_lambdas_equal(a: &[f64], b: &[f64], tol: f64, label: &str, spectral: bool) {
    init();
    assert_eq!(
        a.len(),
        b.len(),
        "{}: lambda vector length mismatch",
        label
    );
    for (i, (&la, &lb)) in a.iter().zip(b.iter()).enumerate() {
        if !spectral {
            assert!(la >= 0.0 && lb >= 0.0, "lambda_a is {} and lambda_b {}", la, lb);
        }
        if relative_ne!(
            la,
            lb,
            epsilon = tol,
            max_relative = tol) {
                println!("lambdas are not equal: {} {}", la, lb);
                panic!("{}[{}]: lambda mismatch", label, i)
            } else {};
    }
}

/// Helper: Compare clustering metadata between two ArrowSpace instances.
fn assert_clustering_equal(a: &ArrowSpace, b: &ArrowSpace, label: &str) {
    init();
    assert_eq!(
        a.n_clusters, b.n_clusters,
        "{}: n_clusters mismatch",
        label
    );
    assert_eq!(
        a.cluster_assignments, b.cluster_assignments,
        "{}: cluster_assignments mismatch",
        label
    );
    assert_eq!(
        a.cluster_sizes, b.cluster_sizes,
        "{}: cluster_sizes mismatch",
        label
    );
    if relative_ne!(
        a.cluster_radius,
        b.cluster_radius,
        epsilon = 1e-9) {
            panic!("{}: cluster_radius mismatch", label)
        } else {};
}

/// Helper: Compare GraphLaplacian properties (shape, nnodes, nnz, sparsity).
fn assert_laplacian_equal(a: &GraphLaplacian, b: &GraphLaplacian, label: &str) {
    init();
    assert_eq!(a.shape(), b.shape(), "{}: Laplacian shape mismatch", label);
    assert_eq!(a.nnodes, b.nnodes, "{}: nnodes mismatch", label);
    assert_eq!(a.nnz(), b.nnz(), "{}: nnz mismatch", label);
    if relative_ne!(
        GraphLaplacian::sparsity(&a.matrix),
        GraphLaplacian::sparsity(&b.matrix),
        epsilon = 1e-9) {
            panic!("{}: sparsity mismatch", label)
        } else {};
}

/// Helper: Compare search results (indices and scores within tolerance).
fn assert_search_results_equal(
    a: &[(usize, f64)],
    b: &[(usize, f64)],
    tol: f64,
    label: &str,
) {
    assert_eq!(a.len(), b.len(), "{}: result count mismatch", label);
    for (i, ((idx_a, score_a), (idx_b, score_b))) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(idx_a, idx_b, "{}[{}]: index mismatch", label, i);
        if relative_ne!(
            score_a,
            score_b,
            epsilon = tol) {
                panic!("{}[{}]: score mismatch", label, i)
            } else {};
    }
}

#[test]
fn test_eigenmaps_vs_build_basic() {
    crate::init(); // Initialize logger
    info!("Test: EigenMaps trait vs build() - basic dataset");

    let rows = make_gaussian_hd(99, 0.6);
    let query = vec![0.5; 100];
    let k = 5;
    let tau = 0.7;

    // Control: original build path
    let builder_control = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_synthesis(TauMode::Median)
        .with_seed(12345) // Deterministic clustering
        .with_inline_sampling(None) // Disable sampling for exact equivalence
        .with_dims_reduction(false, None);

    let (aspace_control, gl_control) = builder_control.build(rows.clone());

    // Experimental: EigenMaps trait stages
    let mut builder_exp = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_synthesis(TauMode::Median)
        .with_seed(12345) // Same seed
        .with_inline_sampling(None)
        .with_dims_reduction(false, None);

    let ClusteredOutput {
        mut aspace,
        centroids,
        n_items,
        ..
    } = ArrowSpace::start_clustering(&mut builder_exp, rows.clone());

    let gl_exp = aspace.eigenmaps(&builder_exp, &centroids, n_items);
    aspace.compute_taumode(&gl_exp);

    // Verify clustering metadata
    assert_clustering_equal(&aspace_control, &aspace, "basic");

    // Verify Laplacian properties
    assert_laplacian_equal(&gl_control, &gl_exp, "basic");

    // Verify lambda values
    assert_lambdas_equal(
        aspace_control.lambdas(),
        aspace.lambdas(),
        1e-6,
        "basic lambdas",
        false
    );

    // Verify search results
    let results_control = aspace_control.search_lambda_aware(
        &crate::core::ArrowItem::new(
            query.clone(),
            aspace_control.prepare_query_item(&query, &gl_control),
        ),
        k,
        tau,
    );

    let results_exp = aspace.search(&query, &gl_exp, k, tau);

    assert_search_results_equal(&results_control, &results_exp, 1e-6, "basic search");

    info!("✓ EigenMaps trait matches build() for basic dataset");
}

#[test]
fn test_eigenmaps_vs_build_with_spectral() {
    crate::init();
    info!("Test: EigenMaps trait vs build() - with spectral Laplacian");

    let rows = make_gaussian_hd(99, 0.6);
    let query = vec![0.1; 100];
    let k = 4;
    let tau = 0.6;

    // Control: build with spectral enabled
    let builder_control = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_synthesis(TauMode::Median)
        .with_spectral(true) // Enable F×F feature Laplacian
        .with_seed(5555)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None);

    let (aspace_control, gl_control) = builder_control.build(rows.clone());

    // Experimental: EigenMaps with spectral stage
    let mut builder_exp = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_synthesis(TauMode::Median)
        .with_spectral(true)
        .with_seed(5555)
        .with_dims_reduction(false, None)
        .with_inline_sampling(None);

    let ClusteredOutput {
        mut aspace,
        centroids,
        n_items,
        ..
    } = ArrowSpace::start_clustering(&mut builder_exp, rows.clone());

    let gl_exp = aspace.eigenmaps(&builder_exp, &centroids, n_items);
    aspace.compute_taumode(&gl_exp);

    // Verify spectral matrix (signals field)
    assert_eq!(
        aspace_control.signals.shape(),
        aspace.signals.shape(),
        "signals shape mismatch"
    );
    assert_eq!(
        aspace_control.signals.nnz(),
        aspace.signals.nnz(),
        "signals nnz mismatch"
    );

    // Verify other properties
    assert_clustering_equal(&aspace_control, &aspace, "spectral");
    assert_laplacian_equal(&gl_control, &gl_exp, "spectral");
    assert_lambdas_equal(
        aspace_control.lambdas(),
        aspace.lambdas(),
        1e-6,
        "spectral lambdas",
        true
    );

    // Search
    let results_control = aspace_control.search_lambda_aware(
        &crate::core::ArrowItem::new(
            query.clone(),
            aspace_control.prepare_query_item(&query, &gl_control),
        ),
        k,
        tau,
    );

    let results_exp = aspace.search(&query, &gl_exp, k, tau);

    assert_search_results_equal(&results_control, &results_exp, 1e-6, "spectral search");

    info!("✓ EigenMaps trait matches build() with spectral Laplacian");
}

#[test]
fn test_eigenmaps_vs_build_different_taumode() {
    crate::init();
    info!("Test: EigenMaps trait vs build() - Mean taumode");

    let rows = make_gaussian_hd(99, 0.6);
    let query = vec![-0.2; 100];
    let k = 6;
    let tau = 0.75;

    // Control: build with Mean taumode
    let builder_control = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_synthesis(TauMode::Mean) // Different tau policy
        .with_seed(7777)
        .with_inline_sampling(None);

    let (aspace_control, gl_control) = builder_control.build(rows.clone());

    // Experimental: EigenMaps with Mean taumode
    let mut builder_exp = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_synthesis(TauMode::Mean)
        .with_seed(7777)
        .with_inline_sampling(None);

    let ClusteredOutput {
        mut aspace,
        centroids,
        n_items,
        ..
    } = ArrowSpace::start_clustering(&mut builder_exp, rows.clone());

    let gl_exp = aspace.eigenmaps(&builder_exp, &centroids, n_items);
    aspace.compute_taumode(&gl_exp);

    // Verify
    assert_clustering_equal(&aspace_control, &aspace, "mean_taumode");
    assert_laplacian_equal(&gl_control, &gl_exp, "mean_taumode");
    assert_lambdas_equal(
        aspace_control.lambdas(),
        aspace.lambdas(),
        1e-6,
        "mean_taumode lambdas",
        false
    );

    // Search
    let results_control = aspace_control.search_lambda_aware(
        &crate::core::ArrowItem::new(
            query.clone(),
            aspace_control.prepare_query_item(&query, &gl_control),
        ),
        k,
        tau,
    );

    let results_exp = aspace.search(&query, &gl_exp, k, tau);

    assert_search_results_equal(&results_control, &results_exp, 1e-6, "mean_taumode search");

    info!("✓ EigenMaps trait matches build() with Mean taumode");
}

#[test]
#[should_panic(expected = "call compute_taumode")]
fn test_search_without_taumode_panics() {
    crate::init();
    info!("Test: Search without compute_taumode should panic in debug");

    let rows = make_gaussian_hd(99, 0.6);
    let query = vec![0.0; 6];

    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_seed(111)
        .with_inline_sampling(None);

    let ClusteredOutput {
        mut aspace,
        centroids,
        n_items,
        ..
    } = ArrowSpace::start_clustering(&mut builder, rows);

    let gl = aspace.eigenmaps(&builder, &centroids, n_items);

    // Skip compute_taumode - should panic in debug
    let _ = aspace.search(&query, &gl, 3, 0.7);
}

#[test]
fn test_eigenmaps_stages_produce_valid_state() {
    crate::init();
    info!("Test: EigenMaps stages produce valid intermediate state");

    let rows = make_gaussian_hd(99, 0.6);

    let mut builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 3, 3, 2.0, None)
        .with_seed(222)
        .with_inline_sampling(None);

    // Stage 1: Clustering
    let ClusteredOutput {
        mut aspace,
        centroids,
        n_items,
        n_features,
        reduced_dim,
    } = ArrowSpace::start_clustering(&mut builder, rows.clone());

    assert_eq!(n_items, 99, "n_items should match input");
    assert_eq!(n_features, 100, "n_features should match input");
    assert!(aspace.n_clusters > 0, "Should have clustered");
    assert_eq!(reduced_dim, n_features, "No projection, dims unchanged");

    // Stage 2: Eigenmaps
    let gl = aspace.eigenmaps(&builder, &centroids, n_items);

    assert_eq!(gl.nnodes, n_items, "Laplacian nodes should match items");
    assert!(gl.nnz() > 0, "Laplacian should have edges");
    assert!(
        GraphLaplacian::sparsity(&gl.matrix) < 1.0,
        "Should not be fully sparse"
    );

    // Stage 3: Taumode (check lambdas are computed)
    let mut aspace = aspace; // Make mutable
    assert!(
        aspace.lambdas().iter().all(|&l| l == 0.0),
        "Lambdas should be zero before compute_taumode"
    );

    aspace.compute_taumode(&gl);

    assert!(
        aspace.lambdas().iter().any(|&l| l > 0.0),
        "Lambdas should be nonzero after compute_taumode"
    );

    info!("✓ All stages produce valid intermediate state");
}
