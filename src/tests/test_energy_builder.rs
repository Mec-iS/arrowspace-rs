// test_energy_builder.rs
#![cfg(test)]

use crate::builder::ArrowSpaceBuilder;
use crate::energymaps::{EnergyMapsBuilder, EnergyParams};
use crate::graph::GraphLaplacian;
use crate::taumode::TauMode;
use log::info;

#[cfg(test)]
mod test_data {
    pub use crate::tests::test_data::{make_gaussian_hd, make_moons_hd};
}

#[test]
fn test_energy_build_basic() {
    crate::tests::init();
    info!("Test: build_energy basic pipeline");

    let rows = test_data::make_gaussian_hd(100, 0.2);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(12345)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows,EnergyParams::new(&builder));

    assert!(aspace.nitems > 0);
    assert!(aspace.nfeatures == 100);
    assert!(gl_energy.nnodes > 0);
    assert!(gl_energy.nnz() > 0);
    assert!(aspace.lambdas.iter().any(|&l| l != 0.0));

    info!(
        "✓ Energy build succeeded with {} items, {} GL nodes",
        aspace.nitems, gl_energy.nnodes
    );
}

#[test]
fn test_energy_build_with_optical_compression() {
    crate::tests::init();
    info!("Test: build_energy with optical compression");

    let rows = test_data::make_moons_hd(150, 0.2, 0.1, 100, 42);
    let mut p = EnergyParams::default();
    p.optical_tokens = Some(30);
    p.trim_quantile = 0.15;

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(9999)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows, p);

    assert!(aspace.nitems > 0);
    assert!(gl_energy.nnodes <= 30 * 2);
    assert!(aspace.lambdas.iter().any(|&l| l > 0.0));

    info!(
        "✓ Optical compression: {} GL nodes (target ≤ {})",
        gl_energy.nnodes, 30
    );
}

#[test]
fn test_energy_build_diffusion_splits() {
    crate::tests::init();
    info!("Test: build_energy diffusion and sub-centroid splitting");

    let rows = test_data::make_gaussian_hd(80, 0.3);
    let mut p = EnergyParams::default();
    p.steps = 6;
    p.split_quantile = 0.85;
    p.split_tau = 0.2;

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(5555)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows, p);

    assert!(aspace.n_clusters > 0);
    assert!(gl_energy.nnodes >= aspace.n_clusters);
    assert!(aspace.lambdas().iter().all(|&l| l.is_finite()));

    info!(
        "✓ Diffusion + splitting: {} clusters → {} GL nodes",
        aspace.n_clusters, gl_energy.nnodes
    );
}

#[test]
fn test_energy_laplacian_properties() {
    crate::tests::init();
    info!("Test: energy Laplacian properties (connectivity, symmetry)");

    let rows = test_data::make_moons_hd(60, 0.2, 0.1, 99, 42);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(7777)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (_, gl_energy) = builder.build_energy(rows, EnergyParams::new(&builder));

    let sparsity = GraphLaplacian::sparsity(&gl_energy.matrix);
    assert!(sparsity < 0.95, "Laplacian should not be too sparse");
    assert!(sparsity > 0.0, "Laplacian should have some sparsity");

    let is_sym = gl_energy.is_symmetric(1e-6);
    assert!(is_sym, "Energy Laplacian should be symmetric");

    info!(
        "✓ Laplacian: {:.2}% sparse, symmetric={}",
        sparsity * 100.0,
        is_sym
    );
}

#[test]
fn test_energy_build_with_projection() {
    crate::tests::init();
    info!("Test: build_energy with JL projection");

    let rows = test_data::make_gaussian_hd(70, 0.4);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(222)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, gl_energy) = builder.build_energy(rows, EnergyParams::new(&builder));

    assert!(aspace.projection_matrix.is_some());
    assert!(aspace.reduced_dim.is_some());
    assert!(aspace.reduced_dim.unwrap() < 128);
    assert!(aspace.lambdas().iter().any(|&l| l > 0.0));

    info!(
        "✓ Projection: 128 → {} dims, {} GL nodes",
        aspace.reduced_dim.unwrap(),
        gl_energy.nnodes
    );
}

#[test]
fn test_energy_build_taumode_consistency() {
    crate::tests::init();
    info!("Test: build_energy taumode consistency");

    let rows = test_data::make_moons_hd(50, 0.2, 0.08, 99, 42);

    let mut builder = ArrowSpaceBuilder::new()
        .with_synthesis(TauMode::Mean)
        .with_seed(111)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, _) = builder.build_energy(rows, EnergyParams::new(&builder));

    assert_eq!(aspace.taumode, TauMode::Mean);
    assert!(aspace.lambdas.len() == aspace.nitems);
    assert!(aspace.lambdas.iter().all(|&l| l >= 0.0 && l.is_finite()));

    let lambda_mean = aspace.lambdas.iter().sum::<f64>() / aspace.lambdas.len() as f64;
    info!(
        "✓ Taumode Mean: {} lambdas, mean={:.6}",
        aspace.lambdas.len(),
        lambda_mean
    );
}

#[test]
fn test_energy_build_custom_params() {
    crate::tests::init();
    info!("Test: build_energy with custom EnergyParams");

    let rows = test_data::make_gaussian_hd(40, 0.1);
    let p = EnergyParams {
        optical_tokens: None,
        trim_quantile: 0.05,
        eta: 0.15,
        steps: 2,
        split_quantile: 0.95,
        neighbor_k: 10,
        split_tau: 0.1,
        w_lambda: 1.5,
        w_disp: 0.3,
        w_dirichlet: 0.15,
        candidate_m: 20,
    };

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(333)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let lambda_k = builder.lambda_k.clone();
    let (aspace, gl_energy) = builder.build_energy(rows, p);

    assert_eq!(gl_energy.graph_params.k, lambda_k);
    assert!(!gl_energy.graph_params.normalise);
    assert!(aspace.lambdas.iter().any(|&l| l > 0.0));

    info!(
        "✓ Custom params: k={}, normalize={}",
        gl_energy.graph_params.k, gl_energy.graph_params.normalise
    );
}

#[test]
fn test_energy_build_lambda_statistics() {
    crate::tests::init();
    info!("Test: build_energy lambda statistics");

    let rows = test_data::make_moons_hd(100, 0.2, 0.1, 99, 42);

    let mut builder = ArrowSpaceBuilder::new()
        .with_seed(444)
        .with_dims_reduction(true, Some(0.3))
        .with_inline_sampling(None);

    let (aspace, _) = builder.build_energy(rows, EnergyParams::new(&builder));

    let lambdas = aspace.lambdas();
    let min = lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = lambdas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;

    assert!(min >= 0.0);
    assert!(max > min);
    assert!(mean > 0.0 && mean.is_finite());

    info!(
        "✓ Lambda stats: min={:.6}, max={:.6}, mean={:.6}",
        min, max, mean
    );
}
