use arrowspace::graph::GraphParams;
use arrowspace::laplacian::build_laplacian_matrix;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::hint::black_box;
use std::time::Duration;

#[path = "../examples/common/lib.rs"]
mod common;

/// Generate synthetic dataset with specified number of items and dimensions
fn generate_synthetic_dataset(n_items: usize, n_dims: usize, seed: u64) -> DenseMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut items = Vec::with_capacity(n_items);

    for i in 0..n_items {
        let mut item = Vec::with_capacity(n_dims);
        // Create some structure: items are variations of a base pattern
        let base_pattern = (i % 10) as f64 * 0.1;
        for j in 0..n_dims {
            let noise: f64 = rng.random_range(-0.1..0.1);
            let value = (base_pattern + (j as f64 * 0.01) + noise).abs();
            item.push(value);
        }
        items.push(item);
    }
    DenseMatrix::from_2d_vec(&items).unwrap()
}

/// Setup function for real dataset with different parameter configurations
fn setup_real_dataset(params: GraphParams) -> (DenseMatrix<f64>, GraphParams) {
    let (_, items) = common::parse_vectors_block();
    (DenseMatrix::from_2d_vec(&items).unwrap(), params)
}

/// Setup function for synthetic datasets
fn setup_synthetic_dataset(
    n_items: usize,
    n_dims: usize,
    params: GraphParams,
    seed: u64,
) -> (DenseMatrix<f64>, GraphParams) {
    let items = generate_synthetic_dataset(n_items, n_dims, seed);
    (items, params)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Group 1: Real dataset with different graph parameters
    let mut group_real = c.benchmark_group("build_laplacian_real_dataset");
    group_real.warm_up_time(Duration::from_millis(500));
    group_real.measurement_time(Duration::from_secs(3));
    group_real.sample_size(20);

    // Test different k values
    for &k in &[2, 5, 10, 15] {
        let params = GraphParams {
            eps: 0.5,
            k,
            topk: k,
            p: 2.0,
            sigma: None,
            normalise: false,
            sparsity_check: false,
        };

        group_real.bench_function(BenchmarkId::new("k_variation", k), |b| {
            b.iter_batched(
                || setup_real_dataset(params.clone()),
                |(items, params)| {
                    let laplacian = build_laplacian_matrix(items, &params, Some(k), false);
                    black_box(laplacian);
                },
                BatchSize::SmallInput,
            )
        });
    }

    // Test different eps values
    for &eps in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        let params = GraphParams {
            eps,
            k: 5,
            topk: 5,
            p: 2.0,
            sigma: None,
            normalise: false,
            sparsity_check: false,
        };

        group_real.bench_function(
            BenchmarkId::new("eps_variation", format!("{:.1}", eps)),
            |b| {
                b.iter_batched(
                    || setup_real_dataset(params.clone()),
                    |(items, params)| {
                        let laplacian = build_laplacian_matrix(items, &params, None, false);
                        black_box(laplacian);
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    // Test normalization vs no normalization
    for &normalise in &[false, true] {
        let params = GraphParams {
            eps: 0.5,
            k: 5,
            topk: 5,
            p: 2.0,
            sigma: None,
            normalise,
            sparsity_check: false,
        };

        let label = if normalise { "normalized" } else { "raw" };
        group_real.bench_function(BenchmarkId::new("normalization", label), |b| {
            b.iter_batched(
                || setup_real_dataset(params.clone()),
                |(items, params)| {
                    let laplacian = build_laplacian_matrix(items, &params, None, false);
                    black_box(laplacian);
                },
                BatchSize::SmallInput,
            )
        });
    }

    // Test different p values (kernel exponent)
    for &p in &[1.0, 1.5, 2.0, 3.0] {
        let params = GraphParams {
            eps: 0.5,
            k: 5,
            topk: 5,
            p,
            sigma: None,
            normalise: false,
            sparsity_check: false,
        };

        group_real.bench_function(BenchmarkId::new("p_variation", format!("{:.1}", p)), |b| {
            b.iter_batched(
                || setup_real_dataset(params.clone()),
                |(items, params)| {
                    let laplacian = build_laplacian_matrix(items, &params, None, false);
                    black_box(laplacian);
                },
                BatchSize::SmallInput,
            )
        });
    }

    group_real.finish();

    // Group 2: Scalability test with synthetic datasets
    let mut group_scalability = c.benchmark_group("build_laplacian_scalability");
    group_scalability.warm_up_time(Duration::from_millis(500));
    group_scalability.measurement_time(Duration::from_secs(5));
    group_scalability.sample_size(10);

    let base_params = GraphParams {
        eps: 0.5,
        k: 5,
        topk: 5,
        p: 2.0,
        sigma: None,
        normalise: false,
        sparsity_check: false,
    };

    // Test scaling with number of items (fixed dimensionality)
    for &n_items in &[50, 100, 200, 400] {
        let n_dims = 24; // Same as real dataset

        group_scalability.bench_function(BenchmarkId::new("n_items", n_items), |b| {
            b.iter_batched(
                || setup_synthetic_dataset(n_items, n_dims, base_params.clone(), 42),
                |(items, params)| {
                    let laplacian = build_laplacian_matrix(items, &params, None, false);
                    black_box(laplacian);
                },
                BatchSize::SmallInput,
            )
        });
    }

    // Test scaling with dimensionality (fixed number of items)
    for &n_dims in &[10, 24, 50, 100] {
        let n_items = 100;

        group_scalability.bench_function(BenchmarkId::new("n_dims", n_dims), |b| {
            b.iter_batched(
                || setup_synthetic_dataset(n_items, n_dims, base_params.clone(), 42),
                |(items, params)| {
                    let laplacian = build_laplacian_matrix(items, &params, None, false);
                    black_box(laplacian);
                },
                BatchSize::SmallInput,
            )
        });
    }

    group_scalability.finish();

    // Group 3: Parameter combinations on medium dataset
    let mut group_combinations = c.benchmark_group("build_laplacian_param_combinations");
    group_combinations.warm_up_time(Duration::from_millis(300));
    group_combinations.measurement_time(Duration::from_secs(3));
    group_combinations.sample_size(15);

    let test_combinations = [
        (
            "sparse_graph",
            GraphParams {
                eps: 0.2,
                k: 3,
                topk: 3,
                p: 2.0,
                sigma: None,
                normalise: false,
                sparsity_check: false,
            },
        ),
        (
            "dense_graph",
            GraphParams {
                eps: 0.8,
                k: 15,
                topk: 7,
                p: 2.0,
                sigma: None,
                normalise: false,
                sparsity_check: false,
            },
        ),
        (
            "normalized_sparse",
            GraphParams {
                eps: 0.2,
                k: 3,
                topk: 3,
                p: 2.0,
                sigma: None,
                normalise: true,
                sparsity_check: false,
            },
        ),
        (
            "normalized_dense",
            GraphParams {
                eps: 0.8,
                k: 15,
                topk: 7,
                p: 2.0,
                sigma: None,
                normalise: true,
                sparsity_check: false,
            },
        ),
        (
            "high_exponent",
            GraphParams {
                eps: 0.5,
                k: 5,
                topk: 3,
                p: 4.0,
                sigma: None,
                normalise: false,
                sparsity_check: false,
            },
        ),
        (
            "custom_sigma",
            GraphParams {
                eps: 0.5,
                k: 5,
                topk: 3,
                p: 2.0,
                sigma: Some(0.1),
                normalise: false,
                sparsity_check: false,
            },
        ),
    ];

    for (name, params) in test_combinations.iter() {
        group_combinations.bench_function(BenchmarkId::new("combination", name), |b| {
            b.iter_batched(
                || setup_synthetic_dataset(100, 24, params.clone(), 42),
                |(items, params)| {
                    let laplacian = build_laplacian_matrix(items, &params, None, false);
                    black_box(laplacian);
                },
                BatchSize::SmallInput,
            )
        });
    }

    group_combinations.finish();

    // Group 4: Memory allocation patterns
    let mut group_memory = c.benchmark_group("build_laplacian_memory_patterns");
    group_memory.warm_up_time(Duration::from_millis(200));
    group_memory.measurement_time(Duration::from_secs(2));
    group_memory.sample_size(20);

    // Test with pre-allocated vs fresh data
    let params = GraphParams {
        eps: 0.5,
        k: 5,
        topk: 5,
        p: 2.0,
        sigma: None,
        normalise: false,
        sparsity_check: false,
    };

    group_memory.bench_function(BenchmarkId::new("fresh_allocation", "100x24"), |b| {
        b.iter(|| {
            let items = generate_synthetic_dataset(100, 24, 42);
            let laplacian = build_laplacian_matrix(items, &params, None, false);
            black_box(laplacian);
        })
    });

    group_memory.bench_function(BenchmarkId::new("reused_data", "100x24"), |b| {
        let items = generate_synthetic_dataset(100, 24, 42);
        b.iter(|| {
            let laplacian = build_laplacian_matrix(items.clone(), &params, None, false);
            black_box(laplacian);
        })
    });

    group_memory.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
