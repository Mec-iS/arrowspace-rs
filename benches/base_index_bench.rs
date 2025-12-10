use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::core::{ArrowItem, ArrowSpace};
use arrowspace::graph::GraphLaplacian;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use smartcore::dataset::iris;
use smartcore::linalg::basic::arrays::Array;
use std::hint::black_box;
use std::time::Duration;

#[path = "../examples/common/lib.rs"]
mod common;

fn build_arrowspace(db: &[Vec<f64>]) -> (ArrowSpace, GraphLaplacian) {
    let eps = 1e-1;
    let cap_k = 10;
    let topk = 3;
    let p = 2.0;
    let (aspace, gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(eps, cap_k, topk, p, None)
        .build(db.to_vec());
    assert_eq!(gl.nnodes, aspace.data.shape().0);
    (aspace, gl)
}

fn pick_query(mut base: Vec<f64>) -> Vec<f64> {
    for v in base.iter_mut() {
        *v *= 1.02;
    }
    base
}

fn setup_single() -> (Vec<Vec<f64>>, Vec<f64>, usize, ArrowSpace) {
    // Prepare a moderate dataset; Iris shown here, but any non-degenerate set works
    let dataset = iris::load_dataset();
    let items: Vec<Vec<f64>> = dataset
        .as_matrix()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| {
                    let mut v = *val as f64;
                    v *= 100.0;
                    v
                })
                .collect()
        })
        .collect();

    let (aspace, _gl) = build_arrowspace(&items);

    let q_index = 3;
    let query = pick_query(items[q_index].clone());
    (items, query, 3, aspace)
}

fn setup_batch(batch_size: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, usize, ArrowSpace) {
    let dataset = iris::load_dataset();
    let items: Vec<Vec<f64>> = dataset
        .as_matrix()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| {
                    let mut v = *val as f64;
                    v *= 100.0;
                    v
                })
                .collect()
        })
        .collect();
    let len_items = items.len();
    let (aspace, _gl) = build_arrowspace(&items);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut queries = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let idx = rng.random_range(0..len_items);
        queries.push(pick_query(items[idx].clone()));
    }
    (items, queries, 3, aspace)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    {
        let (db, query, _, aspace) = setup_single();

        // baseline cosine
        let base_scores: Vec<(usize, f64)> = db
            .iter()
            .enumerate()
            .map(|(i, v)| (i, common::cosine_similarity(&query, v)))
            .collect();
        let ids_base: Vec<usize> = base_scores.iter().map(|x| x.0).collect();

        // arrow cosine-equivalent (alpha=1,beta=0)
        let qrow = ArrowItem::new(query.as_ref(), 0.0);
        let arr_scores: Vec<(usize, f64)> = (0..db.len())
            .map(|i| {
                let item_i = aspace.get_item(i);
                (i, qrow.lambda_similarity(&item_i, 1.0))
            })
            .collect();
        let ids_arr: Vec<usize> = arr_scores.iter().map(|x| x.0).collect();

        assert_eq!(
            ids_base, ids_arr,
            "alpha=1,beta=0 must match baseline cosine"
        );
    }

    let mut group = c.benchmark_group("lookup_topk_k=3");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(40);

    // --- single query ---
    group.bench_function(BenchmarkId::new("baseline_cosine", "single"), |b| {
        b.iter_batched(
            setup_single,
            |(db, query, _k, _aspace)| {
                let scores: Vec<(usize, f64)> = db
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, common::cosine_similarity(&query, v)))
                    .collect();
                black_box(scores);
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function(BenchmarkId::new("arrow_alpha1_beta0", "single"), |b| {
        b.iter_batched(
            setup_single,
            |(db, query, _k, aspace)| {
                let qrow = ArrowItem::new(query.as_ref(), 0.0);
                let scores: Vec<(usize, f64)> = (0..db.len())
                    .map(|i| {
                        let item_i = aspace.get_item(i);
                        (i, qrow.lambda_similarity(&item_i, 1.0))
                    })
                    .collect();
                black_box(scores);
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function(BenchmarkId::new("arrow_alpha0.9_beta0.1", "single"), |b| {
        b.iter_batched(
            setup_single,
            |(db, query, _k, aspace)| {
                let qrow = ArrowItem::new(query.as_ref(), 0.0);
                let scores: Vec<(usize, f64)> = (0..db.len())
                    .map(|i| {
                        let item_i = aspace.get_item(i);
                        (i, qrow.lambda_similarity(&item_i, 0.9))
                    })
                    .collect();
                black_box(scores);
            },
            BatchSize::SmallInput,
        )
    });

    // --- batch queries ---
    for &batch in &[16usize, 64, 128, 256] {
        group.bench_function(
            BenchmarkId::new("baseline_cosine", format!("batch{batch}")),
            |b| {
                b.iter_batched(
                    || setup_batch(batch, 42),
                    |(db, queries, _k, _aspace)| {
                        let mut acc = 0.0;
                        for query in queries {
                            let scores: Vec<(usize, f64)> = db
                                .iter()
                                .enumerate()
                                .map(|(i, v)| (i, common::cosine_similarity(&query, v)))
                                .collect();
                            acc += scores.iter().map(|(_, s)| s).sum::<f64>();
                        }
                        black_box(acc);
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_function(
            BenchmarkId::new("arrow_alpha1_beta0", format!("batch{batch}")),
            |b| {
                b.iter_batched(
                    || setup_batch(batch, 42),
                    |(db, queries, _k, aspace)| {
                        let mut acc = 0.0;
                        for query in queries {
                            let qrow = ArrowItem::new(query.as_ref(), 0.0);
                            let scores: Vec<(usize, f64)> = (0..db.len())
                                .map(|i| {
                                    let item_i = aspace.get_item(i);
                                    (i, qrow.lambda_similarity(&item_i, 1.0))
                                })
                                .collect();
                            acc += scores.iter().map(|(_, s)| s).sum::<f64>();
                        }
                        black_box(acc);
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_function(
            BenchmarkId::new("arrow_alpha0.9_beta0.1", format!("batch{batch}")),
            |b| {
                b.iter_batched(
                    || setup_batch(batch, 42),
                    |(db, queries, _k, aspace)| {
                        let mut acc = 0.0;
                        for query in queries {
                            let qrow = ArrowItem::new(query.as_ref(), 0.0);
                            let scores: Vec<(usize, f64)> = (0..db.len())
                                .map(|i| {
                                    let item_i = aspace.get_item(i);
                                    (i, qrow.lambda_similarity(&item_i, 0.9))
                                })
                                .collect();
                            acc += scores.iter().map(|(_, s)| s).sum::<f64>();
                        }
                        black_box(acc);
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
