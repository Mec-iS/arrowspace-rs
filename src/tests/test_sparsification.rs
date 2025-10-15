#[test]
fn test_sfgrass_basic() {
    let adj_rows = vec![
        vec![(1, 1.0), (2, 0.5)],
        vec![(0, 1.0), (2, 0.8)],
        vec![(0, 0.5), (1, 0.8)],
    ];

    let sparsifier = SfGrassSparsifier::new();
    let result = sparsifier.sparsify_graph(&adj_rows, 3);

    assert_eq!(result.len(), 3);
    assert!(result.iter().all(|row| !row.is_empty()));
}

#[test]
fn test_sfgrass_larger() {
    let n = 50;
    let adj_rows: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|i| {
            (0..n)
                .filter_map(|j| {
                    if i != j && (i + j) % 3 == 0 {
                        Some((j, 1.0 / (1.0 + ((i as i32 - j as i32).abs() as f64))))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    let sparsifier = SfGrassSparsifier::new();
    let result = sparsifier.sparsify_graph(&adj_rows, n);

    assert_eq!(result.len(), n);
    let orig_edges: usize = adj_rows.iter().map(|r| r.len()).sum();
    let sparse_edges: usize = result.iter().map(|r| r.len()).sum();

    assert!(sparse_edges < orig_edges);
}
use crate::{builder::ArrowSpaceBuilder, sparsification::SfGrassSparsifier, tests::test_data::make_gaussian_blob};

#[test]
fn test_sfgrass_sparsification() {
    let rows: Vec<Vec<f64>> = make_gaussian_blob(300, 0.5);

    let (aspace, _) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1.0, 5, 5, 2.0, None)
        .with_seed(42)
        .build(rows);

    // Should produce valid lambdas
    assert!(aspace.lambdas().iter().all(|&l| l >= 0.0));
}

#[test]
#[ignore = "depends on number of nodes"]
fn test_sfgrass_vs_no_sparsification() {
    let rows_sparse: Vec<Vec<f64>> = (0..10000)
        .map(|i| vec![(i as f64 / 100.0).sin(), (i as f64 / 100.0).cos()])
        .collect();

    let rows_full: Vec<Vec<f64>> = (0..10000)
        .map(|i| vec![(i as f64 / 100.0).sin(), (i as f64 / 100.0).cos()])
        .collect();

    // With SF-GRASS
    let start = std::time::Instant::now();
    let (_, _) = ArrowSpaceBuilder::new().build(rows_sparse.clone());
    let time_sparse = start.elapsed();

    // Without
    let start = std::time::Instant::now();
    let (_, _) = ArrowSpaceBuilder::new().build(rows_full);
    let time_full = start.elapsed();

    // Should be faster
    println!("SF-GRASS: {:?}, Full: {:?}", time_sparse, time_full);
    assert!(time_sparse.as_millis() < time_full.as_millis());
}
