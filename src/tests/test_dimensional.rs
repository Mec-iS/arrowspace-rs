use smartcore::linalg::basic::arrays::Array;

use crate::core::ArrowSpace;
use crate::graph::GraphFactory;
use crate::tests::test_data::make_moons_hd;

use log::debug;

#[test]
fn two_emitters_superposition_lambda_with_moons_hd() {
    // High-dimensional moons with moderate noise in the informative plane and small high-D noise.
    // This yields a nontrivial k-NN graph and well-behaved Laplacian spectrum.
    let dims = 10;
    let n = 300;
    let items = make_moons_hd(n, 0.10, 0.02, dims, 42);

    // Split into two "emitters" by taking two disjoint rows (features) that act as separate signals.
    // Here, emulate two emitters by selecting two coordinate dimensions as two rows over the same items.
    // Row A: first coordinate of all items; Row B: second coordinate of all items.
    let row_a: Vec<f64> = items.iter().map(|v| v[0]).collect();
    let row_b: Vec<f64> = items.iter().map(|v| v[1]).collect();

    // ArrowSpace from rows (rows are features over items) — maintain two rows to combine later.
    let aspace_sum = ArrowSpace::from_items_default(vec![row_a.clone(), row_b.clone()]);
    let aspace_mul = ArrowSpace::from_items_default(vec![row_a, row_b]);

    // Reconstruct F×N matrix view (feature-major) for graph factory:
    // get_item(r) returns the r-th item (column) in ArrowSpace, so rebuild rows by iterating columns.
    let (nitems, _nfeatures) = aspace_sum.data.shape();
    // Expect two rows (features) over N items after from_items_default above.
    assert_eq!(
        nitems, 2,
        "Expected exactly two signal rows from row_a/row_b"
    );

    let mut data_matrix: Vec<Vec<f64>> = Vec::with_capacity(nitems);
    for r in 0..nitems {
        data_matrix.push(aspace_sum.get_item(r).item.to_vec());
    }
    // Sanity on sizes: 2 rows, N columns
    assert_eq!(data_matrix.len(), 2, "Expected two rows in data matrix");
    assert_eq!(
        data_matrix[0].len(),
        n,
        "Row length should equal number of items"
    );

    // Build λτ-graph with cosine-like normalization to stabilize affinities in high-D.
    // Tune parameters to yield a connected or near-connected graph with reasonable sparsity.
    // let eps = 1e-3;
    // let k = 12usize;       // slightly larger k to avoid over-sparsity in high-D k-NN [web:198]
    // let topk = 6usize;     // restrict to top edges to keep Laplacian stable
    // let p = 2.0;
    // let sigma = None;      // default σ = eps
    let gl = GraphFactory::build_laplacian_matrix(
        data_matrix,
        1e-3,
        12,
        6,
        2.0,
        Some(1e-3 * 0.50),
        true,
    );

    // Build spectral ArrowSpaces on same graph (Laplacian) and recompute λ
    let mut aspace_sum = GraphFactory::build_spectral_laplacian(aspace_sum, &gl);
    let mut aspace_mul = GraphFactory::build_spectral_laplacian(aspace_mul, &gl);

    aspace_sum.recompute_lambdas(&gl);
    aspace_mul.recompute_lambdas(&gl);

    // PSD sanity: λ values should be finite and non-negative
    assert!(aspace_sum.lambdas().iter().all(|&l| l.is_finite()));
    assert!(aspace_sum.lambdas().iter().all(|&l| l >= 0.0));
    assert!(aspace_mul.lambdas().iter().all(|&l| l.is_finite()));
    assert!(aspace_mul.lambdas().iter().all(|&l| l >= 0.0));

    // Log for inspection
    debug!("λ (sum, initial): {:?}", aspace_sum.lambdas());
    debug!("λ (mul, initial): {:?}", aspace_mul.lambdas());

    // Superpose b into a (item-wise add) and recompute λ
    aspace_sum.add_items(0, 1, &gl);
    debug!("λ (sum, after add): {:?}", aspace_sum.lambdas());
    assert!(aspace_sum.lambdas().iter().all(|&l| l.is_finite()));
    assert!(aspace_sum.lambdas().iter().all(|&l| l >= 0.0));

    // Multiply a times b (item-wise mul) and recompute λ
    aspace_mul.mul_items(0, 1, &gl);
    debug!("λ (mul, after mul): {:?}", aspace_mul.lambdas());
    assert!(aspace_mul.lambdas().iter().all(|&l| l.is_finite()));
    assert!(aspace_mul.lambdas().iter().all(|&l| l >= 0.0));

    // Optional: ensure smallest two λ are non-negative and the smallest is ~0 within tolerance
    // which is consistent with Laplacian PSD and having at least one zero eigenvalue.
    let ls = aspace_sum.lambdas();
    assert!(
        ls[0] >= -1e-12,
        "Smallest eigenvalue should be ≥ 0: {}",
        ls[0]
    );
    if ls.len() > 1 {
        assert!(
            ls[1] >= -1e-12,
            "Fiedler eigenvalue should be ≥ 0: {}",
            ls[1]
        );
    }
}
