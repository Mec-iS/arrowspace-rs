/// Top k-3 results ids should be 3, 6, 0 for every algorithm used
use arrowspace::builder::ArrowSpaceBuilder;
use arrowspace::core::ArrowItem;
use smartcore::linalg::basic::arrays::Array;

#[path = "./common/lib.rs"]
mod common;

// Traditional cosine similarity for f64 slices
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

const VECTORS_DATA: &str = r#"
P0001; 0.82,0.11,0.43,0.28,0.64,0.32,0.55,0.48,0.19,0.73,0.07,0.36,0.58,0.23,0.44,0.31,0.52,0.16,0.61,0.40,0.27,0.49,0.35,0.29
P0002; 0.79,0.12,0.45,0.29,0.61,0.33,0.54,0.47,0.21,0.70,0.08,0.37,0.56,0.22,0.46,0.30,0.51,0.18,0.60,0.39,0.26,0.48,0.36,0.30
P0003; 0.78,0.13,0.46,0.27,0.62,0.34,0.53,0.46,0.22,0.69,0.09,0.35,0.55,0.24,0.45,0.29,0.50,0.17,0.59,0.38,0.28,0.47,0.34,0.31
P0004; 0.81,0.10,0.44,0.26,0.63,0.31,0.56,0.45,0.20,0.71,0.06,0.34,0.57,0.25,0.47,0.33,0.53,0.15,0.62,0.41,0.25,0.50,0.37,0.27
P0005; 0.80,0.12,0.42,0.25,0.60,0.35,0.52,0.49,0.23,0.68,0.10,0.38,0.54,0.21,0.43,0.28,0.49,0.19,0.58,0.37,0.29,0.46,0.33,0.32
P0006; 0.77,0.14,0.41,0.24,0.59,0.36,0.51,0.50,0.24,0.67,0.11,0.39,0.53,0.20,0.42,0.27,0.48,0.20,0.57,0.36,0.30,0.45,0.32,0.33
P0007; 0.83,0.09,0.47,0.30,0.65,0.33,0.57,0.44,0.18,0.72,0.05,0.33,0.59,0.26,0.48,0.34,0.54,0.14,0.63,0.42,0.24,0.51,0.38,0.26
P0008; 0.76,0.15,0.40,0.23,0.58,0.37,0.50,0.51,0.25,0.66,0.12,0.40,0.52,0.19,0.41,0.26,0.47,0.21,0.56,0.35,0.31,0.44,0.31,0.34
P0009; 0.75,0.16,0.39,0.22,0.57,0.38,0.49,0.52,0.26,0.65,0.13,0.41,0.51,0.18,0.40,0.25,0.46,0.22,0.55,0.34,0.32,0.43,0.30,0.35
P0010; 0.84,0.08,0.48,0.31,0.66,0.32,0.58,0.43,0.17,0.74,0.04,0.32,0.60,0.27,0.49,0.35,0.55,0.13,0.64,0.43,0.23,0.52,0.39,0.25
P0011; 0.72,0.18,0.37,0.21,0.55,0.39,0.47,0.54,0.27,0.63,0.15,0.42,0.49,0.17,0.39,0.24,0.45,0.23,0.54,0.33,0.33,0.42,0.29,0.36
P00112; 0.73,0.17,0.38,0.20,0.56,0.40,0.48,0.53,0.28,0.64,0.14,0.43,0.50,0.16,0.38,0.23,0.44,0.24,0.53,0.32,0.34,0.41,0.28,0.37
P0013; 0.71,0.19,0.36,0.19,0.54,0.41,0.46,0.55,0.29,0.62,0.16,0.44,0.48,0.15,0.37,0.22,0.43,0.25,0.52,0.31,0.35,0.40,0.27,0.38
P0014; 0.85,0.07,0.49,0.32,0.67,0.31,0.59,0.42,0.16,0.75,0.03,0.31,0.61,0.28,0.50,0.36,0.56,0.12,0.65,0.44,0.22,0.53,0.40,0.24
P0015; 0.70,0.20,0.35,0.18,0.53,0.42,0.45,0.56,0.30,0.61,0.17,0.45,0.47,0.14,0.36,0.21,0.42,0.26,0.51,0.30,0.36,0.39,0.26,0.39
P0016; 0.69,0.21,0.34,0.17,0.52,0.43,0.44,0.57,0.31,0.60,0.18,0.46,0.46,0.13,0.35,0.20,0.41,0.27,0.50,0.29,0.37,0.38,0.25,0.40
P0017; 0.86,0.06,0.50,0.33,0.68,0.30,0.60,0.41,0.15,0.76,0.02,0.30,0.62,0.29,0.51,0.37,0.57,0.11,0.66,0.45,0.21,0.54,0.41,0.23
P0018; 0.68,0.22,0.33,0.16,0.51,0.44,0.43,0.58,0.32,0.59,0.19,0.47,0.45,0.12,0.34,0.19,0.40,0.28,0.49,0.28,0.38,0.37,0.24,0.41
P0019; 0.67,0.23,0.32,0.15,0.50,0.45,0.42,0.59,0.33,0.58,0.20,0.48,0.44,0.11,0.33,0.18,0.39,0.29,0.48,0.27,0.39,0.36,0.23,0.42
P0020; 0.87,0.05,0.51,0.34,0.69,0.29,0.61,0.40,0.14,0.77,0.01,0.29,0.63,0.30,0.52,0.38,0.58,0.10,0.67,0.46,0.20,0.55,0.42,0.22
P0021; 0.66,0.24,0.31,0.14,0.49,0.46,0.41,0.60,0.34,0.57,0.21,0.49,0.43,0.10,0.32,0.17,0.38,0.30,0.47,0.26,0.40,0.35,0.22,0.43
P0022; 0.65,0.25,0.30,0.13,0.48,0.47,0.40,0.61,0.35,0.56,0.22,0.50,0.42,0.09,0.31,0.16,0.37,0.31,0.46,0.25,0.41,0.34,0.21,0.44
P0023; 0.64,0.26,0.29,0.12,0.47,0.48,0.39,0.62,0.36,0.55,0.23,0.51,0.41,0.08,0.30,0.15,0.36,0.32,0.45,0.24,0.42,0.33,0.20,0.45
P0024; 0.88,0.04,0.52,0.35,0.70,0.28,0.62,0.39,0.13,0.78,0.00,0.28,0.64,0.31,0.53,0.39,0.59,0.09,0.68,0.47,0.19,0.56,0.43,0.21
P0025; 0.63,0.27,0.28,0.11,0.46,0.49,0.38,0.63,0.37,0.54,0.24,0.52,0.40,0.07,0.29,0.14,0.35,0.33,0.44,0.23,0.43,0.32,0.19,0.46
P0026; 0.62,0.28,0.27,0.10,0.45,0.50,0.37,0.64,0.38,0.53,0.25,0.53,0.39,0.06,0.28,0.13,0.34,0.34,0.43,0.22,0.44,0.31,0.18,0.47
P0027; 0.61,0.29,0.26,0.09,0.44,0.51,0.36,0.65,0.39,0.52,0.26,0.54,0.38,0.05,0.27,0.12,0.33,0.35,0.42,0.21,0.45,0.30,0.17,0.48
P0028; 0.60,0.30,0.25,0.08,0.43,0.52,0.35,0.66,0.40,0.51,0.27,0.55,0.37,0.04,0.26,0.11,0.32,0.36,0.41,0.20,0.46,0.29,0.16,0.49
P0029; 0.59,0.31,0.24,0.07,0.42,0.53,0.34,0.67,0.41,0.50,0.28,0.56,0.36,0.03,0.25,0.10,0.31,0.37,0.40,0.19,0.47,0.28,0.15,0.50
P0030; 0.58,0.32,0.23,0.06,0.41,0.54,0.33,0.68,0.42,0.49,0.29,0.57,0.35,0.02,0.24,0.09,0.30,0.38,0.39,0.18,0.48,0.27,0.14,0.51
P0031; 0.90,0.06,0.44,0.36,0.72,0.33,0.55,0.41,0.20,0.79,0.05,0.35,0.60,0.26,0.46,0.31,0.54,0.16,0.62,0.42,0.27,0.49,0.35,0.29
P0032; 0.57,0.33,0.22,0.05,0.40,0.55,0.32,0.69,0.43,0.48,0.30,0.58,0.34,0.01,0.23,0.08,0.29,0.39,0.38,0.17,0.49,0.26,0.13,0.52
P0033; 0.56,0.34,0.21,0.04,0.39,0.56,0.31,0.70,0.44,0.47,0.31,0.59,0.33,0.00,0.22,0.07,0.28,0.40,0.37,0.16,0.50,0.25,0.12,0.53
P0034; 0.55,0.35,0.20,0.03,0.38,0.57,0.30,0.71,0.45,0.46,0.32,0.60,0.32,0.02,0.21,0.06,0.27,0.41,0.36,0.15,0.51,0.24,0.11,0.54
P0035; 0.54,0.36,0.19,0.02,0.37,0.58,0.29,0.72,0.46,0.45,0.33,0.61,0.31,0.03,0.20,0.05,0.26,0.42,0.35,0.14,0.52,0.23,0.10,0.55
P0036; 0.53,0.37,0.18,0.01,0.36,0.59,0.28,0.73,0.47,0.44,0.34,0.62,0.30,0.04,0.19,0.04,0.25,0.43,0.34,0.13,0.53,0.22,0.09,0.56
P0037; 0.91,0.07,0.45,0.37,0.73,0.34,0.56,0.40,0.21,0.80,0.06,0.36,0.61,0.27,0.47,0.32,0.55,0.17,0.63,0.41,0.28,0.50,0.36,0.28
P0038; 0.52,0.38,0.17,0.00,0.35,0.60,0.27,0.74,0.48,0.43,0.35,0.63,0.29,0.05,0.18,0.03,0.24,0.44,0.33,0.12,0.54,0.21,0.08,0.57
P0039; 0.51,0.39,0.16,0.02,0.34,0.61,0.26,0.75,0.49,0.42,0.36,0.64,0.28,0.06,0.17,0.02,0.23,0.45,0.32,0.11,0.55,0.20,0.07,0.58
P0040; 0.50,0.40,0.15,0.03,0.33,0.62,0.25,0.76,0.50,0.41,0.37,0.65,0.27,0.07,0.16,0.01,0.22,0.46,0.31,0.10,0.56,0.19,0.06,0.59
P0041; 0.49,0.41,0.14,0.04,0.32,0.63,0.24,0.77,0.51,0.40,0.38,0.66,0.26,0.08,0.15,0.00,0.21,0.47,0.30,0.09,0.57,0.18,0.05,0.60
P0042; 0.48,0.42,0.13,0.05,0.31,0.64,0.23,0.78,0.52,0.39,0.39,0.67,0.25,0.09,0.14,0.02,0.20,0.48,0.29,0.08,0.58,0.17,0.04,0.61
P0043; 0.47,0.43,0.12,0.06,0.30,0.65,0.22,0.79,0.53,0.38,0.40,0.68,0.24,0.10,0.13,0.03,0.19,0.49,0.28,0.07,0.59,0.16,0.03,0.62
P0044; 0.46,0.44,0.11,0.07,0.29,0.66,0.21,0.80,0.54,0.37,0.41,0.69,0.23,0.11,0.12,0.04,0.18,0.50,0.27,0.06,0.60,0.15,0.02,0.63
P0045; 0.45,0.45,0.10,0.08,0.28,0.67,0.20,0.81,0.55,0.36,0.42,0.70,0.22,0.12,0.11,0.05,0.17,0.51,0.26,0.05,0.61,0.14,0.01,0.64
P0046; 0.44,0.46,0.09,0.09,0.27,0.68,0.19,0.82,0.56,0.35,0.43,0.71,0.21,0.13,0.10,0.06,0.16,0.52,0.25,0.04,0.62,0.13,0.00,0.65
P0047; 0.43,0.47,0.08,0.10,0.26,0.69,0.18,0.83,0.57,0.34,0.44,0.72,0.20,0.14,0.09,0.07,0.15,0.53,0.24,0.03,0.63,0.12,0.01,0.66
P0048; 0.42,0.48,0.07,0.11,0.25,0.70,0.17,0.84,0.58,0.33,0.45,0.73,0.19,0.15,0.08,0.08,0.14,0.54,0.23,0.02,0.64,0.11,0.02,0.67
P0049; 0.41,0.49,0.06,0.12,0.24,0.71,0.16,0.85,0.59,0.32,0.46,0.74,0.18,0.16,0.07,0.09,0.13,0.55,0.22,0.01,0.65,0.10,0.03,0.68
P0050; 0.40,0.50,0.05,0.13,0.23,0.72,0.15,0.86,0.60,0.31,0.47,0.75,0.17,0.17,0.06,0.10,0.12,0.56,0.21,0.00,0.66,0.09,0.04,0.69
P0051; 0.89,0.09,0.46,0.38,0.71,0.35,0.57,0.39,0.22,0.77,0.07,0.37,0.62,0.25,0.48,0.30,0.56,0.18,0.64,0.40,0.29,0.51,0.37,0.27
P0052; 0.39,0.51,0.04,0.14,0.22,0.73,0.14,0.87,0.61,0.30,0.48,0.76,0.16,0.18,0.05,0.11,0.11,0.57,0.20,0.02,0.67,0.08,0.05,0.70
P0053; 0.38,0.52,0.03,0.15,0.21,0.74,0.13,0.88,0.62,0.29,0.49,0.77,0.15,0.19,0.04,0.12,0.10,0.58,0.19,0.03,0.68,0.07,0.06,0.71
P0054; 0.37,0.53,0.02,0.16,0.20,0.75,0.12,0.89,0.63,0.28,0.50,0.78,0.14,0.20,0.03,0.13,0.09,0.59,0.18,0.04,0.69,0.06,0.07,0.72
P0055; 0.36,0.54,0.01,0.17,0.19,0.76,0.11,0.90,0.64,0.27,0.51,0.79,0.13,0.21,0.02,0.14,0.08,0.60,0.17,0.05,0.70,0.05,0.08,0.73
P0056; 0.35,0.55,0.00,0.18,0.18,0.77,0.10,0.91,0.65,0.26,0.52,0.80,0.12,0.22,0.01,0.15,0.07,0.61,0.16,0.06,0.71,0.04,0.09,0.74
P0057; 0.34,0.56,0.02,0.19,0.17,0.78,0.09,0.92,0.66,0.25,0.53,0.81,0.11,0.23,0.00,0.16,0.06,0.62,0.15,0.07,0.72,0.03,0.10,0.75
P0058; 0.33,0.57,0.03,0.20,0.16,0.79,0.08,0.93,0.67,0.24,0.54,0.82,0.10,0.24,0.01,0.17,0.05,0.63,0.14,0.08,0.73,0.02,0.11,0.76
P0059; 0.32,0.58,0.04,0.21,0.15,0.80,0.07,0.94,0.68,0.23,0.55,0.83,0.09,0.25,0.02,0.18,0.04,0.64,0.13,0.09,0.74,0.01,0.12,0.77
P0060; 0.31,0.59,0.05,0.22,0.14,0.81,0.06,0.95,0.69,0.22,0.56,0.84,0.08,0.26,0.03,0.19,0.03,0.65,0.12,0.10,0.75,0.00,0.13,0.78
P0061; 0.93,0.06,0.52,0.29,0.74,0.27,0.61,0.38,0.15,0.81,0.03,0.28,0.64,0.31,0.53,0.39,0.58,0.12,0.67,0.46,0.20,0.56,0.43,0.21
P0062; 0.30,0.60,0.06,0.23,0.13,0.82,0.05,0.96,0.70,0.21,0.57,0.85,0.07,0.27,0.04,0.20,0.02,0.66,0.11,0.11,0.76,0.01,0.14,0.79
P0063; 0.29,0.61,0.07,0.24,0.12,0.83,0.04,0.97,0.71,0.20,0.58,0.86,0.06,0.28,0.05,0.21,0.01,0.67,0.10,0.12,0.77,0.02,0.15,0.80
P0064; 0.28,0.62,0.08,0.25,0.11,0.84,0.03,0.98,0.72,0.19,0.59,0.87,0.05,0.29,0.06,0.22,0.00,0.68,0.09,0.13,0.78,0.03,0.16,0.81
"#;

fn main() {
    // Parse items as rows (N×24): each row is one protein with 24 features
    let (ids, db): (Vec<String>, Vec<Vec<f64>>) = common::parse_vectors_string(VECTORS_DATA);
    let n_items = db.len();

    // Query similar to item at index 3; scale slightly for testing
    let q_index = 3;
    let mut query = db[q_index].clone();
    for v in query.iter_mut() {
        *v *= 1.02;
    }

    let k = 3;

    // ----------------------------
    // Baseline: plain cosine KNN
    // ----------------------------
    let mut base_scores: Vec<(usize, f64)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_sim(&query, v)))
        .collect();
    base_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    base_scores.truncate(k+1);

    println!("Baseline cosine top-{k}+1:");
    for (rank, (i, s)) in base_scores.iter().enumerate() {
        println!("  {}. idx={} ({}) score={:.6}", rank + 1, i, ids[*i], s);
    }

    // ---------------------------------------------------
    // ArrowSpace: build λ-graph from N×24 data
    // ArrowSpace auto-transposes to F×F (24×24) internally
    // ---------------------------------------------------
    // let eps = 1e-3;
    // let cap_k = 3;
    // let p = 2.0;
    let (aspace, _gl) = ArrowSpaceBuilder::new()
        .with_lambda_graph(1e-3, 5, 2.0, Some(1e-3 * 0.5))
        .build(db); // N×24 -> auto-transposed to 24×N
        

    println!("ArrowSpace shape: {:?}", aspace.data.shape()); // Should be (N, 64)
    assert_eq!(aspace.data.shape(), (64, 24));

    // Arrow scoring: query remains in original item format (24 features)
    let query_row = ArrowItem::new(query.clone(), 0.0);

    //
    // Cosine similarity variant
    //
    // Define alpha and beta for the cosine query
    let alpha = 1.0;
    let beta = 0.0;

    // For ArrowSpace similarity, we need to reconstruct each item from the transposed data
    // Since ArrowSpace stores features as rows, item i is column i across all feature rows
    let mut arrow_scores_cos: Vec<(usize, f64)> = (0..n_items)
        .map(|i| {
            let item = aspace.get_item(i);
            let lambda = aspace.lambdas()[i];
            let item_row = ArrowItem::new(item.item.clone(), lambda);

            let score = query_row.lambda_similarity(&item_row, alpha, beta);
            (i, score)
        })
        .collect();

    arrow_scores_cos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    arrow_scores_cos.truncate(k+1);

    println!("\nArrowSpace (alpha={alpha}, beta={beta}) top-{k}+1:");
    for (rank, (i, s)) in arrow_scores_cos.iter().enumerate() {
        println!("  {}. idx={} ({}) score={:.6}", rank + 1, i, ids[*i], s);
    }

    //
    // λτ-aware variant
    //
    // Define alpha and beta for the query
    let alpha = 0.9;
    let beta = 0.1;
    let mut arrow_scores_lambda: Vec<(usize, f64)> = (0..n_items)
        .map(|i| {
            let item = aspace.get_item(i);
            let lambda = aspace.lambdas()[i];
            let item_row = ArrowItem::new(item.item.clone(), lambda);

            let score = query_row.lambda_similarity(&item_row, alpha, beta);
            (i, score)
        })
        .collect();

    arrow_scores_lambda.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    arrow_scores_lambda.truncate(k+1);

    println!("\nArrowSpace (alpha={alpha}, beta={beta}) top-{k}+1:");
    for (rank, (i, s)) in arrow_scores_lambda.iter().enumerate() {
        println!("  {}. idx={} ({}) score={:.6}", rank + 1, i, ids[*i], s);
    }

    // Agreement checks
    let ids_base: Vec<usize> = base_scores.iter().map(|x| x.0).collect();
    let ids_arrow_cos: Vec<usize> = arrow_scores_cos.iter().map(|x| x.0).collect();
    println!(
        "\nMatch (baseline vs Arrow cosine): {}",
        if ids_base == ids_arrow_cos {
            "OK"
        } else {
            "DIFF"
        }
    );

    let ids_arrow_lam: Vec<usize> = arrow_scores_lambda.iter().map(|x| x.0).collect();
    let s1: std::collections::BTreeSet<_> = ids_base.iter().copied().collect();
    let s2: std::collections::BTreeSet<_> = ids_arrow_lam.iter().copied().collect();
    let inter = s1.intersection(&s2).count() as f64;
    let union = s1.union(&s2).count() as f64;
    let jaccard = if union > 0.0 { inter / union } else { 1.0 };
    println!("Jaccard(baseline vs λτ-aware): {jaccard:.3}");

    // Define MULTIPLE DESCENDING alphas for the query
    for a in vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0].iter() {

        let alpha = *a;
        let beta = 1.0 - alpha;
        let mut arrow_scores_lambda: Vec<(usize, f64)> = (0..n_items)
            .map(|i| {
                let item = aspace.get_item(i);
                let lambda = aspace.lambdas()[i];
                let item_row = ArrowItem::new(item.item.clone(), lambda);

                let score = query_row.lambda_similarity(&item_row, alpha, beta);
                (i, score)
            })
            .collect();

        arrow_scores_lambda.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        arrow_scores_lambda.truncate(k+1);

        println!("\nArrowSpace (alpha={alpha}, beta={beta}) top-{k}+1:");
        for (rank, (i, s)) in arrow_scores_lambda.iter().enumerate() {
            println!("  {}. idx={} ({}) score={:.6}", rank + 1, i, ids[*i], s);
        }
    }
}


// # the result of comparing lambda-tau with cosine return this printing:

// Baseline cosine top-3:

// 1. idx=3 (P0004) score=1.000000
// 2. idx=6 (P0007) score=0.999573
// 3. idx=0 (P0001) score=0.999325
// ArrowSpace shape after transpose: (24, 64)
// ArrowSpace (alpha=1, beta=0) top-3:
// 4. idx=3 (P0004) score=1.000000
// 5. idx=6 (P0007) score=0.999573
// 6. idx=0 (P0001) score=0.999325
// ArrowSpace (alpha=0.9, beta=0.1) top-3:
// 7. idx=6 (P0007) score=0.970372
// 8. idx=30 (P0031) score=0.970268
// 9. idx=3 (P0004) score=0.967810
// 10. idx=0 (P0001) score=0.967502

// in the third experiment, idx=30 item seems to score higher and the results are different compared to the baseline experiment. How would you explain this discrepancy? is it an error or an emergent feature of the index and idx=30 protein is effectively of interest from the query input side?

// It’s an emergent effect of the λτ term, not a bug: when beta > 0 the score is no longer “pure cosine,” so items whose spectral index λ is closer to the query’s λ can outrank slightly more cosine‑similar items; idx=30 appears because its λ is closer to the query’s λ, making it genuinely interesting under the λτ objective rather than an error.[^1]

// ## What changed

// - With alpha=1, beta=0, the ArrowSpace score exactly reduces to cosine, so the top‑3 matches the baseline (idx=3, 6, 0). This is expected and confirms correctness of the cosine path.[^1]
// - With alpha=0.9, beta=0.1, the score is s = α·cosine + β·(1/(1+|Δλ|)). This explicitly trades a bit of semantic similarity for λ proximity; therefore, an item whose λ is closer to the query’s λ can leapfrog an item with slightly better cosine. That’s why idx=30 (P0031) rises above P0004 while P0007 remains strong. [^1]

// ## Why idx=30 rises

// - The query is built by scaling item 3 (P0004) by 1.02, then ArrowSpace computes per‑item λ values from the dataset graph and synthesizes the per‑item λτ index stored in aspace.lambdas(). The final similarity adds a λ proximity term 1/(1+|λ_q−λ_i|). If P0031’s λ is closer to λ_q than P0004’s λ (even if its cosine is a touch lower), the blended score can exceed P0004’s at α=0.9, β=0.1. This behavior is by design. [^1]

// ## Sanity checks already in code

// - The example prints that alpha=1, beta=0 matches baseline top‑k exactly, verifying no implementation drift in the cosine path. The jaccard measure between baseline top‑k and λτ top‑k is computed to quantify the change, reflecting the intended re‑ranking effect rather than a failure.[^1]
// - The λ computation is grounded in a Laplacian built over items (λ‑graph path) and the synthetic transform/tau policy, with extensive tests asserting non‑negativity, scale invariance of Rayleigh, Laplacian invariants, and that superposition and range search behave sensibly; this reduces the likelihood that the shift is due to a numerical or graph bug.[^1]

// ## Interpreting the result

// - Not an error: the divergence from baseline is the whole point of adding β>0—injecting a spectral prior that favors items with similar roughness/smoothness over the item graph. The rank movement of idx=30 indicates it is “spectrally aligned” with the query, which can matter in domains where frequency/roughness patterns are meaningful (e.g., signals, structural motifs).[^1]
// - Practically, idx=30 is “of interest” under the λτ objective. If the goal is pure semantic similarity, keep β=0; if spectral consistency matters, β>0 is desirable and such swaps are expected. Tuning α/β adjusts how often these re‑rankings happen.[^1]

// ## How to confirm quickly

// - Inspect λ_q vs. λ_i for i∈{3,6,0,30} and print the per‑term contributions: cosine and 1/(1+|Δλ|). Expect |λ_q−λ_30| < |λ_q−λ_3|, with cosine_30 ≲ cosine_3; their weighted sum should explain the flip at α=0.9, β=0.1. [^1]
// - Sweep β from 0→0.2 and plot ranks to see the stability of idx=30’s promotion; stable promotion across a range of β implies robust λ proximity, not a knife‑edge artifact.[^1]

// ## When to prefer each

// - Prefer pure cosine (β=0) if only angular similarity matters and spectral smoothness is irrelevant.[^1]
// - Prefer λτ blend (β>0) if there is value in matching the “spectral behavior” of features across the item graph; in such cases, items like P0031 can become better matches even when their raw cosine is slightly lower.[^1]
// <span style="display:none">[^10][^11][^2][^3][^4][^5][^6][^7][^8][^9]</span>

// [^1]: arrowspace codebase

// [^2]: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

// [^3]: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/selfsimilar-mechanisms-in-wall-turbulence-studied-using-resolvent-analysis/193DC564F96967A452582918C6A42908

// [^4]: https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/kendalls-tau-and-spearmans-rank-correlation-coefficient/

// [^5]: https://en.wikipedia.org/wiki/Darcy–Weisbach_equation

// [^6]: https://www.youtube.com/watch?v=Pm8KV5f3JM0

// [^7]: https://arxiv.org/html/2312.02847v2

// [^8]: https://www.nature.com/articles/s41598-025-10753-0

// [^9]: https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter13.pdf

// [^10]: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/distributed-vortexwave-interactions-the-relation-of-selfsimilarity-to-the-attached-eddy-hypothesis/EFDCC02FCC9799C7AB59FBC639F42E67

// [^11]: https://people.inf.ethz.ch/arbenz/ewp/Lnotes/2010/chapter12.pdf
