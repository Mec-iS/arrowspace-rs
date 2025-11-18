//! A set of normalised vectors that have taumode value equals 0.0 if computed against their reference dataset
//! `ds = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")`

/// generate a make moon-like dataset
/// NOTE: use with `.with_normalisation(true)`
/// use to cover two cluster edge cases
pub fn make_moons_hd(
    n: usize,
    noise_xy: f64,
    noise_hd: f64,
    dims: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use std::f64::consts::PI;

    let mut rng = Pcg64::seed_from_u64(seed);
    let n0 = n / 2;
    let n1 = n - n0;
    let mut out = Vec::with_capacity(n);

    // Upper moon: (cos t, sin t) in first two dims
    for _ in 0..n0 {
        let t = rng.random::<f64>() * PI;
        let x0: f64 = t.cos() + noise_xy * rng.random::<f64>();
        let x1 = t.sin() + noise_xy * rng.random::<f64>();

        let mut v = vec![0.0_f64; dims];
        v[0] = x0;
        v[1] = x1;
        for d in 2..dims {
            v[d] = noise_hd * rng.random::<f64>();
        }
        out.push(v);
    }

    // Lower moon: (1 - cos t, -sin t - 0.5) in first two dims
    for _ in 0..n1 {
        let t = rng.random::<f64>() * PI;
        let x0 = 1.0 - t.cos() + noise_xy * rng.random::<f64>();
        let x1 = -t.sin() - 0.5 + noise_xy * rng.random::<f64>();

        let mut v = vec![0.0_f64; dims];
        v[0] = x0;
        v[1] = x1;
        for d in 2..dims {
            v[d] = noise_hd * rng.random::<f64>();
        }
        out.push(v);
    }

    out
}

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal, Uniform};

/// Generate Gaussian blob dataset with configurable noise and outliers
///
/// # Arguments
/// * `n_points` - Total number of points to generate
/// * `noise` - Standard deviation of Gaussian noise (controls cluster spread)
///
/// # Returns
/// Vector of n_points data points in 5D space, forming 3 clusters plus outliers
pub fn make_gaussian_blob(n_points: usize, noise: f64) -> Vec<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(789);
    let mut rows = Vec::new();

    let n_outliers = (n_points as f64 * 0.15).round() as usize;
    let n_cluster_points = n_points - n_outliers;
    let points_per_cluster = n_cluster_points / 3;

    // Define cluster centers in 10D space
    let centers = vec![
        vec![0.0; 10], // Cleaner than listing 10 zeros
        {
            let mut c = vec![0.0; 10];
            c[0] = 10.0;
            c
        },
        {
            let mut c = vec![0.0; 10];
            c[1] = 10.0;
            c
        },
    ];

    // Generate cluster points
    for center in &centers {
        for _ in 0..points_per_cluster {
            let mut point = Vec::new();
            for &c in center {
                let normal = Normal::new(c, noise).unwrap();
                point.push(normal.sample(&mut rng));
            }
            rows.push(point);
        }
    }

    // Generate outliers
    let outlier_dist = Uniform::new(-5.0, 15.0).unwrap();
    for _ in 0..n_outliers {
        let mut point = Vec::new();
        for _ in 0..10 {
            // ← Changed from 5 to 10
            point.push(outlier_dist.sample(&mut rng));
        }
        rows.push(point);
    }

    rows.shuffle(&mut rng);
    rows
}

pub fn make_gaussian_hd(n_points: usize, noise: f64) -> Vec<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(435);
    let mut rows = Vec::with_capacity(n_points);

    let n_outliers = (n_points as f64 * 0.15).round() as usize;
    let n_cluster_points = n_points - n_outliers;

    // Split cluster points across 3 clusters, distributing the remainder.
    let base = n_cluster_points / 3;
    let rem = n_cluster_points % 3;
    let cluster_sizes = [
        base + if rem > 0 { 1 } else { 0 },
        base + if rem > 1 { 1 } else { 0 },
        base,
    ];
    debug_assert_eq!(
        cluster_sizes.iter().sum::<usize>(),
        n_cluster_points,
        "cluster size split must match n_cluster_points"
    );

    // Define cluster centers in 100D space
    let centers = vec![
        vec![0.0; 100],
        {
            let mut c = vec![0.0; 100];
            c[0] = 10.0;
            c
        },
        {
            let mut c = vec![0.0; 100];
            c[1] = 10.0;
            c
        },
    ];

    // Generate cluster points
    for (cluster_idx, center) in centers.iter().enumerate() {
        let n_for_cluster = cluster_sizes[cluster_idx];
        for _ in 0..n_for_cluster {
            let mut point = Vec::with_capacity(100);
            for &c in center {
                let normal = Normal::new(c, noise).unwrap();
                point.push(normal.sample(&mut rng));
            }
            rows.push(point);
        }
    }

    // Generate outliers
    let outlier_dist = Uniform::new(-5.0, 15.0).unwrap();
    for _ in 0..n_outliers {
        let mut point = Vec::with_capacity(100);
        for _ in 0..100 {
            point.push(outlier_dist.sample(&mut rng));
        }
        rows.push(point);
    }

    // If rounding made us overshoot (theoretically shouldn't), trim.
    if rows.len() > n_points {
        rows.truncate(n_points);
    }
    // If for any reason we're short, top up with random outliers.
    while rows.len() < n_points {
        let mut point = Vec::with_capacity(100);
        for _ in 0..100 {
            point.push(outlier_dist.sample(&mut rng));
        }
        rows.push(point);
    }

    rows.shuffle(&mut rng);
    rows
}

/// Generates well-structured dataset for energy search testing
///
/// Creates 5 Gaussian clusters in 100D space with controlled separation
pub fn make_energy_test_dataset(n_items: usize, n_features: usize, seed: u64) -> Vec<Vec<f64>> {
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let n_clusters = 5;
    let items_per_cluster = n_items / n_clusters;

    let mut data = Vec::with_capacity(n_items);

    // Generate cluster centers with good separation
    let cluster_spacing = 10.0;
    for cluster_id in 0..n_clusters {
        // Cluster center: offset in first few dimensions
        let mut center = vec![0.0; n_features];
        center[0] = cluster_id as f64 * cluster_spacing;
        center[1] = (cluster_id % 2) as f64 * cluster_spacing;

        // Generate items around center
        for _ in 0..items_per_cluster {
            let mut item = vec![0.0; n_features];
            for j in 0..n_features {
                // Gaussian noise around center
                let noise: f64 = rng.random::<f64>() * 2.0 - 1.0; // [-1, 1]
                item[j] = center[j] + noise * 0.8;
            }
            data.push(item);
        }
    }

    // Add remaining items if n_items % n_clusters != 0
    let remaining = n_items % n_clusters;
    for _ in 0..remaining {
        data.push(
            (0..n_features)
                .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                .collect(),
        );
    }

    data
}

// 3 tight blobs of size 12 each + 15 outliers = 51 points, 10D
pub fn make_gaussian_cliques(
    n_per: usize,
    noise: f64,
    n_out: usize,
    dims: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, Uniform};
    use rand_pcg::Pcg64;
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut rows = Vec::with_capacity(3 * n_per + n_out);
    let centers = vec![
        {
            let mut c = vec![0.0; dims];
            c[0] = 10.0;
            c
        },
        {
            let mut c = vec![0.0; dims];
            c[1] = 10.0;
            c
        },
        {
            let mut c = vec![0.0; dims];
            c[0] = -10.0;
            c[1] = -10.0;
            c
        },
    ];
    for ctr in &centers {
        for _ in 0..n_per {
            let mut v = Vec::with_capacity(dims);
            for &m in ctr {
                let d = Normal::new(m, noise).unwrap();
                v.push(d.sample(&mut rng));
            }
            rows.push(v);
        }
    }
    // a few bridges close to midpoints between clusters to create small cuts
    for b in &[(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)] {
        let mut v = vec![0.0; dims];
        v[0] = 10.0 * b.0 - 10.0 * (1.0 - b.0);
        v[1] = 10.0 * b.1 - 10.0 * (1.0 - b.1);
        for d in 2..dims {
            v[d] = Normal::new(0.0, noise).unwrap().sample(&mut rng);
        }
        rows.push(v);
    }
    // outliers
    let uni = Uniform::new(-5.0, 15.0).unwrap();
    for _ in 0..n_out {
        let mut v = Vec::with_capacity(dims);
        for _ in 0..dims {
            v.push(uni.sample(&mut rng));
        }
        rows.push(v);
    }
    rows
}

/// Generate multiple Gaussian cliques with clear separation for motif detection.
///
/// # Parameters
/// - `n_points`: total number of points
/// - `noise`: intra-cluster Gaussian noise (lower = tighter clusters)
/// - `n_cliques`: number of distinct cliques/clusters
/// - `dims`: feature dimension
/// - `seed`: RNG seed
///
/// Returns a dataset with `n_cliques` well-separated clusters, suitable for
/// motif-based subgraph extraction.
pub fn make_gaussian_cliques_multi(
    n_points: usize,
    noise: f64,
    n_cliques: usize,
    dims: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut rows = Vec::with_capacity(n_points);

    // Add some outliers (5% of data) to make it realistic.
    let n_outliers = (n_points as f64 * 0.05).round() as usize;
    let n_cluster_points = n_points - n_outliers;

    // Distribute points evenly across cliques, with remainder in first clique.
    let base = n_cluster_points / n_cliques;
    let rem = n_cluster_points % n_cliques;

    // Generate clique centers in a grid layout for maximum separation.
    let grid_size = (n_cliques as f64).sqrt().ceil() as usize;
    let spacing = 20.0; // large spacing between cliques

    let mut clique_centers = Vec::new();
    for i in 0..n_cliques {
        let mut center = vec![0.0; dims];
        let grid_x = (i % grid_size) as f64;
        let grid_y = (i / grid_size) as f64;
        // Place clique centers along first two dimensions, rest at 0.
        center[0] = grid_x * spacing;
        if dims > 1 {
            center[1] = grid_y * spacing;
        }
        clique_centers.push(center);
    }

    // Generate points for each clique.
    for (clique_idx, center) in clique_centers.iter().enumerate() {
        let n_for_clique = base + if clique_idx < rem { 1 } else { 0 };

        for _ in 0..n_for_clique {
            let mut point = Vec::with_capacity(dims);
            for &c in center {
                let normal = Normal::new(c, noise).unwrap();
                point.push(normal.sample(&mut rng));
            }
            rows.push(point);
        }
    }

    // Generate outliers uniformly across the space.
    let outlier_dist = Uniform::new(-10.0, (grid_size as f64) * spacing + 10.0).unwrap();
    for _ in 0..n_outliers {
        let mut point = Vec::with_capacity(dims);
        for _ in 0..dims {
            point.push(outlier_dist.sample(&mut rng));
        }
        rows.push(point);
    }

    // Ensure exact count.
    if rows.len() > n_points {
        rows.truncate(n_points);
    }
    while rows.len() < n_points {
        let mut point = Vec::with_capacity(dims);
        for _ in 0..dims {
            point.push(outlier_dist.sample(&mut rng));
        }
        rows.push(point);
    }

    rows.shuffle(&mut rng);
    rows
}
