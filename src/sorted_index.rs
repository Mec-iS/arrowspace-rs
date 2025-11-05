use ordered_float::OrderedFloat;
use std::collections::BTreeMap;

/// Stores items ordered by their tau-mode lambda score.
/// - Keys are OrderedFloat(lambda) so scanning by lambda uses BTreeMap range.
/// - Values are buckets Vec<(idx, id)> sorted lexicographically by id to keep ties deterministic.
#[derive(Clone, Debug, Default)]
pub struct SortedLambdas {
    map: BTreeMap<OrderedFloat<f64>, Vec<(usize, String)>>,
    std_dev: f64,
}

impl SortedLambdas {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            std_dev: 0.0,
        }
    }

    /// Insert one item by lambda and stable string id; ties within the same lambda sorted by id.
    pub fn zadd(&mut self, lambda: f64, idx: usize, id: String) {
        let key = OrderedFloat(lambda);
        let bucket = self.map.entry(key).or_default();
        bucket.push((idx, id));
        // Keep deterministic order for ties
        bucket.sort_by(|a, b| a.1.cmp(&b.1));
    }

    /// Bulk build from per-item lambdas and parallel ids.
    /// Lambdas are assumed normalized to [0,1] upstream by ArrowSpace::normalise_lambdas.
    pub fn build_from(&mut self, lambdas: &[f64]) {
        if crate::laplacian::std_deviation(lambdas).is_some() {
            self.std_dev = crate::laplacian::std_deviation(lambdas).unwrap() as f64;
        } else {
            panic!(
                "Cannot compute proper standard deviations for lambdas, there was probably a problem with lambdas computation"
            )
        }
        for (i, &lam) in lambdas.iter().enumerate() {
            self.zadd(lam, i, i.to_string());
        }
    }

    /// Return values as vector
    pub fn to_vec(&self) -> Vec<(f64, usize)> {
        let mut out = Vec::new();
        for (k, bucket) in &self.map {
            for (idx, _id) in bucket {
                out.push((k.0, *idx));
            }
        }
        out
    }

    /// Return values as iterator
    pub fn iterator(&self) -> impl Iterator<Item = (f64, usize)> + '_ {
        self.map
            .iter()
            .flat_map(|(k, bucket)| bucket.iter().map(move |(idx, _id)| (k.0, *idx)))
    }

    /// Returns (idx, lambda). Lambda comes from the BTreeMap key.
    pub fn range_bylambda(&self, lambda_q: f64, k: usize, p: f64) -> Vec<(usize, f64)> {
        let band = self.std_dev / 2.0_f64.powf(p);
        let lo = OrderedFloat(lambda_q - band);
        let hi = OrderedFloat(lambda_q + band);

        let mut out: Vec<(usize, f64)> = Vec::new();

        for (key, bucket) in self.map.range(lo..=hi) {
            for (idx, _id) in bucket {
                out.push((*idx, key.0));
            }
        }
        if out.len() >= k {
            return out[0..k].to_vec();
        }
        out.to_vec()
    }

    /// k nearest-by-lambda around lambda_q using an expanding window.
    /// Initial window is derived from std_dev * p unless base_delta is provided.
    /// Results are trimmed to the k smallest |λ - λ_q|; ties inside buckets remain id-stable.
    pub fn k_nearest_by_lambda(
        &self,
        lambda_q: f64,
        k: usize,
        lambda_p: f64,
        base_delta: Option<f64>,
        growth: f64,         // e.g., 1.7
        max_multiplier: f64, // e.g., 10.0
    ) -> Vec<(usize, f64, String)> {
        if k == 0 || self.map.is_empty() {
            return Vec::new();
        }
        let mut delta = base_delta
            .unwrap_or_else(|| (self.std_dev * lambda_p).max(1e-9))
            .abs();

        let growth = if growth.is_finite() && growth > 1.0 {
            growth
        } else {
            1.7
        };
        let max_delta = (delta * max_multiplier.max(1.0)).min(1.0);

        let mut candidates: Vec<(usize, f64, String)> = Vec::new();

        loop {
            let lo = (lambda_q - delta).max(0.0);
            let hi = (lambda_q + delta).min(1.0);

            candidates.clear();
            for (k_lambda, bucket) in self.map.range(OrderedFloat(lo)..=OrderedFloat(hi)) {
                // Preserve stable id order inside each equal-lambda bucket
                for (idx, id) in bucket {
                    candidates.push((*idx, k_lambda.0, id.clone()));
                }
            }

            if candidates.len() >= k || delta >= max_delta {
                break;
            }
            delta = (delta * growth).min(max_delta);
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        // Sort by absolute lambda distance; for exact ties, stable id order is already baked in
        candidates.sort_unstable_by(|a, b| {
            let da = (a.1 - lambda_q).abs();
            let db = (b.1 - lambda_q).abs();
            da.partial_cmp(&db).unwrap()
        });
        candidates.truncate(k);
        candidates
    }
}
