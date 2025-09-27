//! Arrow and ArrowSpace: enhanced with search-specific zero-copy operations.
//!
//! This module provides two core abstractions for working with row-major numeric
//! data in search/graph contexts:
//!
//! - ArrowItem: an owned row with convenience methods (norm, dot, cosine_similarity,
//!   lambda-aware similarity), in-place arithmetic, and iterator access.
//! - ArrowSpace: a dense, row-major, zero-copy container of rows with per-row
//!   spectral score `lambda`, supporting row views (immutable/mutable), iteration,
//!   and search utilities.
//!
//! Design goals:
//! - Zero-copy access to rows for performance-critical routines.
//! - Iterator-first APIs for cache-friendly, allocation-free operations.
//! - Spectral-aware scoring via Rayleigh quotient against a Graph Laplacian.
//!
//! # Examples
//!
//! Create a small ArrowSpace and compute cosine similarity between rows:
//!
//! ```
//! use arrowspace::core::{ArrowItem, ArrowSpace};
//!
//! let aspace = ArrowSpace::from_items_default(
//!     vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]
//! );
//!
//! let a = aspace.get_feature(0);
//! let b = aspace.get_feature(1);
//! ```
//!
//! Zero-copy mutate a row using a mutable view and update its lambda from a graph:
//!
//!
//! Run documentation tests with `cargo test --doc`; Rustdoc extracts code blocks
//! and executes them as tests, ensuring examples stay correct over time[3][6].
//!
//! # Panics
//!
//! - Indexing functions panic on out-of-bounds row/column indices.
//! - Arithmetic between mismatched row lengths panics.
//!
//! # Performance
//!
//! - Row accessors favor zero-copy slices/views; prefer `row_view`/`row_view_mut`
//!   over `get_row` when allocation must be avoided.
//! - Batch operations rely on iterators to minimize bounds checks and enable
//!   vectorization opportunities.
//!
//! # Testing examples
//!
//! Rustdoc preprocesses examples: it injects the crate, wraps code in `fn main`
//! if missing, and allows common lints to reduce boilerplate[3][11]. Keep examples
//! small and focused; add hidden setup lines with `#` when needed so that examples
//! compile while showing only the essential lines to readers[3][4][8].

use std::fmt::Debug;

use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::graph::GraphLaplacian;
use crate::taumode::TauMode;

// Add logging
use log::{debug, info, trace, warn};

/// A single owned row with an associated spectral score `lambda`.
///
/// ArrowItem provides iterator-based, allocation-free primitives (norm, dot,
/// cosine similarity, Euclidean distance) and in-place arithmetic. It is useful
/// both as a convenience handle returned by `ArrowSpace::get_row` and as a
/// standalone value in query-time computations.
///
/// # Examples
///
/// Construct, compute similarity, and scale in place:
///
/// ```
/// use arrowspace::core::ArrowItem;
///
/// let mut a = ArrowItem::new(vec![1.0, 2.0, 3.0], 0.5);
/// let b = ArrowItem::new(vec![1.0, 0.0, 1.0], 1.2);
///
/// let cos = a.cosine_similarity(&b);
/// assert!(cos.is_finite());
///
/// a.scale(2.0);
/// assert_eq!(a.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct ArrowItem {
    pub item: Vec<f64>,
    pub lambda: f64,
}

/// A structure representing a feature-column
///  just the data for now but will be useful for index building
#[derive(Clone, Debug)]
pub struct ArrowFeature {
    pub feature: Vec<f64>,
}

impl ArrowItem {
    /// Creates a new ArrowItem from owned data.
    /// This just store the vector with a placeholder lambda, to compute the
    ///  lambda (Rayleigh quotient) use `new_with_graph` or precompute lambda
    ///  and pass it to this method.
    ///
    /// Prefer passing already-allocated vectors to avoid extra copies.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let r = ArrowItem::new(vec![0.0, 1.0], 0.3);
    /// assert_eq!(r.len(), 2);
    /// ```
    #[inline]
    pub fn new(item: Vec<f64>, lambda: f64) -> Self {
        trace!(
            "Creating ArrowItem with {} dimensions, lambda: {:.6}",
            item.len(),
            lambda
        );
        Self { item, lambda }
    }

    /// Returns the length (dimensionality) of the row.
    #[inline]
    pub fn len(&self) -> usize {
        self.item.len()
    }

    /// Returns true if the row has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.item.is_empty()
    }

    /// Computes the dot product with another row without allocating.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 2.0, 3.0], 0.0);
    /// let b = ArrowItem::new(vec![4.0, 5.0, 6.0], 0.0);
    /// assert_eq!(a.dot(&b), 32.0);
    /// ```
    #[inline]
    pub fn dot(&self, other: &ArrowItem) -> f64 {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        let result = self.item.iter().zip(other.item.iter()).map(|(a, b)| a * b).sum();
        trace!("Computed dot product: {:.6}", result);
        result
    }

    /// Computes the Euclidean norm (L2) without allocating.
    #[inline]
    pub fn norm(a: &[f64]) -> f64 {
        let result = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
        trace!("Computed norm: {:.6}", result);
        result
    }

    /// Computes cosine similarity, guarding against zero vectors.
    ///
    /// Returns 0.0 if either vector has zero norm.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 0.0], 0.0);
    /// let b = ArrowItem::new(vec![0.0, 1.0], 0.0);
    /// assert!((a.cosine_similarity(&b) - 0.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn cosine_similarity(&self, other: &ArrowItem) -> f64 {
        let denom = ArrowItem::norm(&self.item) * ArrowItem::norm(&other.item);
        let result = if denom > 0.0 {
            self.dot(other) / denom
        } else {
            warn!("Zero vector encountered in cosine similarity computation");
            0.0
        };
        trace!("Computed cosine similarity: {:.6}", result);
        result
    }

    /// Computes Euclidean distance without allocation.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 1.0], 0.0);
    /// let b = ArrowItem::new(vec![4.0, 5.0], 0.0);
    /// assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn euclidean_distance(&self, other: &ArrowItem) -> f64 {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        let result = self
            .item
            .iter()
            .zip(other.item.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        trace!("Computed Euclidean distance: {:.6}", result);
        result
    }

    /// Combines semantic (cosine) similarity and lambda proximity.
    ///
    /// `alpha` weights semantic similarity; `beta` weights lambda proximity
    /// defined as `1 / (1 + |lambda_a - lambda_b|)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::ArrowItem;
    /// let a = ArrowItem::new(vec![1.0, 0.0], 0.5);
    /// let b = ArrowItem::new(vec![1.0, 0.0], 0.6);
    /// let s = a.lambda_similarity(&b, 0.7, 0.3);
    /// assert!(s <= 1.0 && s >= 0.0);
    /// ```
    #[inline]
    pub fn lambda_similarity(&self, other: &ArrowItem, alpha: f64, beta: f64) -> f64 {
        assert_eq!(
            self.item.len(),
            other.item.len(),
            "items should be of the same length"
        );
        let semantic_sim = self.cosine_similarity(other);
        let lambda_sim = 1.0 / (1.0 + (self.lambda - other.lambda).abs());
        let result = alpha * semantic_sim + beta * lambda_sim;
        trace!(
            "Lambda similarity: semantic={:.6}, lambda={:.6}, combined={:.6}",
            semantic_sim,
            lambda_sim,
            result
        );
        result
    }

    /// Adds another row element-wise in-place.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[inline]
    pub fn add_inplace(&mut self, other: &ArrowItem) {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        trace!("Adding vectors in-place");
        self.item.iter_mut().zip(other.item.iter()).for_each(|(a, b)| *a += *b);
    }

    /// Multiplies element-wise in-place by another row.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[inline]
    pub fn mul_inplace(&mut self, other: &ArrowItem) {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        trace!("Multiplying vectors element-wise in-place");
        self.item.iter_mut().zip(other.item.iter()).for_each(|(a, b)| *a *= *b);
    }

    /// Scales all elements by a scalar in place.
    #[inline]
    pub fn scale(&mut self, scalar: f64) {
        trace!("Scaling vector by {:.6}", scalar);
        self.item.iter_mut().for_each(|x| *x *= scalar);
    }

    /// Immutable iterator over elements.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.item.iter()
    }

    /// Mutable iterator over elements.
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.item.iter_mut()
    }
}

/// A dense, row-major matrix of f64 with per-row spectral scores (`lambda`).
///
/// ArrowSpace stores all data in a flattened row-major `Vec<f64>` and maintains
/// a parallel `lambdas` array. It exposes allocation-free row views and
/// search-oriented operations that recompute spectral scores on mutation.
///
/// # Construction
///
/// - `from_rows` builds from a `Vec<Vec<f64>>`, validating consistent width.
///
///
/// # Panics
///
/// - Constructors panic if row lengths are inconsistent or lambda length mismatches.
/// - Indexing methods panic on out-of-bound indices.
///
/// # Performance
///
#[derive(Clone, Debug)]
pub struct ArrowSpace {
    pub nfeatures: usize,
    pub nitems: usize,
    pub data: DenseMatrix<f64>,    // NxF raw data
    pub signals: DenseMatrix<f64>, // FxF laplacian
    pub lambdas: Vec<f64>, // N lambdas (every lambda is a lambda for an item-row)
    pub taumode: Option<TauMode>, // tau_mode as in select_tau_mode
}

pub const TAUDEFAULT: Option<TauMode> = Some(TauMode::Median);

impl Default for ArrowSpace {
    fn default() -> Self {
        debug!("Creating default ArrowSpace");
        Self {
            nfeatures: 0,
            nitems: 0,
            data: DenseMatrix::new(0, 0, Vec::new(), true).unwrap(),
            signals: DenseMatrix::new(0, 0, Vec::new(), true).unwrap(),
            lambdas: Vec::new(),
            // enable synthetic λ with Median τ by default
            taumode: TAUDEFAULT,
        }
    }
}

impl ArrowSpace {
    /// Builds from a vector of equally-sized rows and per-row lambdas.
    #[inline]
    pub fn from_items(items: Vec<Vec<f64>>, taumode: TauMode) -> Self {
        info!(
            "Creating ArrowSpace from {} items with custom tau mode: {:?}",
            items.len(),
            taumode
        );
        let mut aspace = Self::from_items_default(items);
        aspace.taumode = Some(taumode);
        aspace
    }

    /// Builds from a vector of equally-sized rows and per-row lambdas.
    #[inline]
    pub fn from_items_default(items: Vec<Vec<f64>>) -> Self {
        assert!(!items.is_empty(), "items cannot be empty");
        assert!(items.len() > 1, "cannot create a arrowspace of one arrow only");
        let n_items = items.len(); // Number of items (columns in final layout)
        let n_features = items[0].len(); // Number of features (rows in final layout)

        info!(
            "Creating ArrowSpace from {} items with {} features",
            n_items, n_features
        );
        debug!("Using default tau mode: {:?}", TAUDEFAULT);

        assert!(
            items.iter().all(|item| item.len() == n_features),
            "All items must have identical number of features"
        );

        trace!("Constructing DenseMatrix from 2D vector");
        let data_matrix = DenseMatrix::from_2d_vec(&items).unwrap();
        debug!("ArrowSpace data matrix shape: {:?}", data_matrix.shape());

        Self {
            nfeatures: n_features,
            nitems: n_items,
            data: data_matrix,
            signals: DenseMatrix::new(0, 0, Vec::new(), true).unwrap(), // will be computed later
            lambdas: vec![0.0; n_items], // will be computed later
            taumode: TAUDEFAULT,
        }
    }

    /// Returns a shared reference to all lambdas.
    #[inline]
    pub fn lambdas(&self) -> &[f64] {
        self.lambdas.as_ref()
    }

    /// Returns an owned ArrowFeature copy of the requested column.
    #[inline]
    pub fn get_feature(&self, i: usize) -> ArrowFeature {
        assert!(i < self.nfeatures, "feature index out of bounds");
        trace!("Extracting feature {} from ArrowSpace", i);
        ArrowFeature { feature: self.data.get_col(i).iterator(0).copied().collect() }
    }

    /// Modify feature column in-place
    #[allow(dead_code)]
    fn set_feature(&mut self, f: usize, values: ArrowFeature) {
        assert!(f < self.nfeatures, "feature index out of bounds");
        debug!("Setting feature {} in-place", f);
        // Modify each element in the column
        for i in 0..self.nitems {
            self.data.set((i, f), values.feature[i]);
        }
    }

    /// Returns an owned ArrowItem for the requested item (row).
    #[inline]
    pub fn get_item(&self, i: usize) -> ArrowItem {
        assert!(i < self.nitems, "item index out of bounds");
        trace!("Extracting item {} with lambda {:.6}", i, self.lambdas[i]);

        ArrowItem::new(
            self.data.get_row(i).iterator(0).copied().collect(),
            self.lambdas[i],
        )
    }

    /// Modify item row in-place
    fn set_item(&mut self, i: usize, values: ArrowItem) {
        assert!(i < self.nitems, "item index out of bounds");
        debug!("Setting item {} in-place", i);
        // Modify each element in the column
        for f in 0..self.nfeatures {
            self.data.set((i, f), values.item[f]);
        }
    }

    /// Adds item `b` into item `a` in-place and recomputes feature lambdas.
    ///
    /// This method:
    /// 1. Extracts item `a` and item `b` as complete ArrowItem vectors
    /// 2. Performs element-wise addition: item_a += item_b  
    /// 3. Writes the result back
    /// 4. Recomputes feature lambdas
    #[inline]
    pub fn add_items(&mut self, a: usize, b: usize, gl: &GraphLaplacian) {
        assert!(
            a < self.nitems && b < self.nitems,
            "Item indices out of bounds: a={}, b={}, ncols={}",
            a,
            b,
            self.nitems
        );
        assert_eq!(
            gl.nnodes, self.nitems,
            "Laplacian nodes must match number of items"
        );

        info!("Adding item {} into item {}", b, a);
        debug!(
            "Graph Laplacian has {} nodes, ArrowSpace has {} items",
            gl.nnodes, self.nitems
        );

        // Extract both items as complete ArrowItem vectors
        let mut item_a = self.get_item(a);
        let item_b = self.get_item(b);

        // Perform the addition: item_a += item_b
        item_a.add_inplace(&item_b);

        self.set_item(a, item_a);

        // Recompute lambdas for all features since item values changed
        debug!("Recomputing lambdas after item addition");
        self.recompute_lambdas();
    }

    /// Multiplies item `a` element-wise by item `b` and recomputes feature lambdas.
    #[inline]
    pub fn mul_items(&mut self, a: usize, b: usize, gl: &GraphLaplacian) {
        assert!(
            a < self.nitems && b < self.nitems,
            "Item indices out of bounds: a={}, b={}, ncols={}",
            a,
            b,
            self.nitems
        );
        assert_eq!(
            gl.nnodes, self.nitems,
            "Laplacian nodes must match number of items"
        );

        info!("Multiplying item {} by item {}", a, b);

        // Extract both items as complete ArrowItem vectors
        let mut item_a = self.get_item(a);
        let item_b = self.get_item(b);

        // Perform the multiplication: item_a *= item_b
        item_a.mul_inplace(&item_b);

        self.set_item(a, item_a);

        // Recompute lambdas for all features since item values changed
        debug!("Recomputing lambdas after item multiplication");
        self.recompute_lambdas();
    }

    /// Scales item `a` by a scalar value and recomputes feature lambdas.
    #[inline]
    pub fn scale_item(&mut self, a: usize, scalar: f64, gl: &GraphLaplacian) {
        assert!(
            a < self.nitems,
            "Item index out of bounds: a={}, ncols={}",
            a,
            self.nitems
        );
        assert_eq!(
            gl.nnodes, self.nitems,
            "Laplacian nodes must match number of items"
        );

        info!("Scaling item {} by factor {:.6}", a, scalar);

        // Extract item as complete ArrowItem vector
        let mut item_a = self.get_item(a);

        // Perform the scaling: item_a *= scalar
        item_a.scale(scalar);

        self.set_item(a, item_a);

        // Recompute lambdas for all features since item values changed
        debug!("Recomputing lambdas after item scaling");
        self.recompute_lambdas();
    }

    /// Recomputes all feature lambdas using the provided Graph Laplacian.
    ///
    /// The Laplacian must have nodes equal to the number of items.
    #[inline]
    pub fn recompute_lambdas(&mut self) {
        debug!("Recomputing lambdas with tau mode: {:?}", self.taumode);
        // Use the existing synthetic lambda computation
        TauMode::compute_taumode_lambdas(self, self.taumode);

        let lambda_stats = {
            let min = self.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max: f64 = self.lambdas.iter().fold(0.0, |a, &b| a.max(b));
            let mean = self.lambdas.iter().sum::<f64>() / self.lambdas.len() as f64;
            (min, max, mean)
        };

        debug!(
            "Lambda recomputation completed - min: {:.6}, max: {:.6}, mean: {:.6}",
            lambda_stats.0, lambda_stats.1, lambda_stats.2
        );
    }

    /// Lambda-aware top-k search against an ArrowItem query.
    ///
    /// Returns indices and scores sorted descending by similarity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use arrowspace::core::{ArrowItem, ArrowSpace};
    ///
    /// let aspace = ArrowSpace::from_items_default(
    ///     vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0]]
    /// );
    /// let q = ArrowItem::new(vec![1.0, 0.1], 0.5);
    /// let res = aspace.search_lambda_aware(&q, 2, 0.7, 0.3);
    /// assert_eq!(res.len(), 2);
    /// assert!(res.1 >= 0.0);
    /// ```
    #[inline]
    pub fn search_lambda_aware(
        &self,
        query: &ArrowItem,
        k: usize,
        alpha: f64,
        beta: f64,
    ) -> Vec<(usize, f64)> {
        info!("Lambda-aware search: k={}, alpha={:.3}, beta={:.3}", k, alpha, beta);
        debug!("Query vector dimension: {}, lambda: {:.6}", query.len(), query.lambda);

        let mut results: Vec<_> = (0..self.nitems)
            .map(|i| {
                let item = self.get_item(i);
                let similarity = query.lambda_similarity(&item, alpha, beta);
                (i, similarity)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        debug!("Search completed, returning {} results", results.len());
        if !results.is_empty() {
            trace!("Top result: index={}, score={:.6}", results[0].0, results[0].1);
        }

        results
    }

    /// Range search by Euclidean distance within `radius`.
    ///
    /// Returns (row_index, distance) pairs for all rows whose distance is
    /// ≤ `radius` from the query.
    ///
    /// # Examples
    ///
    /// ```
    /// use arrowspace::core::{ArrowItem, ArrowSpace};
    ///
    /// let aspace = ArrowSpace::from_items_default(
    ///     vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]]
    /// );
    /// let q = ArrowItem::new(vec![0.5, 0.0], 0.0);
    /// let hits = aspace.range_search(&q, 0.6);
    /// ```
    #[inline]
    pub fn range_search(&self, query: &ArrowItem, radius: f64) -> Vec<(usize, f64)> {
        info!("Range search with radius: {:.6}", radius);
        debug!("Query vector dimension: {}", query.len());

        let results: Vec<(usize, f64)> = (0..self.nitems)
            .filter_map(|i| {
                let item = self.get_item(i);
                let distance = query.euclidean_distance(&item);
                if distance <= radius {
                    Some((i, distance))
                } else {
                    None
                }
            })
            .collect();

        debug!("Range search completed, found {} items within radius", results.len());
        results
    }

    /// Computes a pairwise cosine similarity submatrix for selected indices.
    ///
    /// This allocates an output matrix of size `indices.len() x indices.len()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use arrowspace::core::ArrowSpace;
    ///
    /// let aspace = ArrowSpace::from_items_default(
    ///     vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![1.0, 1.0, 0.0]]
    /// );
    /// let sims = aspace.pairwise_similarities(&);[1]
    /// assert_eq!(sims.len(), 3);
    /// assert_eq!(sims.len(), 3);
    /// ```
    #[inline]
    pub fn pairwise_similarities(&self, indices: &[usize]) -> Vec<Vec<f64>> {
        info!("Computing pairwise similarities for {} indices", indices.len());
        debug!("Selected indices: {:?}", indices);

        let results = indices
            .iter()
            .map(|&i| {
                let item_i = self.get_item(i);
                indices
                    .iter()
                    .map(|&j| {
                        let item_j = self.get_item(j);
                        item_i.cosine_similarity(&item_j)
                    })
                    .collect()
            })
            .collect();

        debug!("Pairwise similarity computation completed");
        results
    }

    /// Update the lambdas with new synthetic values
    pub fn update_lambdas(&mut self, new_lambdas: Vec<f64>) {
        assert_eq!(
            new_lambdas.len(),
            self.lambdas.len(),
            "New lambdas length must match existing lambdas length"
        );

        info!("Updating lambdas with {} new values", new_lambdas.len());
        let old_stats = {
            let min = self.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max: f64 = self.lambdas.iter().fold(0.0, |a, &b| a.max(b));
            (min, max)
        };

        self.lambdas = new_lambdas;

        let new_stats = {
            let min = self.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max: f64 = self.lambdas.iter().fold(0.0, |a, &b| a.max(b));
            (min, max)
        };

        debug!(
            "Lambda update: old range [{:.6}, {:.6}] -> new range [{:.6}, {:.6}]",
            old_stats.0, old_stats.1, new_stats.0, new_stats.1
        );
    }

    /// Compute synthetic lambda for an external query vector
    /// Uses the same normalization and parameters as the stored data
    pub fn compute_query_synthetic_lambda(&self, query_vector: &[f64]) -> f64 {
        debug!(
            "Computing synthetic lambda for query vector (dimension: {})",
            query_vector.len()
        );
        let result = TauMode::compute_item_vector_synthetic_lambda(
            query_vector,
            self,
            Some(TauMode::select_tau(query_vector, self.taumode.unwrap())),
        );
        debug!("Query synthetic lambda computed: {:.6}", result);
        result
    }
}

// Flattened AsRef/AsMut for ArrowSpace
impl AsRef<DenseMatrix<f64>> for ArrowSpace {
    #[inline]
    fn as_ref(&self) -> &DenseMatrix<f64> {
        &self.data
    }
}
impl AsMut<DenseMatrix<f64>> for ArrowSpace {
    #[inline]
    fn as_mut(&mut self) -> &mut DenseMatrix<f64> {
        &mut self.data
    }
}

// Iterate all elements by reference (feature-major)
impl<'a> IntoIterator for &'a ArrowItem {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.item.iter()
    }
}

// Iterate all elements mutably (feature-major)
impl<'a> IntoIterator for &'a mut ArrowItem {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.item.iter_mut()
    }
}
