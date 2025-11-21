//! ArrowSpace: enhanced with search-specific zero-copy operations.
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
//!
//! Zero-copy mutate a row using a mutable view and update its lambda from a graph:
//!
//!
//! Run documentation tests with `cargo test --doc`; Rustdoc extracts code blocks
//! and executes them as tests, ensuring examples stay correct over time.
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
//! if missing, and allows common lints to reduce boilerplate. Keep examples
//! small and focused; add hidden setup lines with `#` when needed so that examples
//! compile while showing only the essential lines to readers.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::{BinaryHeap, HashSet};
use std::fmt::Debug;

use approx::relative_eq;
use rayon::prelude::*;
use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;

use crate::builder::ConfigValue;
use crate::graph::GraphLaplacian;
use crate::reduction::ImplicitProjection;
use crate::sorted_index::SortedLambdas;
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
/// let b = vec![1.0, 0.0, 1.0];
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

    /// Lambda component similarity (spectral distance)
    #[inline]
    pub fn lambda_component_similarity(&self, other: &ArrowItem) -> f64 {
        let lambda_diff = (self.lambda - other.lambda).abs();
        1.0 - lambda_diff.min(1.0)
    }

    /// Combined lambda-aware similarity
    /// Combines semantic (cosine) similarity and lambda proximity (Rayleigh plus dispersion).
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
    /// let s = a.lambda_similarity(&b, 0.7);
    /// assert!(s <= 1.0 && s >= 0.0);
    /// ```
    #[inline]
    pub fn lambda_similarity(&self, other: &ArrowItem, alpha: f64) -> f64 {
        assert_eq!(
            self.item.len(),
            other.item.len(),
            "items should be of the same length"
        );
        let cosine_sim = self.cosine_similarity(&other.item);
        let lambda_sim = self.lambda_component_similarity(other);

        let result = alpha * cosine_sim + (1.0 - alpha) * lambda_sim;

        trace!(
            "Lambda similarity: semantic={:.6}, lambda={:.6}, combined={:.6}",
            cosine_sim, lambda_sim, result
        );

        result
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
        let result = self
            .item
            .iter()
            .zip(other.item.iter())
            .map(|(a, b)| a * b)
            .sum();
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
    /// let b = vec![0.0, 1.0];
    /// assert!((a.cosine_similarity(&b) - 0.0).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn cosine_similarity(&self, other: &[f64]) -> f64 {
        let denom = ArrowItem::norm(&self.item) * ArrowItem::norm(other);
        let result = if denom > 0.0 {
            self.dot(&ArrowItem::new(other.to_vec(), 0.0)) / denom
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

    /// Adds another row element-wise in-place.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    #[inline]
    pub fn add_inplace(&mut self, other: &ArrowItem) {
        assert_eq!(self.len(), other.len(), "Dimension mismatch");
        trace!("Adding vectors in-place");
        self.item
            .iter_mut()
            .zip(other.item.iter())
            .for_each(|(a, b)| *a += *b);
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
        self.item
            .iter_mut()
            .zip(other.item.iter())
            .for_each(|(a, b)| *a *= *b);
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

/// Scored item for min-heap (keeps top-k by popping smallest)
#[derive(Debug, Clone, Copy)]
struct ScoredItem {
    index: usize,
    score: f64,
}

impl PartialEq for ScoredItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredItem {}

impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse for min-heap (smallest score at top)
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
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
    pub nfeatures: usize, // F: original dimensions
    pub nitems: usize,
    pub data: DenseMatrix<f64>,        // NxF raw data
    pub signals: CsMat<f64>,           // Laplacian(Transpose(FfxFn))
    pub lambdas: Vec<f64>,             // N lambdas (every lambda is a lambda for an item-row)
    pub lambdas_sorted: SortedLambdas, // sorted by lambda ascending
    pub taumode: TauMode,              // tau_mode as in select_tau_mode

    // lambdas normalisation
    min_lambdas: f64,
    max_lambdas: f64,
    pub(crate) range_lambdas: f64,

    pub n_clusters: usize,
    /// Cluster assignment per original row (N entries, each in 0..X or None for outliers)
    pub cluster_assignments: Vec<Option<usize>>,
    /// Cluster sizes (X entries)
    pub cluster_sizes: Vec<usize>,
    /// Squared distance threshold used during clustering
    pub cluster_radius: f64,

    // Projection data: dims reduction data (needed to prepare the query vector)
    pub projection_matrix: Option<ImplicitProjection>, // F × r (if projection was used)
    pub reduced_dim: Option<usize>, // r (reduced dimension, None if no projection)
    pub extra_reduced_dim: bool,    // optional extra dimensionality reduction for energymaps

    // energymaps specific
    pub centroid_map: Option<Vec<usize>>, // Maps item_idx -> centroid_idx
    pub sub_centroids: Option<DenseMatrix<f64>>,
    pub subcentroid_lambdas: Option<Vec<f64>>,

    /// Pre-computed L2 norms for tie-breaking (energy mode)
    ///
    /// Computed during build to accelerate cosine similarity in search.
    /// Only used when items have identical lambdas (same subcentroid).
    pub item_norms: Option<Vec<f64>>,
}

pub const TAUDEFAULT: TauMode = TauMode::Median;

impl Default for ArrowSpace {
    fn default() -> Self {
        debug!("Creating default ArrowSpace");
        Self {
            nfeatures: 0,
            nitems: 0,
            data: DenseMatrix::new(0, 0, Vec::new(), true).unwrap(),
            signals: sprs::CsMat::zero((0, 0)),
            lambdas: Vec::new(),
            lambdas_sorted: SortedLambdas::new(),
            // lambdas normalisation
            min_lambdas: f64::NAN,
            max_lambdas: f64::NAN,
            range_lambdas: f64::NAN,
            // enable synthetic λ with Median τ by default
            taumode: TAUDEFAULT,
            // Clustering defaults
            n_clusters: 0,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius: 0.0,
            // projection
            projection_matrix: None,
            reduced_dim: None,
            extra_reduced_dim: false,
            // energymaps
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        }
    }
}

impl ArrowSpace {
    /// Returns an empty space from the initial data
    pub(crate) fn new(items: Vec<Vec<f64>>, taumode: TauMode) -> Self {
        assert!(!items.is_empty(), "items cannot be empty");
        assert!(
            items.len() > 1,
            "cannot create a arrowspace of one arrow only"
        );
        let n_items = items.len(); // Number of items (columns in final layout)
        let n_features = items[0].len(); // Number of features (rows in final layout)
        Self {
            nfeatures: n_features,
            nitems: n_items,
            data: DenseMatrix::from_2d_vec(&items).unwrap(),
            signals: sprs::CsMat::zero((0, 0)), // will be computed later
            lambdas: vec![0.0; n_items],        // will be computed later
            lambdas_sorted: SortedLambdas::new(),
            // lambdas normalisation
            min_lambdas: f64::NAN,
            max_lambdas: f64::NAN,
            range_lambdas: f64::NAN,
            taumode,
            // Clustering defaults
            n_clusters: 0,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius: 0.0,
            // projection
            projection_matrix: None,
            reduced_dim: None,
            extra_reduced_dim: false,
            // energymaps
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        }
    }

    /// Convenience method to generate a temporary `ArrowSpace` to reproject vectors
    pub fn empty_with_projection(
        proj_data: HashMap<String, ConfigValue>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        debug!(
            "ArrowSpace::empty_with_projection called with nrows={}, ncols={}",
            nrows, ncols
        );

        let extra_reduced = proj_data["extra_reduced_dim"].as_bool().unwrap();
        debug!("extra_reduced_dim from proj_data: {}", extra_reduced);
        assert!(
            extra_reduced == false,
            "Reconstructing with extra dim reduction is not implemented yet"
        );

        let has_projection = proj_data["pj_mtx_original_dim"].as_usize().is_some();
        debug!("projection present in proj_data: {}", has_projection);

        let mut aspace = Self::default();
        aspace.nitems = nrows;
        aspace.nfeatures = ncols;

        if has_projection {
            let original_dim = proj_data["pj_mtx_original_dim"]
                .as_usize()
                .expect("pj_mtx_original_dim must be usize when projection is present");
            let reduced_dim = proj_data["pj_mtx_reduced_dim"]
                .as_usize()
                .expect("pj_mtx_reduced_dim must be usize when projection is present");
            let seed = proj_data["pj_mtx_seed"]
                .as_u64()
                .expect("pj_mtx_seed must be u64 when projection is present");

            info!(
                "Reconstructing ImplicitProjection: original_dim={}, reduced_dim={}, seed={}",
                original_dim, reduced_dim, seed
            );

            aspace.projection_matrix = Some(ImplicitProjection {
                original_dim,
                reduced_dim,
                seed,
            });
            aspace.reduced_dim = Some(reduced_dim);
            aspace.extra_reduced_dim = extra_reduced;
        } else {
            warn!(
                "empty_with_projection called without projection metadata; \
                returning ArrowSpace without projection_matrix"
            );
        }

        debug!(
            "ArrowSpace::empty_with_projection created ArrowSpace \
            with nitems={}, nfeatures={}, reduced_dim={:?}",
            aspace.nitems, aspace.nfeatures, aspace.reduced_dim
        );

        aspace
    }

    /// Recreates an ArrowSpace from a aspace configuration HashMap.
    ///
    /// This method reconstructs a workable ArrowSpace with all properties set
    /// from the builder configuration, but with an empty data matrix.
    ///
    /// Returns:
    /// A fully configured ArrowSpace with empty data
    pub fn from_config(config: HashMap<String, ConfigValue>) -> Self {
        let nitems = config
            .get("nitems")
            .and_then(|v| v.as_usize())
            .expect("from_config: missing nitems");
        let nfeatures = config
            .get("nfeatures")
            .and_then(|v| v.as_usize())
            .expect("from_config: missing nfeatures");

        debug!(
            "ArrowSpace::from_config called (nitems={}, nfeatures={})",
            nitems, nfeatures
        );

        // --- Projection matrix ---
        let projection_matrix = if let (
            Some(ConfigValue::OptionUsize(Some(original_dim))),
            Some(ConfigValue::OptionUsize(Some(reduced_dim))),
            Some(ConfigValue::OptionU64(Some(seed))),
        ) = (
            config.get("pj_mtx_original_dim"),
            config.get("pj_mtx_reduced_dim"),
            config.get("pj_mtx_seed"),
        ) {
            info!(
                "ArrowSpace::from_config: projection matrix used: original_dim={}, reduced_dim={}",
                original_dim, reduced_dim
            );
            Some(ImplicitProjection {
                original_dim: *original_dim,
                reduced_dim: *reduced_dim,
                seed: *seed,
            })
        } else {
            debug!("ArrowSpace::from_config: projection matrix not used");
            None
        };
        let reduced_dim = match config.get("pj_mtx_reduced_dim") {
            Some(ConfigValue::OptionUsize(Some(d))) => Some(*d),
            _ => None,
        };
        let extra_reduced_dim = config
            .get("extra_reduced_dim")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // --- Tau mode (synthesis) ---
        let taumode = config
            .get("taumode")
            .and_then(|v| match v {
                ConfigValue::TauMode(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // --- Clustering ---
        let n_clusters = config
            .get("n_clusters")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let cluster_radius = config
            .get("cluster_radius")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        info!(
            "ArrowSpace::from_config: n_clusters={}, cluster_radius={}",
            n_clusters, cluster_radius
        );

        // --- Empty data and auxiliary fields ---
        let data = DenseMatrix::new(0, 0, Vec::new(), true).unwrap();
        let signals = sprs::CsMat::zero((0, 0));
        let lambdas = vec![0.0; nitems];
        let lambdas_sorted = SortedLambdas::new();

        let aspace = ArrowSpace {
            nfeatures,
            nitems,
            data,
            signals,
            lambdas,
            lambdas_sorted,
            // Normalization fields
            min_lambdas: f64::NAN,
            max_lambdas: f64::NAN,
            range_lambdas: f64::NAN,
            taumode,
            n_clusters,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius,
            // Projection
            projection_matrix,
            reduced_dim,
            extra_reduced_dim,
            // Energy-maps related fields
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        };

        debug!(
            "ArrowSpace::from_config created ArrowSpace: nitems={}, nfeatures={}, n_clusters={}, reduced_dim={:?}, extra_reduced_dim={}",
            aspace.nitems,
            aspace.nfeatures,
            aspace.n_clusters,
            aspace.reduced_dim,
            aspace.extra_reduced_dim
        );
        aspace
    }

    // drop stored in-memory data
    // to be used after data has been persisted to file
    pub fn drop_data(&mut self) {
        info!("Freeing raw input memory, should have been persisted to file");
        self.data = DenseMatrix::new(0, 0, vec![], true).unwrap();
    }

    /// Returns an empty space from the initial data
    pub(crate) fn new_from_dense(items: DenseMatrix<f64>, taumode: TauMode) -> Self {
        let n_items = items.shape().0; // Number of items (columns in final layout)
        let n_features = items.shape().1; // Number of features (rows in final layout)
        let empty = items.is_empty();
        info!(r#"new_from_dense: {n_items}x{n_features} -> empty: {empty}"#);
        assert!(
            items.shape().0 > 1,
            "cannot create a arrowspace of one arrow only"
        );
        Self {
            nfeatures: n_features,
            nitems: n_items,
            data: items,
            signals: sprs::CsMat::zero((0, 0)), // will be computed later
            lambdas: vec![0.0; n_items],        // will be computed later
            lambdas_sorted: SortedLambdas::new(),
            // lambdas normalisation
            min_lambdas: f64::NAN,
            max_lambdas: f64::NAN,
            range_lambdas: f64::NAN,
            taumode,
            // Clustering defaults
            n_clusters: 0,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius: 0.0,
            // projection
            projection_matrix: None,
            reduced_dim: None,
            extra_reduced_dim: false,
            // energymaps
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        }
    }

    /// Builds from a vector of equally-sized rows and per-row lambdas.
    /// Only to be used in tests. Use `ArrowSpaceBuilder`
    #[inline]
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn from_items(items: Vec<Vec<f64>>, taumode: TauMode) -> Self {
        warn!(
            "This is just a test method. Use ArrowSpaceBuilder. Creating ArrowSpace from {} items with custom tau mode: {:?}",
            items.len(),
            taumode
        );
        let mut aspace = Self::from_items_default(items);
        aspace.taumode = taumode;
        aspace
    }

    /// Builds from a vector of equally-sized rows and per-row lambdas.
    /// Only to be used in tests. `ArrowSpaceBuilder`
    #[inline]
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn from_items_default(items: Vec<Vec<f64>>) -> Self {
        assert!(!items.is_empty(), "items cannot be empty");
        assert!(
            items.len() > 1,
            "cannot create a arrowspace of one arrow only"
        );
        let n_items = items.len(); // Number of items (columns in final layout)
        let n_features = items[0].len(); // Number of features (rows in final layout)

        warn!(
            "This is a test method, use ArrowSpaceBuilder. Creating ArrowSpace from {} items with {} features",
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
            signals: sprs::CsMat::zero((0, 0)), // will be computed later
            lambdas: vec![0.0; n_items],        // will be computed later
            lambdas_sorted: SortedLambdas::new(),
            // lambdas normalisation
            min_lambdas: f64::NAN,
            max_lambdas: f64::NAN,
            range_lambdas: f64::NAN,
            taumode: TAUDEFAULT,
            // Clustering defaults
            n_clusters: 0,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius: 0.0,
            // projection
            projection_matrix: None,
            reduced_dim: None,
            extra_reduced_dim: false,
            // energymaps
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        }
    }

    // core.rs
    pub(crate) fn subcentroids_from_dense_matrix(matrix: DenseMatrix<f64>) -> Self {
        let (n_rows, n_cols) = matrix.shape();
        // For subcentroids, rows = C (items), cols = F (features)
        let nitems = n_rows;
        let nfeatures = n_cols;

        info!(
            "Creating subcentroid ArrowSpace from DenseMatrix({}, {})",
            n_rows, n_cols
        );
        info!(
            "→ Interpreted as: {} subcentroids × {} features (row-major)",
            nitems, nfeatures
        );

        Self {
            data: matrix,
            nitems,
            nfeatures,
            signals: sprs::CsMat::zero((0, 0)),
            lambdas: vec![0.0; nitems],
            lambdas_sorted: SortedLambdas::new(),
            min_lambdas: f64::NAN,
            max_lambdas: f64::NAN,
            range_lambdas: f64::NAN,
            taumode: TAUDEFAULT,
            n_clusters: 0,
            cluster_assignments: Vec::new(),
            cluster_sizes: Vec::new(),
            cluster_radius: 0.0,
            projection_matrix: None,
            reduced_dim: None,
            extra_reduced_dim: false,
            centroid_map: None,
            sub_centroids: None,
            subcentroid_lambdas: None,
            item_norms: None,
        }
    }

    /// Project query vector to reduced space if projection was used during indexing
    ///
    /// # Arguments
    /// * `query` - Original F-dimensional query vector
    ///
    /// # Returns
    /// * If projection was used: r-dimensional projected query
    /// * If no projection: original query unchanged
    pub fn project_query(&self, query: &[f64]) -> Vec<f64> {
        assert_eq!(
            query.len(),
            self.nfeatures,
            "Query dimension {} doesn't match index original dimension {}",
            query.len(),
            self.nfeatures
        );

        if let Some(ref proj) = self.projection_matrix {
            debug!(
                "Projecting query: {} → {} dimensions using seed-based projection",
                self.nfeatures,
                self.reduced_dim.unwrap()
            );
            proj.project(query)
        } else {
            debug!("No projection applied, returning original query");
            query.to_vec()
        }
    }

    /// Compute query lambda for mode
    ///
    /// Maps query to nearest subcentroid and returns its lambda.
    /// Pre-computed subcentroids and lambdas are already stored in ArrowSpace.
    pub fn prepare_query_item(&self, query: &[f64], gl: &GraphLaplacian) -> f64 {
        assert!(
            query.iter().all(|x| x.is_finite()),
            "query item has non-finite values"
        );

        // Energy mode: subcentroid mapping (fast)
        if let (Some(subcentroids), Some(sc_lambdas)) =
            (&self.sub_centroids, &self.subcentroid_lambdas)
        {
            let mut best_idx = 0;
            let mut best_dist = f64::INFINITY;

            // Find nearest subcentroid
            for sc_idx in 0..subcentroids.shape().0 {
                let query = if self.extra_reduced_dim {
                    &self.project_query(query)
                } else {
                    query
                };
                let dist: f64 = query
                    .iter()
                    .zip(subcentroids.get_row(sc_idx).iterator(0))
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = sc_idx;
                }
            }

            let lambda = sc_lambdas[best_idx];

            debug!(
                "Query mapped to subcentroid {}/{} with λ={:.6} (dist={:.4})",
                best_idx,
                subcentroids.shape().0,
                lambda,
                best_dist
            );

            return lambda;
        }

        // Eigen mode
        let tau = TauMode::select_tau(&query, self.taumode);
        let raw_lambda = TauMode::compute_synthetic_lambda(&query, &self, &gl.matrix, tau);

        // Normalize if stats are available
        let msg = "Check your eps parameter for the builder, every dataset has an optimal eps. \n \
            Also, the query item may be out of context for the dataset (undecidable), \
            despite all safeguards its lambda is 0.0";
        if self.range_lambdas.is_finite() {
            if relative_eq!(raw_lambda, 0.0, epsilon = 1e-12) {
                panic!("{}", msg)
            }
            return self.normalise_query_lambda(raw_lambda);
        } else {
            if relative_eq!(raw_lambda, 0.0, epsilon = 1e-12) {
                panic!("{}", msg)
            }
            return raw_lambda;
        }
    }

    /// Build the sorted index
    pub fn build_lambdas_sorted(&mut self) {
        self.lambdas_sorted.build_from(&self.lambdas);
    }

    /// Returns a shared reference to all lambdas.
    #[inline]
    pub fn lambdas(&self) -> &[f64] {
        self.lambdas.as_ref()
    }

    /// Returns cluster assignment for row i (None if outlier or not clustered).
    #[inline]
    pub fn cluster_of(&self, i: usize) -> Option<usize> {
        self.cluster_assignments.get(i).copied().flatten()
    }

    /// Returns an owned ArrowFeature copy of the requested column.
    #[inline]
    pub fn get_feature(&self, i: usize) -> ArrowFeature {
        assert!(i < self.nfeatures, "feature index out of bounds");
        trace!("Extracting feature {} from ArrowSpace", i);
        ArrowFeature {
            feature: self.data.get_col(i).iterator(0).copied().collect(),
        }
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

        debug!("Adding item {} into item {}", b, a);
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
        self.recompute_lambdas(gl);
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

        debug!("Multiplying item {} by item {}", a, b);

        // Extract both items as complete ArrowItem vectors
        let mut item_a = self.get_item(a);
        let item_b = self.get_item(b);

        // Perform the multiplication: item_a *= item_b
        item_a.mul_inplace(&item_b);

        self.set_item(a, item_a);

        // Recompute lambdas for all features since item values changed
        debug!("Recomputing lambdas after item multiplication");
        self.recompute_lambdas(gl);
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

        debug!("Scaling item {} by factor {:.6}", a, scalar);

        // Extract item as complete ArrowItem vector
        let mut item_a = self.get_item(a);

        // Perform the scaling: item_a *= scalar
        item_a.scale(scalar);

        self.set_item(a, item_a);

        // Recompute lambdas for all features since item values changed
        debug!("Recomputing lambdas after item scaling");
        self.recompute_lambdas(gl);
    }

    /// Recomputes all feature lambdas using the provided Graph Laplacian.
    ///
    /// The Laplacian must have nodes equal to the number of items.
    #[inline]
    pub fn recompute_lambdas(&mut self, gl: &GraphLaplacian) {
        debug!("Recomputing lambdas with tau mode: {:?}", self.taumode);
        // Use the existing synthetic lambda computation
        TauMode::compute_taumode_lambdas_parallel(self, gl, self.taumode);

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
    /// Single-pass parallel dual scoring with `ArrowScore`.
    ///
    /// Returns indices and scores sorted descending by similarity.
    /// Algorithm:
    /// 1. Single parallel scan maintains:
    ///    - Semantic top-1 (best cosine)
    ///    - High semantic matches (cosine > 0.95)
    ///    - Top-k lambda-aware candidates (min-heap)
    /// 2. Union: high semantic + top-k lambda (deduplicated)
    /// 3. Replace lowest lambda score with semantic top-1 if needed
    ///
    /// # Performance
    /// - O(N) scan with O(k log k) heap operations per thread
    /// - No full array allocation or sorting
    /// - Lock-free concurrent top-k tracking via DashMap
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
    ) -> Vec<(usize, f64)> {
        info!("Lambda-aware search: k={}", k);
        debug!(
            "Query vector dimension: {}, lambda: {:.6}",
            query.len(),
            query.lambda
        );

        assert_ne!(
            query.lambda, 0.0,
            "Lambda of the item is 0.0, prepare the item before searching"
        );

        let mut results: Vec<_> = (0..self.nitems)
            .map(|i| {
                let item = self.get_item(i);
                let similarity = query.lambda_similarity(&item, alpha);
                (i, similarity)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        debug!("Search completed, returning {} results", results.len());
        if !results.is_empty() {
            trace!(
                "Top result: index={}, score={:.6}",
                results[0].0, results[0].1
            );
        }

        results
    }

    /// A version of `search_lambda_aware` that mix-in results from pure cosine similarity
    #[inline]
    pub fn search_lambda_aware_hybrid(
        &self,
        query: &ArrowItem,
        k: usize,
        alpha: f64,
    ) -> Vec<(usize, f64)> {
        info!("Hybrid search: k={}, alpha={}", k, alpha);

        if k == 0 {
            return Vec::new();
        }

        let beta = 1.0 - alpha;
        let semantic_threshold = 0.9999;

        // Parallel fold-reduce with thread-local state
        let (lambda_heap, semantic_top, high_semantic_vec) = (0..self.nitems)
            .into_par_iter()
            .fold(
                || {
                    (
                        BinaryHeap::with_capacity(k), // Lambda top-k heap
                        (0usize, f64::NEG_INFINITY),  // Semantic best
                        Vec::new(),                   // High semantic matches
                    )
                },
                |(mut heap, mut sem_best, mut high_sem), i| {
                    let item = self.get_item(i);

                    // Single computation of both similarities
                    let cosine = query.cosine_similarity(&item.item);
                    let lambda_component = query.lambda_component_similarity(&item);
                    let lambda_score = alpha * cosine + beta * lambda_component;

                    // Track semantic best
                    if cosine > sem_best.1 {
                        sem_best = (i, cosine);
                    }

                    // Track high semantic matches
                    if cosine > semantic_threshold {
                        high_sem.push((i, cosine));
                    }

                    // Maintain lambda top-k heap
                    if heap.len() < k {
                        heap.push(ScoredItem {
                            index: i,
                            score: lambda_score,
                        });
                    } else if let Some(&min) = heap.peek() {
                        if lambda_score > min.score {
                            heap.pop();
                            heap.push(ScoredItem {
                                index: i,
                                score: lambda_score,
                            });
                        }
                    }

                    (heap, sem_best, high_sem)
                },
            )
            .reduce(
                || (BinaryHeap::new(), (0, f64::NEG_INFINITY), Vec::new()),
                |(mut h1, s1, mut hs1), (h2, s2, hs2)| {
                    // Merge semantic best
                    let sem = if s1.1 > s2.1 { s1 } else { s2 };

                    // Merge high semantic
                    hs1.extend(hs2);

                    // Merge heaps
                    for item in h2 {
                        if h1.len() < k {
                            h1.push(item);
                        } else if let Some(&min) = h1.peek() {
                            if item.score > min.score {
                                h1.pop();
                                h1.push(item);
                            }
                        }
                    }

                    (h1, sem, hs1)
                },
            );

        debug!(
            "Semantic top: idx={}, score={:.6}, high_semantic_count={}",
            semantic_top.0,
            semantic_top.1,
            high_semantic_vec.len()
        );

        // Union: high semantic + lambda top-k
        let mut result_set = HashSet::with_capacity(k + high_semantic_vec.len());
        let mut score_map = std::collections::HashMap::with_capacity(result_set.capacity());

        // Add high semantic matches
        for (idx, score) in high_semantic_vec {
            result_set.insert(idx);
            score_map.insert(idx, score);
        }

        // Add lambda top-k
        for item in lambda_heap.into_sorted_vec().into_iter().rev() {
            result_set.insert(item.index);
            score_map.entry(item.index).or_insert(item.score);
        }

        // Ensure semantic top-1 is included
        result_set.insert(semantic_top.0);
        score_map.entry(semantic_top.0).or_insert(semantic_top.1);

        // Convert to vector and sort
        let mut final_results: Vec<(usize, f64)> = result_set
            .into_iter()
            .map(|idx| (idx, *score_map.get(&idx).unwrap()))
            .collect();

        final_results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        final_results.truncate(k);

        debug!("Final: {} results", final_results.len());
        final_results
    }

    /// Search using only taumode lambdas sorted index
    pub fn search_linear_sorted(
        &self,
        query: &[f64],
        gl: &GraphLaplacian,
        k: usize,
    ) -> Vec<(usize, f64)> {
        let q_lambda = self.prepare_query_item(query, gl); // f64
        self.lambdas_sorted
            .range_bylambda(q_lambda, k, gl.graph_params.p)
    }

    /// Normalise lambdas to [0, 1] range for consistent search behavior
    /// This is called every time taumode is recomputed
    /// Lambdas stats are used by `prepare_query` to normalise the query item
    #[inline]
    pub fn normalise_lambdas(&mut self) {
        self.min_lambdas = self.lambdas.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        self.max_lambdas = self.lambdas.iter().fold(0.0_f64, |a, &b| a.max(b));
        self.range_lambdas = (self.max_lambdas - self.min_lambdas).max(1e-9);

        for lambda in &mut self.lambdas {
            *lambda = (*lambda - self.min_lambdas) / self.range_lambdas;
        }

        info!(
            "Normalized lambdas to [0, 1] range (original spread: {:.6})",
            self.range_lambdas
        );
    }

    /// Normalise a single query lambda to [0, 1] range using stored statistics
    ///
    /// This method applies the same normalization transform used for indexed lambdas
    /// to a raw query lambda value, ensuring consistent distance calculations.
    #[inline]
    pub fn normalise_query_lambda(&self, raw_lambda: f64) -> f64 {
        debug_assert!(
            self.range_lambdas > 0.0,
            "Call normalise_lambdas() before normalising query lambdas"
        );

        // Apply same transform as batch normalization
        let normalized = (raw_lambda - self.min_lambdas) / self.range_lambdas;

        // Clamp to [0, 1] to handle edge cases where query lambda is outside
        // the training range (e.g., out-of-distribution queries)
        normalized.clamp(0.0, 1.0)
    }

    /// Range search by Euclidean distance within `radius`.
    ///
    /// Returns (row_index, distance) pairs for all rows whose distance is
    /// ≤ `radius` from the query.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use arrowspace::core::{ArrowItem, ArrowSpace};
    ///
    /// let q = ArrowItem::new(vec![0.5, 0.0], 0.0);
    /// let hits = aspace.range_search(&q, 0.6);
    /// ```
    #[inline]
    pub fn range_search(
        &self,
        query: &ArrowItem,
        gl: &GraphLaplacian,
        eps: f64,
    ) -> Vec<(usize, f64)> {
        info!("Range search with radius: {:.6}", eps);
        debug!("Query vector dimension: {}", query.len());

        let query: ArrowItem = if relative_eq!(query.lambda, 0.0, epsilon = 1e-9) {
            ArrowItem::new(query.item.clone(), self.prepare_query_item(&query.item, gl))
        } else {
            query.clone()
        };

        let results: Vec<(usize, f64)> = (0..self.nitems)
            .filter_map(|i| {
                let item = self.get_item(i);
                let distance = query.lambda - item.lambda;
                if distance <= eps {
                    Some((i, distance))
                } else {
                    None
                }
            })
            .collect();

        debug!(
            "Range search completed, found {} items within radius",
            results.len()
        );
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
        self.normalise_lambdas();

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

    /// Convert all ArrowSpace configuration into a typed hashmap.
    ///
    /// This extracts all parameters needed to rebuild an ArrowSpace with identical
    /// behavior: tau mode, projection settings, and dimension metadata.
    pub fn arrowspace_config_typed(&self) -> HashMap<String, ConfigValue> {
        let mut config = HashMap::new();

        config.insert("nitems".to_string(), ConfigValue::Usize(self.nitems));
        config.insert("nfeatures".to_string(), ConfigValue::Usize(self.nfeatures));

        // projection matrix
        if self.projection_matrix.is_some() {
            config.insert(
                "pj_mtx_original_dim".to_string(),
                ConfigValue::OptionUsize(Some(
                    self.projection_matrix.as_ref().unwrap().original_dim,
                )),
            );
            config.insert(
                "pj_mtx_reduced_dim".to_string(),
                ConfigValue::OptionUsize(Some(
                    self.projection_matrix.as_ref().unwrap().reduced_dim,
                )),
            );
            config.insert(
                "pj_mtx_seed".to_string(),
                ConfigValue::OptionU64(Some(self.projection_matrix.as_ref().unwrap().seed)),
            );

            config.insert(
                "extra_reduced_dim".to_string(),
                ConfigValue::Bool(self.extra_reduced_dim),
            );
        } else {
            config.insert(
                "pj_mtx_original_dim".to_string(),
                ConfigValue::OptionUsize(None),
            );
            config.insert(
                "pj_mtx_reduced_dim".to_string(),
                ConfigValue::OptionUsize(None),
            );
            config.insert("pj_mtx_seed".to_string(), ConfigValue::OptionU64(None));
            config.insert("extra_reduced_dim".to_string(), ConfigValue::Bool(false));
        }

        config.insert(
            "taumode".to_string(),
            ConfigValue::TauMode(self.taumode.clone()),
        );

        config.insert(
            "n_clusters".to_string(),
            ConfigValue::Usize(self.n_clusters),
        );
        config.insert(
            "cluster_radius".to_string(),
            ConfigValue::F64(self.cluster_radius),
        );

        config
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

pub fn densematrix_to_vecvec(matrix: &DenseMatrix<f64>) -> Vec<Vec<f64>> {
    let (rows, cols) = matrix.shape();
    (0..rows)
        .map(|r| (0..cols).map(|c| matrix.get((r, c)).clone()).collect())
        .collect()
}
