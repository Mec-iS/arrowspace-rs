use crate::builder::{ArrowSpaceBuilder, ConfigValue};
use crate::sampling::SamplerType;
use crate::taumode::TauMode;
use approx::assert_relative_eq;
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::{CsMat, TriMat};
use tempfile::TempDir;

use crate::storage::parquet::*;

// ========================================================================
// Helper Functions
// ========================================================================

pub(crate) fn create_test_dense_matrix() -> DenseMatrix<f64> {
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![1e-3, 1e-5, 0.342352362362323],
    ];
    DenseMatrix::from_2d_vec(&data).unwrap()
}

pub(crate) fn create_test_dense_matrix_with_size(rows: usize, cols: usize) -> DenseMatrix<f64> {
    let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();
    DenseMatrix::from_iterator(data.into_iter(), rows, cols, 1)
}

pub(crate) fn create_test_sparse_matrix() -> CsMat<f64> {
    let mut trimat = TriMat::new((4, 4));
    trimat.add_triplet(0, 0, 2.0);
    trimat.add_triplet(0, 1, -1.0);
    trimat.add_triplet(1, 1, 3.0);
    trimat.add_triplet(1, 2, -1.5);
    trimat.add_triplet(2, 2, 1.5);
    trimat.add_triplet(3, 3, 4.0);
    trimat.to_csr()
}

pub(crate) fn create_test_sparse_matrix_with_size(rows: usize, cols: usize) -> CsMat<f64> {
    let mut trimat = TriMat::new((rows, cols));
    for i in 0..rows {
        trimat.add_triplet(i, i % cols, 1.0);
    }
    trimat.to_csr::<usize>()
}

pub(crate) fn create_test_builder() -> ArrowSpaceBuilder {
    ArrowSpaceBuilder::new()
        .with_lambda_graph(0.5, 10, 4, 2.0, Some(0.6))
        .with_synthesis(TauMode::Median)
        .with_inline_sampling(Some(SamplerType::Simple(0.6)))
        .with_normalisation(true)
        .with_seed(42)
}

fn assert_matrices_equal(m1: &DenseMatrix<f64>, m2: &DenseMatrix<f64>) {
    assert_eq!(m1.shape(), m2.shape());
    let (rows, cols) = m1.shape();
    for i in 0..rows {
        for j in 0..cols {
            assert_relative_eq!(*m1.get((i, j)), *m2.get((i, j)), epsilon = 1e-10);
        }
    }
}

#[allow(dead_code)]
fn assert_sparse_matrices_equal(m1: &CsMat<f64>, m2: &CsMat<f64>) {
    assert_eq!(m1.shape(), m2.shape());
    assert_eq!(m1.nnz(), m2.nnz());

    let (rows, cols) = m1.shape();
    for i in 0..rows {
        for j in 0..cols {
            let v1 = m1.get(i, j).copied().unwrap_or(0.0);
            let v2 = m2.get(i, j).copied().unwrap_or(0.0);
            assert_relative_eq!(v1, v2, epsilon = 1e-10);
        }
    }
}

// ========================================================================
// Metadata Tests
// ========================================================================

#[test]
fn test_metadata_creation() {
    let metadata = ArrowSpaceMetadata::new("test_matrix");

    assert_eq!(metadata.name_id, "test_matrix");
    assert_eq!(metadata.n_rows, 0);
    assert_eq!(metadata.n_cols, 0);
    assert!(metadata.builder_config.is_empty());
    assert!(metadata.files.is_empty());
    assert!(!metadata.timestamp.is_empty());
}

#[test]
fn test_metadata_from_builder() {
    let builder = create_test_builder();
    let metadata = ArrowSpaceMetadata::from_builder("test", &builder);

    assert_eq!(metadata.name_id, "test");
    assert!(!metadata.builder_config.is_empty());

    // Verify config values
    assert_eq!(metadata.lambda_eps(), Some(0.5));
    assert_eq!(metadata.lambda_k(), Some(10));

    // Verify synthesis mode
    if let Some(tau) = metadata.synthesis() {
        assert_eq!(tau, TauMode::Median);
    } else {
        panic!("Expected TauMode::Median");
    }
}

#[test]
fn test_metadata_builder_pattern() {
    let metadata = ArrowSpaceMetadata::new("test")
        .with_dimensions(100, 50)
        .add_file(
            "data",
            FileInfo {
                filename: "data.parquet".to_string(),
                file_type: "dense".to_string(),
                rows: 100,
                cols: 50,
                nnz: None,
                size_bytes: Some(1024),
            },
        );

    assert_eq!(metadata.n_rows, 100);
    assert_eq!(metadata.n_cols, 50);
    assert_eq!(metadata.files.len(), 1);
    assert!(metadata.files.contains_key("data"));
}

#[test]
fn test_metadata_save_load() {
    let temp_dir = TempDir::new().unwrap();
    let builder = create_test_builder();

    let original =
        ArrowSpaceMetadata::from_builder("test_metadata", &builder).with_dimensions(100, 50);

    save_metadata(&original, temp_dir.path(), "test_metadata").unwrap();

    let loaded = load_metadata(temp_dir.path(), "test_metadata").unwrap();

    assert_eq!(original.name_id, loaded.name_id);
    assert_eq!(original.n_rows, loaded.n_rows);
    assert_eq!(original.n_cols, loaded.n_cols);
    assert_eq!(loaded.lambda_eps(), Some(0.5));
    assert_eq!(loaded.lambda_k(), Some(10));
}

#[test]
fn test_metadata_config_summary() {
    let builder = create_test_builder();
    let metadata = ArrowSpaceMetadata::from_builder("test", &builder);

    let summary = metadata.config_summary();
    assert!(summary.contains("lambda_eps"));
    assert!(summary.contains("0.5"));
    assert!(summary.contains("synthesis"));
}

// ========================================================================
// Dense Matrix Storage Tests
// ========================================================================

#[test]
fn test_dense_matrix_save_load_without_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let original = create_test_dense_matrix();

    save_dense_matrix(&original, temp_dir.path(), "test_dense", None).unwrap();

    // Verify file exists
    let file_path = temp_dir.path().join("test_dense.parquet");
    assert!(file_path.exists());

    // Verify no metadata file
    let metadata_path = temp_dir.path().join("test_dense_metadata.json");
    assert!(!metadata_path.exists());
}

#[test]
fn test_dense_matrix_save_load_with_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let original = create_test_dense_matrix();
    let builder = create_test_builder();

    save_dense_matrix_with_builder(&original, temp_dir.path(), "test_dense", Some(&builder))
        .unwrap();

    // Verify both files exist
    assert!(temp_dir.path().join("test_dense.parquet").exists());
    assert!(temp_dir.path().join("test_dense_metadata.json").exists());

    // Load and verify metadata
    let metadata = load_metadata(temp_dir.path(), "test_dense").unwrap();
    assert_eq!(metadata.n_rows, 4);
    assert_eq!(metadata.n_cols, 3);
    assert_eq!(metadata.files.len(), 1);
    assert_eq!(metadata.lambda_eps(), Some(0.5));
}

#[test]
fn test_dense_matrix_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let original = create_test_dense_matrix();

    save_dense_matrix(&original, temp_dir.path(), "roundtrip", None).unwrap();

    let loaded = load_dense_matrix(temp_dir.path().join("roundtrip.parquet")).unwrap();
    assert_matrices_equal(&original, &loaded);
}

#[test]
fn test_dense_matrix_large_dimensions() {
    let temp_dir = TempDir::new().unwrap();

    // Create larger matrix
    let data: Vec<Vec<f64>> = (0..100)
        .map(|i| (0..50).map(|j| (i * 50 + j) as f64).collect())
        .collect();
    let large_matrix = DenseMatrix::from_2d_vec(&data).unwrap();

    save_dense_matrix(&large_matrix, temp_dir.path(), "large", None).unwrap();

    // Verify file was created and has reasonable size
    let file_path = temp_dir.path().join("large.parquet");
    let metadata = std::fs::metadata(&file_path).unwrap();
    assert!(metadata.len() > 1000); // Should be at least 1KB
}

// ========================================================================
// Sparse Matrix Storage Tests
// ========================================================================

#[test]
fn test_sparse_matrix_save_without_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let original = create_test_sparse_matrix();

    save_sparse_matrix(&original, temp_dir.path(), "test_sparse", None).unwrap();

    assert!(temp_dir.path().join("test_sparse.parquet").exists());
    assert!(!temp_dir.path().join("test_sparse_metadata.json").exists());
}

#[test]
fn test_sparse_matrix_save_with_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let original = create_test_sparse_matrix();
    let builder = create_test_builder();

    save_sparse_matrix_with_builder(&original, temp_dir.path(), "test_sparse", Some(&builder))
        .unwrap();

    assert!(temp_dir.path().join("test_sparse.parquet").exists());
    assert!(temp_dir.path().join("test_sparse_metadata.json").exists());

    let metadata = load_metadata(temp_dir.path(), "test_sparse").unwrap();
    assert_eq!(metadata.n_rows, 4);
    assert_eq!(metadata.n_cols, 4);

    let file_info = metadata.files.get("matrix").unwrap();
    assert_eq!(file_info.file_type, "sparse");
    assert_eq!(file_info.nnz, Some(6));
}

#[test]
fn test_sparse_matrix_empty() {
    let temp_dir = TempDir::new().unwrap();

    let empty = TriMat::new((5, 5)).to_csr();
    save_sparse_matrix(&empty, temp_dir.path(), "empty", None).unwrap();

    assert!(temp_dir.path().join("empty.parquet").exists());
}

// ========================================================================
// Checkpoint Tests
// ========================================================================

#[test]
fn test_checkpoint_save_all_artifacts() {
    let temp_dir = TempDir::new().unwrap();
    let builder = create_test_builder();

    let raw_data = create_test_dense_matrix();
    let centroids = DenseMatrix::from_2d_vec(&vec![vec![1.5, 2.5, 3.5]]).unwrap();
    let adjacency = create_test_sparse_matrix();
    let laplacian = create_test_sparse_matrix();
    let signals = create_test_sparse_matrix();

    save_arrowspace_checkpoint_with_builder(
        temp_dir.path(),
        "checkpoint_test",
        &raw_data,
        &adjacency,
        &centroids,
        &laplacian,
        &signals,
        &builder,
    )
    .unwrap();

    // Verify all files exist
    let expected_files = vec![
        "checkpoint_test_raw_data.parquet",
        "checkpoint_test_adjacency.parquet",
        "checkpoint_test_centroids.parquet",
        "checkpoint_test_laplacian.parquet",
        "checkpoint_test_signals.parquet",
        "checkpoint_test_metadata.json",
    ];

    for filename in expected_files {
        let path = temp_dir.path().join(filename);
        assert!(path.exists(), "Missing file: {}", filename);
    }
}

#[test]
fn test_checkpoint_metadata_completeness() {
    let temp_dir = TempDir::new().unwrap();
    let builder = create_test_builder();

    let raw_data = create_test_dense_matrix();
    let centroids = DenseMatrix::from_2d_vec(&vec![vec![1.5]]).unwrap();
    let adj = create_test_sparse_matrix();

    save_arrowspace_checkpoint_with_builder(
        temp_dir.path(),
        "complete",
        &raw_data,
        &adj,
        &centroids,
        &adj,
        &adj,
        &builder,
    )
    .unwrap();

    let metadata = load_metadata(temp_dir.path(), "complete").unwrap();

    // Verify all artifacts are registered
    assert_eq!(metadata.files.len(), 5);
    assert!(metadata.files.contains_key("raw_data"));
    assert!(metadata.files.contains_key("adjacency"));
    assert!(metadata.files.contains_key("centroids"));
    assert!(metadata.files.contains_key("laplacian"));
    assert!(metadata.files.contains_key("signals"));

    // Verify file info
    let raw_info = metadata.files.get("raw_data").unwrap();
    assert_eq!(raw_info.file_type, "dense");
    assert_eq!(raw_info.rows, 4);
    assert_eq!(raw_info.cols, 3);

    let adj_info = metadata.files.get("adjacency").unwrap();
    assert_eq!(adj_info.file_type, "sparse");
    assert_eq!(adj_info.nnz, Some(6));
}

#[test]
fn test_checkpoint_with_auto_graph_builder() {
    let temp_dir = TempDir::new().unwrap();

    // Use auto-graph configuration
    let builder = ArrowSpaceBuilder::new().with_synthesis(TauMode::Fixed(0.7));

    let matrix = create_test_dense_matrix();
    let sparse = create_test_sparse_matrix();

    save_arrowspace_checkpoint_with_builder(
        temp_dir.path(),
        "auto_config",
        &matrix,
        &sparse,
        &matrix,
        &sparse,
        &sparse,
        &builder,
    )
    .unwrap();

    let metadata = load_metadata(temp_dir.path(), "auto_config").unwrap();

    // Verify auto-graph parameters were saved
    assert!(metadata.lambda_eps().is_some());
    assert!(metadata.lambda_k().is_some());

    // Verify synthesis mode
    if let Some(tau) = metadata.synthesis() {
        match tau {
            TauMode::Fixed(alpha) => assert_relative_eq!(alpha, 0.7, epsilon = 1e-10),
            _ => panic!("Expected TauMode::Alpha"),
        }
    }
}

// ========================================================================
// Error Handling Tests
// ========================================================================

#[test]
fn test_load_metadata_nonexistent() {
    let temp_dir = TempDir::new().unwrap();

    let result = load_metadata(temp_dir.path(), "nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_save_to_readonly_directory() {
    // Skip on systems where we can't create readonly dirs
    #[cfg(unix)]
    {
        let temp_dir = TempDir::new().unwrap();
        let readonly_path = temp_dir.path().join("readonly");
        std::fs::create_dir(&readonly_path).unwrap();

        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&readonly_path).unwrap().permissions();
        perms.set_mode(0o444);
        std::fs::set_permissions(&readonly_path, perms).unwrap();

        let matrix = create_test_dense_matrix();
        let result = save_dense_matrix(&matrix, &readonly_path, "test", None);
        assert!(result.is_err());
    }
}

// ========================================================================
// Integration Tests
// ========================================================================

#[test]
fn test_multiple_checkpoints_same_directory() {
    let temp_dir = TempDir::new().unwrap();

    let matrix = create_test_dense_matrix();
    let sparse = create_test_sparse_matrix();

    let builder1 = ArrowSpaceBuilder::new().with_lambda_graph(0.5, 10, 4, 2.0, None);

    let builder2 = ArrowSpaceBuilder::new().with_lambda_graph(1.0, 20, 8, 2.0, None);

    // Save two different checkpoints
    save_arrowspace_checkpoint_with_builder(
        temp_dir.path(),
        "checkpoint_v1",
        &matrix,
        &sparse,
        &matrix,
        &sparse,
        &sparse,
        &builder1,
    )
    .unwrap();

    save_arrowspace_checkpoint_with_builder(
        temp_dir.path(),
        "checkpoint_v2",
        &matrix,
        &sparse,
        &matrix,
        &sparse,
        &sparse,
        &builder2,
    )
    .unwrap();

    // Load both and verify they're different
    let meta1 = load_metadata(temp_dir.path(), "checkpoint_v1").unwrap();
    let meta2 = load_metadata(temp_dir.path(), "checkpoint_v2").unwrap();

    assert_eq!(meta1.lambda_eps(), Some(0.5));
    assert_eq!(meta2.lambda_eps(), Some(1.0));
    assert_eq!(meta1.lambda_k(), Some(10));
    assert_eq!(meta2.lambda_k(), Some(20));
}

#[test]
fn test_config_value_preservation() {
    let temp_dir = TempDir::new().unwrap();

    let builder = ArrowSpaceBuilder::new()
        .with_lambda_graph(0.123, 7, 3, 2.5, Some(0.456))
        .with_synthesis(TauMode::Percentile(95.0))
        .with_inline_sampling(Some(SamplerType::DensityAdaptive(0.789)))
        .with_seed(12345);

    let matrix = create_test_dense_matrix();

    save_dense_matrix_with_builder(&matrix, temp_dir.path(), "precise", Some(&builder)).unwrap();

    let metadata = load_metadata(temp_dir.path(), "precise").unwrap();

    // Verify precise values
    assert_eq!(metadata.lambda_eps(), Some(0.123));
    assert_eq!(metadata.lambda_k(), Some(7));

    if let Some(ConfigValue::F64(radius)) = metadata.get_config("cluster_radius") {
        assert_relative_eq!(*radius, 1.0, epsilon = 1e-10);
    } else {
        panic!("Expected cluster_radius");
    }

    if let Some(ConfigValue::OptionU64(Some(seed))) = metadata.get_config("clustering_seed") {
        assert_eq!(*seed, 12345);
    } else {
        panic!("Expected clustering_seed");
    }
}

// ========================================================================
// File Size and Performance Tests
// ========================================================================

#[test]
fn test_file_size_tracking() {
    let temp_dir = TempDir::new().unwrap();
    let builder = create_test_builder();
    let matrix = create_test_dense_matrix();

    save_dense_matrix_with_builder(&matrix, temp_dir.path(), "sized", Some(&builder)).unwrap();

    let metadata = load_metadata(temp_dir.path(), "sized").unwrap();
    let file_info = metadata.files.get("matrix").unwrap();

    assert!(file_info.size_bytes.is_some());
    assert!(file_info.size_bytes.unwrap() > 0);
}

#[test]
fn test_metadata_json_format() {
    let temp_dir = TempDir::new().unwrap();
    let builder = create_test_builder();

    let metadata = ArrowSpaceMetadata::from_builder("json_test", &builder);
    save_metadata(&metadata, temp_dir.path(), "json_test").unwrap();

    // Read raw JSON
    let json_path = temp_dir.path().join("json_test_metadata.json");
    let json_str = std::fs::read_to_string(json_path).unwrap();

    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed.is_object());
    assert!(parsed.get("name_id").is_some());
    assert!(parsed.get("builder_config").is_some());
}
