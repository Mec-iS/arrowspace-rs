use arrow::{
    array::{Float64Array, RecordBatch, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
};
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::Compression,
    file::properties::WriterProperties,
};
use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::{
    arrays::{Array, Array2},
    matrix::DenseMatrix,
};
use sprs::CsMat;
use std::{collections::HashMap, fs::File, path::Path, sync::Arc};

// Import from your builder module
use crate::{
    builder::{ArrowSpaceBuilder, ConfigValue},
    storage::StorageError,
};

// ============================================================================
// Metadata Storage using ConfigValue
// ============================================================================

/// Metadata container for ArrowSpace configuration and matrix information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrowSpaceMetadata {
    /// Matrix identification
    pub name_id: String,
    pub timestamp: String,

    /// Matrix dimensions
    pub n_rows: usize,
    pub n_cols: usize,

    /// ArrowSpaceBuilder configuration (typed values)
    pub builder_config: HashMap<String, ConfigValue>,

    /// File information
    pub files: HashMap<String, FileInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub filename: String,
    pub file_type: String, // "dense" or "sparse"
    pub rows: usize,
    pub cols: usize,
    pub nnz: Option<usize>, // for sparse matrices
    pub size_bytes: Option<u64>,
}

impl ArrowSpaceMetadata {
    pub fn new(name_id: &str) -> Self {
        Self {
            name_id: name_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            n_rows: 0,
            n_cols: 0,
            builder_config: HashMap::new(),
            files: HashMap::new(),
        }
    }

    /// Create metadata directly from ArrowSpaceBuilder
    pub fn from_builder(name_id: &str, builder: &ArrowSpaceBuilder) -> Self {
        Self {
            name_id: name_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            n_rows: 0,
            n_cols: 0,
            builder_config: builder.builder_config_typed(),
            files: HashMap::new(),
        }
    }

    pub fn with_builder_config(mut self, config: HashMap<String, ConfigValue>) -> Self {
        self.builder_config = config;
        self
    }

    pub fn with_dimensions(mut self, rows: usize, cols: usize) -> Self {
        self.n_rows = rows;
        self.n_cols = cols;
        self
    }

    pub fn add_file(mut self, key: &str, info: FileInfo) -> Self {
        self.files.insert(key.to_string(), info);
        self
    }

    /// Get a typed config value
    pub fn get_config<'a>(&'a self, key: &str) -> Option<&'a ConfigValue> {
        self.builder_config.get(key)
    }

    /// Extract lambda_eps as f64
    pub fn lambda_eps(&self) -> Option<f64> {
        self.get_config("lambda_eps").and_then(|v| v.as_f64())
    }

    /// Extract lambda_k as usize
    pub fn lambda_k(&self) -> Option<usize> {
        self.get_config("lambda_k").and_then(|v| v.as_usize())
    }

    /// Extract synthesis TauMode
    pub fn synthesis(&self) -> Option<crate::taumode::TauMode> {
        self.get_config("synthesis").and_then(|v| v.as_tau_mode())
    }

    /// Get human-readable config summary
    pub fn config_summary(&self) -> String {
        let mut lines = Vec::new();

        for (key, value) in &self.builder_config {
            lines.push(format!("  {} = {}", key, value));
        }

        lines.join("\n")
    }
}

/// Save metadata to JSON file
pub fn save_metadata(
    metadata: &ArrowSpaceMetadata,
    path: impl AsRef<Path>,
    name_id: &str,
) -> Result<(), StorageError> {
    let metadata_path = path.as_ref().join(format!("{}_metadata.json", name_id));

    let json = serde_json::to_string_pretty(metadata)
        .map_err(|e| StorageError::Invalid(format!("Failed to serialize metadata: {}", e)))?;

    std::fs::write(&metadata_path, json)
        .map_err(|e| StorageError::Io(format!("Failed to write metadata: {}", e)))?;

    Ok(())
}

/// Load metadata from JSON file
pub fn load_metadata(
    path: impl AsRef<Path>,
    name_id: &str,
) -> Result<ArrowSpaceMetadata, StorageError> {
    let metadata_path = path.as_ref().join(format!("{}_metadata.json", name_id));

    let json = std::fs::read_to_string(&metadata_path)
        .map_err(|e| StorageError::Io(format!("Failed to read metadata: {}", e)))?;

    let metadata: ArrowSpaceMetadata = serde_json::from_str(&json)
        .map_err(|e| StorageError::Invalid(format!("Failed to parse metadata: {}", e)))?;

    Ok(metadata)
}

// ============================================================================
// Updated Dense Matrix Storage
// ============================================================================

/// Save a DenseMatrix to Parquet with optional builder configuration metadata
///
/// Accepts either typed ConfigValue HashMap or ArrowSpaceBuilder directly.
///
/// # Example
/// ```ignore
/// use arrowspace::builder::ArrowSpaceBuilder;
/// use arrowspace::storage::parquet::save_dense_matrix_with_builder;
/// // Option 1: Pass ArrowSpaceBuilder directly
/// let builder = ArrowSpaceBuilder::new().with_auto_graph(10000, None);
/// save_dense_matrix_with_builder(&matrix, "./data", "embeddings", Some(&builder)).unwrap();
///
/// // Option 2: Pass typed config
/// let config = builder.builder_config_typed();
/// save_dense_matrix(&matrix, "./data", "embeddings", Some(config)).unwrap();
/// ```
pub fn save_dense_matrix_with_builder(
    matrix: &DenseMatrix<f64>,
    path: impl AsRef<Path>,
    name_id: &str,
    builder: Option<&ArrowSpaceBuilder>,
) -> Result<(), StorageError> {
    let config = builder.map(|b| b.builder_config_typed());
    save_dense_matrix(matrix, path, name_id, config)
}

/// Save a DenseMatrix to Parquet with optional typed configuration metadata
pub fn save_dense_matrix(
    matrix: &DenseMatrix<f64>,
    path: impl AsRef<Path>,
    name_id: &str,
    builder_config: Option<HashMap<String, ConfigValue>>,
) -> Result<(), StorageError> {
    let (n_rows, n_cols) = matrix.shape();

    // [Previous Parquet saving code remains the same until metadata creation]

    // Build schema: metadata + column vectors
    let mut fields = vec![
        Field::new("name_id", DataType::Utf8, false),
        Field::new("n_rows", DataType::UInt64, false),
        Field::new("n_cols", DataType::UInt64, false),
    ];

    for i in 0..n_cols {
        fields.push(Field::new(format!("col_{}", i), DataType::Float64, false));
    }

    let schema = Arc::new(Schema::new(fields));

    let name_array = StringArray::from(vec![name_id; n_rows]);
    let n_rows_array = UInt64Array::from(vec![n_rows as u64; n_rows]);
    let n_cols_array = UInt64Array::from(vec![n_cols as u64; n_rows]);

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(name_array),
        Arc::new(n_rows_array),
        Arc::new(n_cols_array),
    ];

    for col_idx in 0..n_cols {
        let col_data: Vec<f64> = (0..n_rows)
            .map(|row_idx| *matrix.get((row_idx, col_idx)))
            .collect();
        columns.push(Arc::new(Float64Array::from(col_data)));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| StorageError::Arrow(e.to_string()))?;

    let file_path = path.as_ref().join(format!("{}.parquet", name_id));
    let file = File::create(&file_path).map_err(|e| StorageError::Io(e.to_string()))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    writer
        .write(&batch)
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    writer
        .close()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    // Get file size
    let file_size = std::fs::metadata(&file_path).map(|m| m.len()).ok();

    // Create and save metadata if builder config is provided
    if let Some(config) = builder_config {
        let metadata = ArrowSpaceMetadata::new(name_id)
            .with_builder_config(config)
            .with_dimensions(n_rows, n_cols)
            .add_file(
                "matrix",
                FileInfo {
                    filename: format!("{}.parquet", name_id),
                    file_type: "dense".to_string(),
                    rows: n_rows,
                    cols: n_cols,
                    nnz: None,
                    size_bytes: file_size,
                },
            );

        save_metadata(&metadata, path.as_ref(), name_id)?;
    }

    Ok(())
}

/// Load a DenseMatrix from Parquet
///
/// Reconstructs the matrix in column-major format.
///
/// # Arguments
/// * `path` - Full path to the parquet file (including .parquet extension)
///
/// # Example
/// ```ignore
/// use arrowspace::storage::parquet::load_dense_matrix;
/// let matrix = load_dense_matrix("./data/my_matrix.parquet").unwrap();
/// ```
pub fn load_dense_matrix(path: impl AsRef<Path>) -> Result<DenseMatrix<f64>, StorageError> {
    let file = File::open(path.as_ref()).map_err(|e| StorageError::Io(e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    let mut reader = builder
        .build()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    let batch = reader
        .next()
        .ok_or_else(|| StorageError::Invalid("No data in parquet file".to_string()))?
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    // Extract dimensions from first row
    let n_rows_col = batch
        .column_by_name("n_rows")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StorageError::Invalid("n_rows column missing".to_string()))?;
    let n_rows = n_rows_col.value(0) as usize;

    let n_cols_col = batch
        .column_by_name("n_cols")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StorageError::Invalid("n_cols column missing".to_string()))?;
    let n_cols = n_cols_col.value(0) as usize;

    // Extract data in column-major order
    let mut flat_data = Vec::with_capacity(n_rows * n_cols);

    for col_idx in 0..n_cols {
        let col_name = format!("col_{}", col_idx);
        let col = batch
            .column_by_name(&col_name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| StorageError::Invalid(format!("Column {} missing", col_name)))?;

        for row_idx in 0..n_rows {
            flat_data.push(col.value(row_idx));
        }
    }

    // Reconstruct with axis=1 (column-major)
    let matrix = DenseMatrix::from_iterator(flat_data.into_iter(), n_rows, n_cols, 1);

    Ok(matrix)
}

// ============================================================================
// Updated Sparse Matrix Storage
// ============================================================================

pub fn save_sparse_matrix_with_builder(
    matrix: &CsMat<f64>,
    path: impl AsRef<Path>,
    name_id: &str,
    builder: Option<&ArrowSpaceBuilder>,
) -> Result<(), StorageError> {
    let config = builder.map(|b| b.builder_config_typed());
    save_sparse_matrix(matrix, path, name_id, config)
}

pub fn save_sparse_matrix(
    matrix: &CsMat<f64>,
    path: impl AsRef<Path>,
    name_id: &str,
    builder_config: Option<HashMap<String, ConfigValue>>,
) -> Result<(), StorageError> {
    // [Previous sparse saving code remains the same until metadata creation]

    let (n_rows, n_cols) = matrix.shape();
    let nnz = matrix.nnz();

    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);

    for (row_idx, row_vec) in matrix.outer_iterator().enumerate() {
        for (col_idx, &value) in row_vec.iter() {
            rows.push(row_idx as u64);
            cols.push(col_idx as u64);
            vals.push(value);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("name_id", DataType::Utf8, false),
        Field::new("n_rows", DataType::UInt64, false),
        Field::new("n_cols", DataType::UInt64, false),
        Field::new("nnz", DataType::UInt64, false),
        Field::new("row", DataType::UInt64, false),
        Field::new("col", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let name_array = StringArray::from(vec![name_id; nnz]);
    let n_rows_array = UInt64Array::from(vec![n_rows as u64; nnz]);
    let n_cols_array = UInt64Array::from(vec![n_cols as u64; nnz]);
    let nnz_array = UInt64Array::from(vec![nnz as u64; nnz]);
    let row_array = UInt64Array::from(rows);
    let col_array = UInt64Array::from(cols);
    let val_array = Float64Array::from(vals);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(name_array),
            Arc::new(n_rows_array),
            Arc::new(n_cols_array),
            Arc::new(nnz_array),
            Arc::new(row_array),
            Arc::new(col_array),
            Arc::new(val_array),
        ],
    )
    .map_err(|e| StorageError::Arrow(e.to_string()))?;

    let file_path = path.as_ref().join(format!("{}.parquet", name_id));
    let file = File::create(&file_path).map_err(|e| StorageError::Io(e.to_string()))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    writer
        .write(&batch)
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    writer
        .close()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    let file_size = std::fs::metadata(&file_path).map(|m| m.len()).ok();

    if let Some(config) = builder_config {
        let metadata = ArrowSpaceMetadata::new(name_id)
            .with_builder_config(config)
            .with_dimensions(n_rows, n_cols)
            .add_file(
                "matrix",
                FileInfo {
                    filename: format!("{}.parquet", name_id),
                    file_type: "sparse".to_string(),
                    rows: n_rows,
                    cols: n_cols,
                    nnz: Some(nnz),
                    size_bytes: file_size,
                },
            );

        save_metadata(&metadata, path.as_ref(), name_id)?;
    }

    Ok(())
}

/// Load a sparse matrix from Parquet
///
/// Reconstructs CSR matrix from COO triplets.
///
/// # Arguments
/// * `path` - Full path to the parquet file (including .parquet extension)
///
/// # Example
/// ```ignore
/// let matrix = load_sparse_matrix("./data/sparse_matrix.parquet").unwrap();
/// ```
pub fn load_sparse_matrix(path: impl AsRef<Path>) -> Result<CsMat<f64>, StorageError> {
    let file = File::open(path.as_ref()).map_err(|e| StorageError::Io(e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    let mut reader = builder
        .build()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    let batch = reader
        .next()
        .ok_or_else(|| StorageError::Invalid("No data in parquet file".to_string()))?
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    // Extract dimensions
    let n_rows_col = batch
        .column_by_name("n_rows")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StorageError::Invalid("n_rows missing".to_string()))?;
    let n_rows = n_rows_col.value(0) as usize;

    let n_cols_col = batch
        .column_by_name("n_cols")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StorageError::Invalid("n_cols missing".to_string()))?;
    let n_cols = n_cols_col.value(0) as usize;

    // Extract triplets
    let row_col = batch
        .column_by_name("row")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StorageError::Invalid("row missing".to_string()))?;

    let col_col = batch
        .column_by_name("col")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StorageError::Invalid("col missing".to_string()))?;

    let val_col = batch
        .column_by_name("value")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .ok_or_else(|| StorageError::Invalid("value missing".to_string()))?;

    // Build sparse matrix from triplets
    use sprs::TriMat;
    let mut trimat = TriMat::new((n_rows, n_cols));

    for i in 0..row_col.len() {
        trimat.add_triplet(
            row_col.value(i) as usize,
            col_col.value(i) as usize,
            val_col.value(i),
        );
    }

    Ok(trimat.to_csr())
}

// ============================================================================
// Checkpoint with ArrowSpaceBuilder
// ============================================================================

/// Save all ArrowSpace artifacts using ArrowSpaceBuilder directly
///
/// This is the recommended approach - pass the builder directly.
pub fn save_arrowspace_checkpoint_with_builder(
    path: impl AsRef<Path>,
    checkpoint_name: &str,
    raw_data: &DenseMatrix<f64>,
    adjacency: &CsMat<f64>,
    centroids: &DenseMatrix<f64>,
    laplacian: &CsMat<f64>,
    signals: &CsMat<f64>,
    builder: &ArrowSpaceBuilder,
) -> Result<(), StorageError> {
    let base_path = path.as_ref();

    std::fs::create_dir_all(base_path)
        .map_err(|e| StorageError::Io(format!("Failed to create directory: {}", e)))?;

    // Save all matrices (without individual metadata)
    save_dense_matrix(
        raw_data,
        base_path,
        &format!("{}_raw_data", checkpoint_name),
        None,
    )?;
    save_sparse_matrix(
        adjacency,
        base_path,
        &format!("{}_adjacency", checkpoint_name),
        None,
    )?;
    save_dense_matrix(
        centroids,
        base_path,
        &format!("{}_centroids", checkpoint_name),
        None,
    )?;
    save_sparse_matrix(
        laplacian,
        base_path,
        &format!("{}_laplacian", checkpoint_name),
        None,
    )?;
    save_sparse_matrix(
        signals,
        base_path,
        &format!("{}_signals", checkpoint_name),
        None,
    )?;

    // Create comprehensive metadata using builder
    let mut metadata = ArrowSpaceMetadata::from_builder(checkpoint_name, builder)
        .with_dimensions(raw_data.shape().0, raw_data.shape().1);

    let artifacts = vec![
        ("raw_data", "dense", raw_data.shape(), None),
        (
            "adjacency",
            "sparse",
            adjacency.shape(),
            Some(adjacency.nnz()),
        ),
        ("centroids", "dense", centroids.shape(), None),
        (
            "laplacian",
            "sparse",
            laplacian.shape(),
            Some(laplacian.nnz()),
        ),
        ("signals", "sparse", signals.shape(), Some(signals.nnz())),
    ];

    for (name, file_type, (rows, cols), nnz) in artifacts {
        let filename = format!("{}_{}.parquet", checkpoint_name, name);
        let file_size = std::fs::metadata(base_path.join(&filename))
            .map(|m| m.len())
            .ok();

        metadata = metadata.add_file(
            name,
            FileInfo {
                filename,
                file_type: file_type.to_string(),
                rows,
                cols,
                nnz,
                size_bytes: file_size,
            },
        );
    }

    save_metadata(&metadata, base_path, checkpoint_name)?;

    Ok(())
}

/// Save a lambda vector (per-row computed values) with ArrowSpaceBuilder metadata
///
/// # Arguments
/// * `lambdas` - Vector of lambda values, one per data row
/// * `path` - Directory path where file will be saved
/// * `name_id` - Identifier for the lambda file (e.g., "lambda_values", "row_lambdas")
/// * `builder` - Optional ArrowSpaceBuilder for metadata tracking
///
/// # Example
/// ```ignore
/// use arrowspace::builder::ArrowSpaceBuilder;
/// use arrowspace::storage::parquet::save_lambda_with_builder;
/// let lambdas = vec![0.5, 0.6, 0.7, 0.8];
/// let builder = ArrowSpaceBuilder::default();
/// save_lambda_with_builder(&lambdas, "./data", "lambda_values", Some(&builder)).unwrap();
/// ```
pub fn save_lambda_with_builder(
    lambdas: &[f64],
    path: impl AsRef<Path>,
    name_id: &str,
    builder: Option<&ArrowSpaceBuilder>,
) -> Result<(), StorageError> {
    let config = builder.map(|b| b.builder_config_typed());
    save_lambda(lambdas, path, name_id, config)
}

/// Save a lambda vector to Parquet with optional typed configuration metadata
///
/// Stores the lambda values as a single-column Parquet file with metadata
/// about the vector dimensions and optional builder configuration.
///
/// # Arguments
/// * `lambdas` - Vector of lambda values (one per row in the original data)
/// * `path` - Directory path where file will be saved
/// * `name_id` - Identifier for the lambda file
/// * `builder_config` - Optional ArrowSpaceBuilder configuration to store
///
/// # Example
/// ```ignore
/// use arrowspace::storage::parquet::save_lambda;
/// let lambdas = vec![0.1, 0.2, 0.3];
/// save_lambda(&lambdas, "./data", "lambdas", None).unwrap();
/// ```
pub fn save_lambda(
    lambdas: &[f64],
    path: impl AsRef<Path>,
    name_id: &str,
    builder_config: Option<HashMap<String, ConfigValue>>,
) -> Result<(), StorageError> {
    let n_values = lambdas.len();

    if n_values == 0 {
        return Err(StorageError::Invalid(
            "Cannot save empty lambda vector".to_string(),
        ));
    }

    // Build schema: metadata + lambda values column
    let schema = Arc::new(Schema::new(vec![
        Field::new("name_id", DataType::Utf8, false),
        Field::new("n_values", DataType::UInt64, false),
        Field::new("row_index", DataType::UInt64, false),
        Field::new("lambda", DataType::Float64, false),
    ]));

    // Build arrays
    let name_array = StringArray::from(vec![name_id; n_values]);
    let n_values_array = UInt64Array::from(vec![n_values as u64; n_values]);
    let row_index_array = UInt64Array::from((0..n_values as u64).collect::<Vec<_>>());
    let lambda_array = Float64Array::from(lambdas.to_vec());

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(name_array),
            Arc::new(n_values_array),
            Arc::new(row_index_array),
            Arc::new(lambda_array),
        ],
    )
    .map_err(|e| StorageError::Arrow(e.to_string()))?;

    // Write to Parquet with Snappy compression
    let file_path = path.as_ref().join(format!("{}.parquet", name_id));
    let file = File::create(&file_path).map_err(|e| StorageError::Io(e.to_string()))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    writer
        .write(&batch)
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    writer
        .close()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    // Get file size
    let file_size = std::fs::metadata(&file_path).map(|m| m.len()).ok();

    // Create and save metadata if builder config is provided
    if let Some(config) = builder_config {
        let metadata = ArrowSpaceMetadata::new(name_id)
            .with_builder_config(config)
            .with_dimensions(n_values, 1) // n_values rows, 1 column
            .add_file(
                "lambda_vector",
                FileInfo {
                    filename: format!("{}.parquet", name_id),
                    file_type: "lambda_vector".to_string(),
                    rows: n_values,
                    cols: 1,
                    nnz: None,
                    size_bytes: file_size,
                },
            );

        save_metadata(&metadata, path.as_ref(), name_id)?;
    }

    Ok(())
}

/// Load a lambda vector from Parquet
///
/// # Arguments
/// * `path` - Full path to the parquet file (including .parquet extension)
///
/// # Returns
/// Vector of lambda values in the original row order
///
/// # Example
/// ```ignore
/// use arrowspace::storage::parquet::load_lambda;
/// let lambdas = load_lambda("./data/lambda_values.parquet").unwrap();
/// println!("Loaded {} lambda values", lambdas.len());
/// ```
pub fn load_lambda(path: impl AsRef<Path>) -> Result<Vec<f64>, StorageError> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path.as_ref()).map_err(|e| StorageError::Io(e.to_string()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    let mut reader = builder
        .build()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    let batch = reader
        .next()
        .ok_or_else(|| StorageError::Invalid("No data in parquet file".to_string()))?
        .map_err(|e| StorageError::Parquet(e.to_string()))?;

    // Extract n_values from first row
    let n_values_col = batch
        .column_by_name("n_values")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StorageError::Invalid("n_values column missing".to_string()))?;
    let n_values = n_values_col.value(0) as usize;

    // Extract lambda values
    let lambda_col = batch
        .column_by_name("lambda")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .ok_or_else(|| StorageError::Invalid("lambda column missing".to_string()))?;

    let lambdas: Vec<f64> = (0..n_values).map(|i| lambda_col.value(i)).collect();

    Ok(lambdas)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::test_storage::{
        create_test_builder, create_test_dense_matrix, create_test_dense_matrix_with_size,
        create_test_sparse_matrix, create_test_sparse_matrix_with_size,
    };
    use approx::assert_relative_eq;
    use arrow::datatypes::SchemaRef;
    use sprs::TriMat;
    use std::fs;

    /// Helper function for *_multibatch tests
    fn create_forced_multibatch_parquet(
        path: impl AsRef<Path>,
        schema: SchemaRef,
        batches: impl Iterator<Item = RecordBatch>,
    ) {
        let file = File::create(path).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(1024)
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();

        for batch in batches {
            writer.write(&batch).unwrap();
        }
        writer.close().unwrap();
    }

    #[test]
    fn test_dense_roundtrip() {
        let _ = fs::create_dir_all("./test_data");

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![1e-3, 1e-5, 0.34235236234234],
        ];
        let original = DenseMatrix::from_2d_vec(&data).unwrap();

        save_dense_matrix(&original, "./test_data", "test_dense", None).unwrap();

        let loaded = load_dense_matrix("./test_data/test_dense.parquet").unwrap();

        assert_eq!(original.shape(), loaded.shape());

        let (rows, cols) = original.shape();
        for i in 0..rows {
            for j in 0..cols {
                assert_relative_eq!(*original.get((i, j)), *loaded.get((i, j)), epsilon = 1e-10);
            }
        }

        // let _ = fs::remove_dir_all("./test_data");
    }

    #[test]
    fn test_sparse_roundtrip() {
        let _ = fs::create_dir_all("./test_data");

        let mut trimat = TriMat::new((4, 4));
        trimat.add_triplet(0, 0, 2.0);
        trimat.add_triplet(0, 1, -1.0);
        trimat.add_triplet(1, 1, 3.0);
        trimat.add_triplet(2, 2, 1.5);
        let original = trimat.to_csr();

        save_sparse_matrix(&original, "./test_data", "test_sparse", None).unwrap();

        let loaded = load_sparse_matrix("./test_data/test_sparse.parquet").unwrap();

        assert_eq!(original.shape(), loaded.shape());
        assert_eq!(original.nnz(), loaded.nnz());

        for i in 0..4 {
            for j in 0..4 {
                let orig_val = original.get(i, j).copied().unwrap_or(0.0);
                let loaded_val = loaded.get(i, j).copied().unwrap_or(0.0);
                assert_relative_eq!(orig_val, loaded_val, epsilon = 1e-10);
            }
        }

        //let _ = fs::remove_dir_all("./test_data");
    }

    #[test]
    fn test_checkpoint_save_all_artifacts() {
        let temp_dir = Path::new("./test_data");
        let builder = create_test_builder();

        let raw_data = create_test_dense_matrix();
        let centroids = DenseMatrix::from_2d_vec(&vec![vec![1.5, 2.5, 3.5]]).unwrap();
        let adjacency = create_test_sparse_matrix();
        let laplacian = create_test_sparse_matrix();
        let signals = create_test_sparse_matrix();

        save_arrowspace_checkpoint_with_builder(
            temp_dir,
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
            let path = temp_dir.join(filename);
            assert!(path.exists(), "Missing file: {}", filename);
        }
    }

    #[test]
    fn test_dense_multibatch() {
        let _ = fs::create_dir_all("./test_data");
        let path = Path::new("./test_data/test_dense_multibatch.parquet");

        // Create a matrix with 2000 rows. This exceeds the default Arrow batch size (1024),
        // forcing the reader to process multiple RecordBatches.
        let rows = 2000;
        let cols = 2;
        let matrix = create_test_dense_matrix_with_size(rows, cols);

        let schema = Arc::new(Schema::new(vec![
            Field::new("name_id", DataType::Utf8, false),
            Field::new("n_rows", DataType::UInt64, false),
            Field::new("n_cols", DataType::UInt64, false),
            Field::new("col_0", DataType::Float64, false),
            Field::new("col_1", DataType::Float64, false),
        ]));

        // Create batches
        let batches = (0..2).map(|i| {
            let start = i * 1000;
            let end = std::cmp::min(start + 1000, rows);
            let len = end - start;

            let name_arr = StringArray::from(vec!["test"; len]);
            let n_rows_arr = UInt64Array::from(vec![rows as u64; len]);
            let n_cols_arr = UInt64Array::from(vec![cols as u64; len]);

            let col0_data: Vec<f64> = (start..end).map(|r| matrix.get((r, 0)).clone()).collect();
            let col1_data: Vec<f64> = (start..end).map(|r| matrix.get((r, 1)).clone()).collect();

            let col0_arr = Float64Array::from(col0_data);
            let col1_arr = Float64Array::from(col1_data);

            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(name_arr),
                    Arc::new(n_rows_arr),
                    Arc::new(n_cols_arr),
                    Arc::new(col0_arr),
                    Arc::new(col1_arr),
                ],
            )
            .unwrap()
        });

        create_forced_multibatch_parquet(path, schema.clone(), batches);

        // Verify that load_dense_matrix correctly concatenates all batches.
        let loaded = load_dense_matrix(path).unwrap();

        assert_eq!(loaded.shape(), (rows, cols), "Loaded matrix shape mismatch");
    }

    #[test]
    fn test_sparse_multibatch() {
        let _ = fs::create_dir_all("./test_data");
        let path = Path::new("./test_data/test_sparse_multibatch.parquet");

        // Create a sparse matrix with 2000 rows.
        let rows = 2000;
        let cols = 10;
        let matrix = create_test_sparse_matrix_with_size(rows, cols);
        let nnz = matrix.nnz();

        let schema = Arc::new(Schema::new(vec![
            Field::new("name_id", DataType::Utf8, false),
            Field::new("n_rows", DataType::UInt64, false),
            Field::new("n_cols", DataType::UInt64, false),
            Field::new("nnz", DataType::UInt64, false),
            Field::new("row", DataType::UInt64, false),
            Field::new("col", DataType::UInt64, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let chunk_size = 1000;
        let total_chunks = (nnz + chunk_size - 1) / chunk_size;

        let batches = (0..total_chunks).map(|i| {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, nnz);
            let len = end - start;

            let name_arr = StringArray::from(vec!["test"; len]);
            let n_rows_arr = UInt64Array::from(vec![rows as u64; len]);
            let n_cols_arr = UInt64Array::from(vec![cols as u64; len]);
            let nnz_arr = UInt64Array::from(vec![nnz as u64; len]);

            let mut chunk_rows = Vec::with_capacity(len);
            let mut chunk_cols = Vec::with_capacity(len);
            let mut chunk_vals = Vec::with_capacity(len);

            for idx in start..end {
                chunk_rows.push(idx as u64);
                chunk_cols.push((idx % cols) as u64);
                chunk_vals.push(1.0);
            }

            let row_arr = UInt64Array::from(chunk_rows);
            let col_arr = UInt64Array::from(chunk_cols);
            let val_arr = Float64Array::from(chunk_vals);

            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(name_arr),
                    Arc::new(n_rows_arr),
                    Arc::new(n_cols_arr),
                    Arc::new(nnz_arr),
                    Arc::new(row_arr),
                    Arc::new(col_arr),
                    Arc::new(val_arr),
                ],
            )
            .unwrap()
        });

        create_forced_multibatch_parquet(path, schema.clone(), batches);

        let loaded = load_sparse_matrix(path).unwrap();

        // If bug exists, loaded.nnz() will be ~1024 instead of 2000
        assert_eq!(loaded.nnz(), nnz, "Loaded sparse matrix nnz mismatch");
        assert_eq!(
            loaded.shape(),
            (rows, cols),
            "Loaded sparse matrix shape mismatch"
        );
    }

    #[test]
    fn test_lambda_multibatch() {
        let _ = fs::create_dir_all("./test_data");
        let path = Path::new("./test_data/test_lambda_multibatch.parquet");

        let n_values = 2000;
        let lambdas: Vec<_> = (0..n_values).map(|i| i as f64).collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("name_id", DataType::Utf8, false),
            Field::new("n_values", DataType::UInt64, false),
            Field::new("row_index", DataType::UInt64, false),
            Field::new("lambda", DataType::Float64, false),
        ]));

        let chunk_size = 1000;
        let total_chunks = (n_values + chunk_size - 1) / chunk_size;

        let batches = (0..total_chunks).map(|i| {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, n_values);
            let len = end - start;

            let name_arr = StringArray::from(vec!["test"; len]);
            let n_vals_arr = UInt64Array::from(vec![n_values as u64; len]);
            let row_idx_arr = UInt64Array::from((start as u64..end as u64).collect::<Vec<_>>());
            let lambda_arr = Float64Array::from(lambdas[start..end].to_vec());

            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(name_arr),
                    Arc::new(n_vals_arr),
                    Arc::new(row_idx_arr),
                    Arc::new(lambda_arr),
                ],
            )
            .unwrap()
        });

        create_forced_multibatch_parquet(path, schema.clone(), batches);

        let loaded = load_lambda(path).unwrap();

        assert_eq!(loaded.len(), n_values, "Loaded lambda vector length mismatch");
    }
}
