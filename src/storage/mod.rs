pub mod parquet;

#[cfg(test)]
#[cfg(feature = "storage")]
mod test_storage;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum StorageError {
    Io(String),
    Arrow(String),
    Serde(String),
    Invalid(String),
    Parquet(String)
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(e) => write!(f, "IO error: {}", e),
            StorageError::Arrow(e) => write!(f, "Arrow error: {}", e),
            StorageError::Serde(e) => write!(f, "Serde error: {}", e),
            StorageError::Parquet(e) => write!(f, "Parquet error: {}", e),
            StorageError::Invalid(e) => write!(f, "Invalid: {}", e),
        }
    }
}

impl std::error::Error for StorageError {}

pub type StorageResult<T> = Result<T, StorageError>;
