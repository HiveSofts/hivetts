use std::io;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Settings error: {0}")]
    Settings(#[from] serde_json::Error),
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, AppError>;
