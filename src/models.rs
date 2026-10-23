use crate::error::{AppError, Result};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperSize {
    Tiny,
    Base,
    Small,
}

impl WhisperSize {
    pub fn filename(self) -> &'static str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
        }
    }

    fn url(self) -> String {
        format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            self.filename()
        )
    }
}

pub fn models_directory() -> Result<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Application Support"))
    } else {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
    }
    .ok_or_else(|| AppError::Message("Could not locate the user data directory.".into()))?;
    Ok(base.join("hivetts").join("models"))
}

pub fn list_models() -> Result<Vec<PathBuf>> {
    let directory = models_directory()?;
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut models = fs::read_dir(directory)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect::<Vec<_>>();
    models.sort();
    Ok(models)
}

pub fn download_model(size: WhisperSize) -> Result<PathBuf> {
    let directory = models_directory()?;
    fs::create_dir_all(&directory)?;
    let destination = directory.join(size.filename());
    if destination.is_file() {
        return Ok(destination);
    }
    let temporary = directory.join(format!("{}.part-{}", size.filename(), std::process::id()));
    let curl = find_executable("curl")
        .ok_or_else(|| AppError::Message("Install curl to download speech models.".into()))?;
    let status = Command::new(curl)
        .args(["--fail", "--location", "--progress-bar", "--output"])
        .arg(&temporary)
        .arg(size.url())
        .status()?;
    if !status.success() {
        if let Err(error) = fs::remove_file(&temporary) {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!("Could not remove partial model download: {error}");
            }
        }
        return Err(AppError::Message(format!(
            "Model download failed with status {status}."
        )));
    }
    fs::rename(&temporary, &destination)?;
    Ok(destination)
}

pub fn delete_model(path: &Path) -> Result<()> {
    let directory = models_directory()?;
    let canonical_directory = directory.canonicalize()?;
    let canonical_file = path.canonicalize()?;
    if !canonical_file.starts_with(&canonical_directory) {
        return Err(AppError::Message(
            "Only models inside the HiveTTS models folder can be deleted.".into(),
        ));
    }
    fs::remove_file(canonical_file)?;
    Ok(())
}

fn find_executable(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}
