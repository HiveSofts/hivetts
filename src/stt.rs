use crate::error::{AppError, Result};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

pub trait SpeechRecognizer {
    fn transcribe(&self, model: &Path, audio: &Path, language: &str) -> Result<String>;
}

pub struct WhisperCliRecognizer;

impl SpeechRecognizer for WhisperCliRecognizer {
    fn transcribe(&self, model: &Path, audio: &Path, language: &str) -> Result<String> {
        if !model.is_file() {
            return Err(AppError::Message(format!(
                "Whisper model does not exist: {}",
                model.display()
            )));
        }
        if !audio.is_file() {
            return Err(AppError::Message(format!(
                "Audio file does not exist: {}",
                audio.display()
            )));
        }
        let executable = find_executable("whisper-cli")
            .or_else(|| find_executable("main"))
            .ok_or_else(|| {
                AppError::Message(
                    "Install whisper.cpp and make whisper-cli available on PATH.".into(),
                )
            })?;
        let mut command = Command::new(executable);
        command.args(["-m"]).arg(model).args(["-f"]).arg(audio);
        if language == "auto" || language.is_empty() {
            command.args(["-l", "auto"]);
        } else {
            command.args(["-l", language]);
        }
        let output = command.args(["-nt", "-np"]).output()?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(AppError::Message(format!(
                "Transcription failed ({}): {}",
                output.status,
                if detail.is_empty() {
                    "whisper.cpp returned no diagnostic output."
                } else {
                    &detail
                }
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

fn find_executable(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}
