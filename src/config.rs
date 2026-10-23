use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AppConfig {
    pub whisper_model: String,
    pub piper_executable: String,
    pub piper_voice: String,
    pub speech_rate: f32,
    pub language: String,
    pub theme: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            whisper_model: String::new(),
            piper_executable: String::new(),
            piper_voice: String::new(),
            speech_rate: 1.0,
            language: "auto".into(),
            theme: "system".into(),
        }
    }
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let path = settings_path()?;
        match fs::read_to_string(path) {
            Ok(contents) => Ok(serde_json::from_str(&contents)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.into()),
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = settings_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(self)?;
        fs::write(path, contents)?;
        Ok(())
    }
}

fn settings_path() -> Result<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Application Support"))
    } else {
        env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }
    .ok_or_else(|| AppError::Message("Could not locate the user settings directory.".into()))?;
    Ok(base.join("hivetts").join("settings.json"))
}

pub fn read_text_file(path: &Path) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_round_trip() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config).expect("default config serializes");
        let restored: AppConfig = serde_json::from_str(&json).expect("config deserializes");
        assert_eq!(restored.language, "auto");
        assert_eq!(restored.speech_rate, 1.0);
    }

    #[test]
    fn older_settings_use_defaults_for_new_fields() {
        let restored: AppConfig =
            serde_json::from_str(r#"{"whisper_model":"model.bin"}"#).expect("settings parse");
        assert_eq!(restored.whisper_model, "model.bin");
        assert_eq!(restored.theme, "system");
    }
}
