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
    pub tts_language: String,
    pub theme: String,
    pub voice: String,
    pub pitch: f32,
    pub volume: f32,
    /// Voice Style – the delivery character chosen on the TTS page
    /// (standard | calm | formal | friendly). Previously called `emotion`.
    pub voice_style: String,
    /// Speech Tone – the emotional coloring layered on top of the style
    /// (neutral | gentle | informative | conversational).
    pub speech_tone: String,
    /// Approximate pause length inserted at sentence boundaries, in seconds.
    pub pause_sentence: f32,
    /// Approximate pause length inserted at paragraph boundaries, in seconds.
    pub pause_paragraph: f32,
    pub auto_save_directory: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            whisper_model: String::new(),
            piper_executable: String::new(),
            piper_voice: String::new(),
            speech_rate: 1.0,
            language: "auto".into(),
            tts_language: "fa".into(),
            theme: "system".into(),
            voice: "female".into(),
            pitch: 0.0,
            volume: 1.0,
            voice_style: "standard".into(),
            speech_tone: "neutral".into(),
            pause_sentence: 0.5,
            pause_paragraph: 1.2,
            auto_save_directory: default_output_directory(),
        }
    }
}

fn default_output_directory() -> String {
    // Prefer the user's Downloads folder; fall back to the temporary directory.
    if let Some(downloads) = env::var_os("XDG_DOWNLOAD_DIR") {
        return PathBuf::from(downloads).display().to_string();
    }
    let base = if cfg!(target_os = "windows") {
        env::var_os("USERPROFILE").map(PathBuf::from)
    } else {
        env::var_os("HOME").map(PathBuf::from)
    };
    base.map(|directory| directory.join("Downloads").display().to_string())
        .unwrap_or_else(|| env::temp_dir().display().to_string())
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
        assert_eq!(restored.tts_language, "fa");
        assert_eq!(restored.voice_style, "standard");
        assert_eq!(restored.speech_tone, "neutral");
        assert_eq!(restored.pause_sentence, 0.5);
        assert_eq!(restored.pause_paragraph, 1.2);
    }

    #[test]
    fn older_settings_use_defaults_for_new_fields() {
        let restored: AppConfig =
            serde_json::from_str(r#"{"whisper_model":"model.bin"}"#).expect("settings parse");
        assert_eq!(restored.whisper_model, "model.bin");
        assert_eq!(restored.theme, "system");
        assert_eq!(restored.voice, "female");
        assert_eq!(restored.pitch, 0.0);
        assert_eq!(restored.volume, 1.0);
        assert!(!restored.auto_save_directory.is_empty());
    }
}
