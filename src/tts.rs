use crate::{
    config::AppConfig,
    error::{AppError, Result},
};
use std::{
    env,
    path::PathBuf,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

pub trait SpeechSynthesizer {
    fn speak(&self, text: &str, config: &AppConfig) -> Result<()>;
    fn export_wav(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()>;
}

pub struct LocalSynthesizer;

impl SpeechSynthesizer for LocalSynthesizer {
    fn speak(&self, text: &str, config: &AppConfig) -> Result<()> {
        validate_text(text)?;
        if !config.piper_executable.is_empty() && !config.piper_voice.is_empty() {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let temp_path = env::temp_dir().join(format!(
                "hivetts-speech-{}-{timestamp}.wav",
                std::process::id()
            ));
            piper_synthesize(text, &temp_path, config)?;
            let result = play_wav(&temp_path);
            if let Err(error) = std::fs::remove_file(&temp_path) {
                eprintln!("Could not remove temporary speech file: {error}");
            }
            return result;
        }
        native_speak(text, config.speech_rate)
    }

    fn export_wav(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()> {
        validate_text(text)?;
        if !config.piper_executable.is_empty() && !config.piper_voice.is_empty() {
            return piper_synthesize(text, destination, config);
        }
        export_native_wav(text, destination, config.speech_rate)
    }
}

fn piper_synthesize(text: &str, destination: &std::path::Path, config: &AppConfig) -> Result<()> {
    let mut child = Command::new(&config.piper_executable)
        .args(["--model", &config.piper_voice, "--length_scale"])
        .arg((1.0 / config.speech_rate.clamp(0.5, 2.0)).to_string())
        .arg("--output_file")
        .arg(destination)
        .stdin(Stdio::piped())
        .spawn()?;
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| AppError::Message("Piper input stream unavailable.".into()))?;
    use std::io::Write;
    input.write_all(text.as_bytes())?;
    drop(input);
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(AppError::Message(format!("Piper exited with {status}.")))
    }
}

fn play_wav(path: &std::path::Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    let player = "afplay";
    #[cfg(target_os = "linux")]
    let player = "ffplay";
    #[cfg(target_os = "windows")]
    let player = "powershell";
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    return Err(AppError::Message(
        "Audio playback is unsupported on this platform.".into(),
    ));

    #[cfg(target_os = "windows")]
    {
        let mut child = Command::new(player)
            .args([
                "-NoProfile",
                "-Command",
                "$p = [Console]::In.ReadToEnd(); (New-Object Media.SoundPlayer $p.Trim()).PlaySync()",
            ])
            .stdin(Stdio::piped())
            .spawn()?;
        if let Some(mut input) = child.stdin.take() {
            use std::io::Write;
            input.write_all(path.to_string_lossy().as_bytes())?;
        }
        let status = child.wait()?;
        return if status.success() {
            Ok(())
        } else {
            Err(AppError::Message(format!(
                "Audio player exited with {status}."
            )))
        };
    }
    #[cfg(not(target_os = "windows"))]
    {
        let mut command = Command::new(player);
        #[cfg(target_os = "linux")]
        command.args(["-nodisp", "-autoexit", "-loglevel", "error"]);
        let status = command.arg(path).status()?;
        if status.success() {
            Ok(())
        } else {
            Err(AppError::Message(format!(
                "Audio player exited with {status}."
            )))
        }
    }
}

fn native_speak(text: &str, rate: f32) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut command = Command::new("say");
    #[cfg(target_os = "linux")]
    let mut command = {
        let exe = find_executable("espeak-ng")
            .or_else(|| find_executable("espeak"))
            .ok_or_else(|| AppError::Message("Install eSpeak NG to enable speech.".into()))?;
        Command::new(exe)
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut cmd = Command::new("powershell");
        cmd.args(["-NoProfile", "-Command", "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak([Console]::In.ReadToEnd())"]);
        cmd
    };
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    return Err(AppError::Message(
        "No native speech engine for this platform.".into(),
    ));

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    command.arg(text);
    #[cfg(target_os = "linux")]
    command.args(["-s", &((175.0 * rate.clamp(0.5, 2.0)) as u32).to_string()]);
    #[cfg(target_os = "windows")]
    command.stdin(Stdio::piped());

    let mut child = command.spawn()?;
    #[cfg(target_os = "windows")]
    if let Some(mut input) = child.stdin.take() {
        use std::io::Write;
        input.write_all(text.as_bytes())?;
    }
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(AppError::Message(format!(
            "Speech engine exited with {status}."
        )))
    }
}

fn export_native_wav(text: &str, destination: &std::path::Path, rate: f32) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("say");
        command
            .args(["-o"])
            .arg(destination)
            .args(["--file-format=WAVE"]);
        command
    };
    #[cfg(target_os = "linux")]
    let mut command = {
        let exe = find_executable("espeak-ng")
            .or_else(|| find_executable("espeak"))
            .ok_or_else(|| AppError::Message("Install eSpeak NG to export WAV.".into()))?;
        let mut command = Command::new(exe);
        command.args(["-w"]).arg(destination);
        command.args(["-s", &((175.0 * rate.clamp(0.5, 2.0)) as u32).to_string()]);
        command
    };
    #[cfg(target_os = "windows")]
    return Err(AppError::Message(
        "WAV export requires Piper on Windows; configure Piper in settings.".into(),
    ));
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    return Err(AppError::Message(
        "WAV export is unsupported on this platform.".into(),
    ));

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let status = command.arg(text).status()?;
        if status.success() {
            Ok(())
        } else {
            Err(AppError::Message(format!(
                "Speech engine exited with {status}."
            )))
        }
    }
}

fn validate_text(text: &str) -> Result<()> {
    if text.trim().is_empty() {
        Err(AppError::Message("Enter some text before speaking.".into()))
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn find_executable(name: &str) -> Option<PathBuf> {
    use std::env;
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}
