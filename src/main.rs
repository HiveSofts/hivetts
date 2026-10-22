slint::include_modules!();

use std::{
    env, fs,
    io::{self, Read, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn main() {
    let app = AppWindow::new().unwrap();

    if let Ok(model_path) = env::var("WHISPER_MODEL") {
        app.set_model_path(model_path.into());
    }

    let recording: Arc<Mutex<Option<(Child, PathBuf)>>> = Arc::new(Mutex::new(None));

    let weak = app.as_weak();
    app.on_speak(move |text| {
        let weak = weak.clone();
        thread::spawn(move || {
            let result = speak(&text);
            set_status(weak, result.unwrap_or_else(|error| error));
        });
    });

    let weak = app.as_weak();
    let active_recording = Arc::clone(&recording);
    app.on_start_recording(move || {
        let mut active = active_recording.lock().unwrap();
        if active.is_some() {
            set_status(weak.clone(), "A recording is already in progress.".into());
            return;
        }

        let path = recording_path();
        match start_recording(&path) {
            Ok(child) => {
                *active = Some((child, path));
                if let Some(app) = weak.upgrade() {
                    app.set_is_recording(true);
                    app.set_status_text("Recording…".into());
                }
                monitor_recording(Arc::clone(&active_recording), weak.clone());
            }
            Err(error) => {
                if let Some(app) = weak.upgrade() {
                    app.set_status_text(format!("Could not start recording: {error}").into());
                }
            }
        }
    });

    let weak = app.as_weak();
    let active_recording = Arc::clone(&recording);
    app.on_stop_recording(move || {
        let recording = active_recording.lock().unwrap().take();
        let Some((mut child, path)) = recording else {
            if let Some(app) = weak.upgrade() {
                app.set_is_recording(false);
                app.set_status_text("There is no active recording.".into());
            }
            return;
        };

        if let Some(app) = weak.upgrade() {
            app.set_is_recording(false);
            app.set_status_text("Finishing recording…".into());
        }

        let weak = weak.clone();
        thread::spawn(move || {
            let stop_result = child
                .stdin
                .as_mut()
                .ok_or_else(|| io::Error::other("recording process stdin is unavailable"))
                .and_then(|stdin| stdin.write_all(b"q\n"));

            if let Err(error) = stop_result {
                if let Err(cleanup_error) = child.kill() {
                    eprintln!("Could not terminate recording process: {cleanup_error}");
                }
                if let Err(cleanup_error) = child.wait() {
                    eprintln!("Could not reap recording process: {cleanup_error}");
                }
                remove_recording(&path);
                set_status(weak, format!("Could not stop recording: {error}"));
                return;
            }

            match child.wait() {
                Ok(status) if status.success() => {
                    transcribe_recording(weak, path);
                }
                Ok(status) => {
                    let diagnostic = child
                        .stderr
                        .take()
                        .map(read_stderr)
                        .unwrap_or_else(|| "No diagnostic output available.".into());
                    remove_recording(&path);
                    set_status(weak, format!("Recording failed ({status}): {diagnostic}"));
                }
                Err(error) => {
                    remove_recording(&path);
                    set_status(weak, format!("Could not finish recording: {error}"));
                }
            }
        });
    });

    app.run().unwrap();
}

fn set_status(weak: slint::Weak<AppWindow>, message: String) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            app.set_status_text(message.into());
        }
    }) {
        eprintln!("Could not update the UI: {error}");
    }
}

fn speak(text: &str) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("Enter some text before selecting Speak.".into());
    }

    #[cfg(target_os = "macos")]
    let (program, args): (PathBuf, Vec<&str>) = (
        find_executable("say").ok_or("The macOS `say` command was not found.".to_string())?,
        vec![],
    );

    #[cfg(target_os = "linux")]
    let (program, args): (PathBuf, Vec<&str>) = {
        let executable = find_executable("espeak-ng")
            .or_else(|| find_executable("espeak"))
            .ok_or("Install eSpeak NG (`espeak-ng`) to enable speech.".to_string())?;
        (executable, vec![])
    };

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return Err("Text-to-speech is currently supported on Linux and macOS.".into());

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let status = Command::new(program)
            .args(args)
            .arg(text)
            .status()
            .map_err(|error| format!("Could not start speech engine: {error}"))?;
        if status.success() {
            Ok("Speech finished.".into())
        } else {
            Err(format!("Speech engine exited with {status}."))
        }
    }
}

fn recording_path() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    env::temp_dir().join(format!(
        "hivetts-recording-{}-{timestamp}.wav",
        std::process::id()
    ))
}

#[cfg(target_os = "linux")]
fn start_recording(path: &PathBuf) -> io::Result<Child> {
    let ffmpeg = find_executable("ffmpeg")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Install FFmpeg to record audio"))?;
    Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "pulse",
            "-i",
            "default",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
        ])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
}

#[cfg(target_os = "macos")]
fn start_recording(path: &PathBuf) -> io::Result<Child> {
    let ffmpeg = find_executable("ffmpeg")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Install FFmpeg to record audio"))?;
    Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "avfoundation",
            "-i",
            ":0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
        ])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn start_recording(_path: &PathBuf) -> io::Result<Child> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Microphone recording is currently supported on Linux and macOS",
    ))
}

fn monitor_recording(
    recording: Arc<Mutex<Option<(Child, PathBuf)>>>,
    weak: slint::Weak<AppWindow>,
) {
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        let failure = {
            let mut active = recording.lock().unwrap();
            let Some((child, path)) = active.as_mut() else {
                return;
            };
            match child.try_wait() {
                Ok(None) => return,
                Ok(Some(status)) => {
                    let diagnostic = child
                        .stderr
                        .take()
                        .map(read_stderr)
                        .unwrap_or_else(|| "No diagnostic output available.".into());
                    let path = path.clone();
                    *active = None;
                    remove_recording(&path);
                    format!("Recording stopped unexpectedly ({status}): {diagnostic}")
                }
                Err(error) => {
                    if let Err(kill_error) = child.kill() {
                        eprintln!("Could not terminate recording process: {kill_error}");
                    }
                    if let Err(wait_error) = child.wait() {
                        eprintln!("Could not reap recording process: {wait_error}");
                    }
                    let path = path.clone();
                    *active = None;
                    remove_recording(&path);
                    format!("Could not check recording status: {error}")
                }
            }
        };
        if let Err(error) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_is_recording(false);
                app.set_status_text(failure.into());
            }
        }) {
            eprintln!("Could not update the UI: {error}");
        }
    });
}

fn transcribe_recording(weak: slint::Weak<AppWindow>, path: PathBuf) {
    let Some(app) = weak.upgrade() else {
        remove_recording(&path);
        return;
    };
    let model_path = app.get_model_path().to_string();
    drop(app);

    if model_path.trim().is_empty() {
        remove_recording(&path);
        set_status(
            weak,
            "Select a whisper.cpp model file to transcribe the recording.".into(),
        );
        return;
    }
    let model_path = PathBuf::from(model_path);
    if !model_path.is_file() {
        remove_recording(&path);
        set_status(
            weak,
            format!(
                "Whisper model file does not exist: {}",
                model_path.display()
            ),
        );
        return;
    }

    set_status(weak.clone(), "Transcribing recording…".into());
    thread::spawn(move || {
        let result = transcribe(&model_path, &path);
        remove_recording(&path);
        match result {
            Ok(text) if text.trim().is_empty() => {
                set_status(weak, "No speech was recognized.".into());
            }
            Ok(text) => {
                if let Err(error) = slint::invoke_from_event_loop(move || {
                    if let Some(app) = weak.upgrade() {
                        app.set_speech_text(text.into());
                        app.set_status_text("Transcription finished.".into());
                    }
                }) {
                    eprintln!("Could not update the UI: {error}");
                }
            }
            Err(error) => set_status(weak, error),
        }
    });
}

fn transcribe(model_path: &PathBuf, audio_path: &PathBuf) -> Result<String, String> {
    let executable = find_executable("whisper-cli")
        .or_else(|| find_executable("main"))
        .ok_or("Install whisper.cpp and make `whisper-cli` available on PATH.".to_string())?;
    let output = Command::new(executable)
        .args(["-m"])
        .arg(model_path)
        .args(["-f"])
        .arg(audio_path)
        .args(["-nt", "-np"])
        .output()
        .map_err(|error| format!("Could not start whisper.cpp: {error}"))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "Transcription failed ({}): {}",
            output.status,
            if detail.is_empty() {
                "whisper.cpp returned no diagnostic output."
            } else {
                &detail
            }
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn find_executable(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}

fn read_stderr(mut stderr: impl Read) -> String {
    let mut diagnostic = String::new();
    if let Err(error) = stderr.read_to_string(&mut diagnostic) {
        return format!("Could not read process error output: {error}");
    }
    diagnostic.trim().to_string()
}

fn remove_recording(path: &PathBuf) {
    if let Err(error) = fs::remove_file(path) {
        if error.kind() != io::ErrorKind::NotFound {
            eprintln!(
                "Could not remove temporary recording {}: {error}",
                path.display()
            );
        }
    }
}
