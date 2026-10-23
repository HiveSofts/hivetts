use std::{
    env, fs,
    io::{self, Read},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub type ActiveRecording = Arc<Mutex<Option<(Child, PathBuf)>>>;

pub fn new_recording_state() -> ActiveRecording {
    Arc::new(Mutex::new(None))
}

pub fn recording_path() -> PathBuf {
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
pub fn start_recording(path: &PathBuf) -> io::Result<Child> {
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
pub fn start_recording(path: &PathBuf) -> io::Result<Child> {
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
pub fn start_recording(_path: &PathBuf) -> io::Result<Child> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Microphone recording is currently supported on Linux and macOS",
    ))
}

pub fn monitor_recording<F>(recording: ActiveRecording, on_failure: F)
where
    F: FnOnce(String) + Send + 'static,
{
    thread::spawn(move || {
        let failure = loop {
            thread::sleep(Duration::from_millis(500));
            let mut active = match recording.lock() {
                Ok(active) => active,
                Err(error) => break format!("Recording state is unavailable: {error}"),
            };
            let Some((child, path)) = active.as_mut() else {
                return;
            };
            match child.try_wait() {
                Ok(None) => continue,
                Ok(Some(status)) => {
                    let diagnostic = child
                        .stderr
                        .take()
                        .map(read_stderr)
                        .unwrap_or_else(|| "No diagnostic output available.".into());
                    let path = path.clone();
                    *active = None;
                    remove_recording(&path);
                    break format!("Recording stopped unexpectedly ({status}): {diagnostic}");
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
                    break format!("Could not check recording status: {error}");
                }
            }
        };
        on_failure(failure);
    });
}

pub fn read_stderr(mut stderr: impl Read) -> String {
    let mut diagnostic = String::new();
    if let Err(error) = stderr.read_to_string(&mut diagnostic) {
        return format!("Could not read process error output: {error}");
    }
    diagnostic.trim().to_string()
}

pub fn remove_recording(path: &PathBuf) {
    if let Err(error) = fs::remove_file(path) {
        if error.kind() != io::ErrorKind::NotFound {
            eprintln!(
                "Could not remove temporary recording {}: {error}",
                path.display()
            );
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn find_executable(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}
