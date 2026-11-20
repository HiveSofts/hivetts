//! Controllable playback for synthesized speech clips.
//!
//! The TTS page speaks a whole text in one consistent voice; this module
//! drives the finished WAV like a small media player: play/stop, pause and
//! resume, sentence skipping, fast-forward/rewind, and live speed changes.
//! Playback runs in an `ffplay` child process; pause uses SIGSTOP/SIGCONT on
//! unix and a position-preserving restart elsewhere. Seeks and speed changes
//! restart the player at the tracked position with `-ss` and `atempo`, so no
//! audio crates are needed. A monitor thread advances the position clock and
//! reports snapshots to the UI.

use crate::{
    error::{AppError, Result},
    tts::WordTiming,
};
use std::{
    fs,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex, MutexGuard},
    thread,
    time::{Duration, Instant},
};

/// How far the fast-forward / rewind buttons jump, in seconds.
pub const SKIP_SECONDS: f32 = 10.0;
/// Restarting the current sentence is preferred over jumping to the previous
/// one until playback is at least this far into it (common player behavior).
const SENTENCE_RESTART_GRACE: f32 = 1.5;
const TICK: Duration = Duration::from_millis(100);
const SENTENCE_ENDS: [char; 8] = ['.', '!', '?', '۔', '؟', '।', '॥', '…'];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlaybackStatus {
    Idle,
    Playing,
    Paused,
    Finished,
}

impl PlaybackStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Playing => "playing",
            Self::Paused => "paused",
            Self::Finished => "finished",
        }
    }
}

/// One navigable sentence of the clip: where it starts in the audio and the
/// words shown in the sentence preview.
#[derive(Clone, Debug)]
pub struct SentenceSpan {
    pub start: f32,
    pub text: String,
}

/// Everything the UI needs to render the playback deck.
#[derive(Clone, Debug)]
pub struct PlaybackSnapshot {
    pub status: PlaybackStatus,
    pub ready: bool,
    pub position: f32,
    pub duration: f32,
    pub speed: f32,
    pub word_index: i32,
    pub sentence_index: i32,
    pub sentence_count: i32,
    pub sentence_text: String,
    /// Player diagnostic when playback failed (empty otherwise).
    pub error: String,
}

/// A synthesized clip prepared for playback. The controller owns `path` and
/// removes the temporary file when the clip is replaced.
pub struct Clip {
    pub path: PathBuf,
    pub duration: f32,
    pub words: Vec<WordTiming>,
    pub sentences: Vec<SentenceSpan>,
}

struct State {
    generation: u64,
    clip: Option<Clip>,
    status: PlaybackStatus,
    position: f32,
    speed: f32,
    child: Option<Child>,
    last_error: String,
}

struct Inner {
    state: Mutex<State>,
    on_update: Arc<dyn Fn(PlaybackSnapshot) + Send + Sync>,
}

/// Handle shared between the UI callbacks and the monitor thread.
#[derive(Clone)]
pub struct PlaybackController {
    inner: Arc<Inner>,
}

impl PlaybackController {
    pub fn new(on_update: impl Fn(PlaybackSnapshot) + Send + Sync + 'static) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    generation: 0,
                    clip: None,
                    status: PlaybackStatus::Idle,
                    position: 0.0,
                    speed: 1.0,
                    child: None,
                    last_error: String::new(),
                }),
                on_update: Arc::new(on_update),
            }),
        }
    }

    /// Load a freshly synthesized clip and start playing it from the top.
    /// Replaces (and cleans up) any previous clip.
    pub fn start(&self, clip: Clip, speed: f32) -> Result<()> {
        let mut state = self.lock()?;
        state.generation += 1;
        let generation = state.generation;
        kill_child(&mut state);
        if let Some(old) = state.clip.take() {
            remove_clip_file(&old.path);
        }
        state.speed = speed.clamp(0.5, 2.0);
        state.position = 0.0;
        state.last_error.clear();
        state.clip = Some(clip);
        state.status = PlaybackStatus::Idle;
        let result = spawn_player(&mut state);
        if result.is_ok() {
            state.status = PlaybackStatus::Playing;
        }
        drop(state);
        self.spawn_monitor(generation);
        self.poke();
        result
    }

    /// Play/pause toggle. With a finished or stopped clip it replays from 0.
    pub fn toggle(&self) -> Result<()> {
        let action = {
            let state = self.lock()?;
            match state.status {
                PlaybackStatus::Playing => "pause",
                PlaybackStatus::Paused => "resume",
                _ if state.clip.is_some() => "restart",
                _ => "none",
            }
        };
        match action {
            "pause" => self.pause(),
            "resume" => self.resume(),
            "restart" => self.seek(0.0),
            _ => Ok(()),
        }
    }

    pub fn pause(&self) -> Result<()> {
        let mut state = self.lock()?;
        if state.status != PlaybackStatus::Playing {
            return Ok(());
        }
        #[cfg(unix)]
        {
            if let Some(child) = &state.child {
                let _ = Command::new("kill")
                    .arg("-STOP")
                    .arg(child.id().to_string())
                    .status();
            }
        }
        #[cfg(not(unix))]
        {
            // No job control elsewhere: drop the player, keep the position.
            kill_child(&mut state);
        }
        state.status = PlaybackStatus::Paused;
        drop(state);
        self.poke();
        Ok(())
    }

    pub fn resume(&self) -> Result<()> {
        let mut state = self.lock()?;
        if state.status != PlaybackStatus::Paused {
            return Ok(());
        }
        #[cfg(unix)]
        {
            if let Some(child) = &state.child {
                let _ = Command::new("kill")
                    .arg("-CONT")
                    .arg(child.id().to_string())
                    .status();
            } else {
                spawn_player(&mut state)?;
            }
        }
        #[cfg(not(unix))]
        {
            spawn_player(&mut state)?;
        }
        state.status = PlaybackStatus::Playing;
        drop(state);
        self.poke();
        Ok(())
    }

    /// Stop playback and rewind to the top; the clip stays loaded for replay.
    pub fn stop(&self) -> Result<()> {
        let mut state = self.lock()?;
        kill_child(&mut state);
        state.status = PlaybackStatus::Idle;
        state.position = 0.0;
        drop(state);
        self.poke();
        Ok(())
    }

    /// Jump to an absolute position in seconds. Seeking a finished or stopped
    /// clip starts playback there; seeking a paused clip stays paused.
    pub fn seek(&self, position: f32) -> Result<()> {
        let mut state = self.lock()?;
        let Some(clip) = &state.clip else {
            return Ok(());
        };
        let duration = clip.duration;
        let was_paused = state.status == PlaybackStatus::Paused;
        state.position = position.clamp(0.0, (duration - 0.05).max(0.0));
        state.last_error.clear();
        kill_child(&mut state);
        spawn_player(&mut state)?;
        state.status = if was_paused {
            PlaybackStatus::Paused
        } else {
            PlaybackStatus::Playing
        };
        #[cfg(unix)]
        if was_paused {
            if let Some(child) = &state.child {
                let _ = Command::new("kill")
                    .arg("-STOP")
                    .arg(child.id().to_string())
                    .status();
            }
        }
        drop(state);
        self.poke();
        Ok(())
    }

    /// Fast-forward (positive) or rewind (negative) by `delta` seconds.
    pub fn skip(&self, delta: f32) -> Result<()> {
        let position = {
            let state = self.lock()?;
            if state.clip.is_none() {
                return Ok(());
            }
            state.position + delta
        };
        self.seek(position)
    }

    /// Skip to the next (`direction >= 0`) or previous sentence.
    pub fn step_sentence(&self, direction: i32) -> Result<()> {
        let target = {
            let state = self.lock()?;
            let Some(clip) = &state.clip else {
                return Ok(());
            };
            sentence_target(&clip.sentences, state.position, direction)
        };
        self.seek(target)
    }

    /// Change the playback speed live (0.5x … 2x). The player restarts at the
    /// current position with the new `atempo` filter.
    pub fn set_speed(&self, speed: f32) -> Result<()> {
        let speed = speed.clamp(0.5, 2.0);
        let mut state = self.lock()?;
        if (state.speed - speed).abs() < f32::EPSILON {
            return Ok(());
        }
        state.speed = speed;
        if matches!(state.status, PlaybackStatus::Playing | PlaybackStatus::Paused) {
            let was_paused = state.status == PlaybackStatus::Paused;
            kill_child(&mut state);
            spawn_player(&mut state)?;
            state.status = if was_paused {
                PlaybackStatus::Paused
            } else {
                PlaybackStatus::Playing
            };
            #[cfg(unix)]
            if was_paused {
                if let Some(child) = &state.child {
                    let _ = Command::new("kill")
                        .arg("-STOP")
                        .arg(child.id().to_string())
                        .status();
                }
            }
        }
        drop(state);
        self.poke();
        Ok(())
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>> {
        self.inner
            .state
            .lock()
            .map_err(|_| AppError::Message("Playback state is unavailable.".into()))
    }

    /// Push the current state to the UI immediately after a command.
    fn poke(&self) {
        let snapshot = {
            let Ok(state) = self.inner.state.lock() else {
                return;
            };
            snapshot_of(&state)
        };
        (self.inner.on_update)(snapshot);
    }

    /// One monitor per loaded clip; the generation counter retires old ones.
    fn spawn_monitor(&self, generation: u64) {
        let inner = Arc::clone(&self.inner);
        thread::spawn(move || {
            let mut last = Instant::now();
            loop {
                thread::sleep(TICK);
                let snapshot = {
                    let Ok(mut state) = inner.state.lock() else {
                        return;
                    };
                    if state.generation != generation {
                        return;
                    }
                    // Keep the clock fresh on every tick, even while paused,
                    // so resuming never adds the paused span to the position.
                    let now = Instant::now();
                    let elapsed = now.duration_since(last).as_secs_f32();
                    last = now;
                    // Commands poke the UI themselves; while idle, paused, or
                    // finished nothing moves, so the monitor stays quiet.
                    if state.status != PlaybackStatus::Playing {
                        continue;
                    }
                    if let Some(clip) = &state.clip {
                        state.position =
                            (state.position + elapsed * state.speed).min(clip.duration);
                    }
                    let mut finished = false;
                    let mut failure = String::new();
                    match state.child.as_mut() {
                        Some(child) => match child.try_wait() {
                            Ok(Some(status)) => {
                                finished = true;
                                if !status.success() {
                                    if let Some(stderr) = child.stderr.take() {
                                        failure = crate::audio::read_stderr(stderr);
                                    }
                                }
                            }
                            Ok(None) => {}
                            Err(_) => finished = true,
                        },
                        None => finished = true,
                    }
                    if finished {
                        state.status = PlaybackStatus::Finished;
                        state.child = None;
                        state.last_error = failure;
                        if let Some(clip) = &state.clip {
                            state.position = clip.duration;
                        }
                    }
                    snapshot_of(&state)
                };
                (inner.on_update)(snapshot);
            }
        });
    }
}

fn snapshot_of(state: &State) -> PlaybackSnapshot {
    let (duration, word_index, sentence_index, sentence_count, sentence_text) = match &state.clip {
        Some(clip) => {
            let word_index = word_index_at(&clip.words, state.position);
            if clip.sentences.is_empty() {
                (clip.duration, word_index, -1, 0, String::new())
            } else {
                let index = sentence_index_at(&clip.sentences, state.position);
                (
                    clip.duration,
                    word_index,
                    index as i32,
                    clip.sentences.len() as i32,
                    clip.sentences[index].text.clone(),
                )
            }
        }
        None => (0.0, -1, -1, 0, String::new()),
    };
    PlaybackSnapshot {
        status: state.status,
        ready: state.clip.is_some(),
        position: state.position,
        duration,
        speed: state.speed,
        word_index,
        sentence_index,
        sentence_count,
        sentence_text,
        error: state.last_error.clone(),
    }
}

/// Start `ffplay` at the tracked position with the current tempo. `-ss` before
/// `-i` seeks the WAV sample-accurately; `atempo` stretches without pitch.
fn spawn_player(state: &mut State) -> Result<()> {
    let Some(clip) = &state.clip else {
        return Ok(());
    };
    let position = state.position.clamp(0.0, (clip.duration - 0.05).max(0.0));
    let child = Command::new("ffplay")
        .args([
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-ss",
            &format!("{position:.3}"),
            "-af",
            &format!("atempo={:.3}", state.speed.clamp(0.5, 2.0)),
            "-i",
        ])
        .arg(&clip.path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            AppError::Message(format!(
                "Could not start ffplay: {error}. Install FFmpeg to use the playback controls."
            ))
        })?;
    state.child = Some(child);
    Ok(())
}

fn kill_child(state: &mut State) {
    if let Some(mut child) = state.child.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn remove_clip_file(path: &PathBuf) {
    if let Err(error) = fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            eprintln!("Could not remove temporary playback file: {error}");
        }
    }
}

/// Group the word timeline into sentences: a word ending in a sentence
/// delimiter closes the current span. Text without delimiters stays one span.
pub fn sentence_spans(words: &[WordTiming]) -> Vec<SentenceSpan> {
    let mut spans: Vec<SentenceSpan> = Vec::new();
    let mut start: Option<f32> = None;
    let mut text = String::new();
    for timing in words {
        if start.is_none() {
            start = Some(timing.start);
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&timing.word);
        let ends_sentence = timing
            .word
            .chars()
            .last()
            .is_some_and(|character| SENTENCE_ENDS.contains(&character));
        if ends_sentence {
            spans.push(SentenceSpan {
                start: start.unwrap_or(0.0),
                text: std::mem::take(&mut text),
            });
            start = None;
        }
    }
    if !text.is_empty() {
        spans.push(SentenceSpan {
            start: start.unwrap_or(0.0),
            text,
        });
    }
    spans
}

/// Index of the sentence containing `position` (the last one that started).
pub fn sentence_index_at(sentences: &[SentenceSpan], position: f32) -> usize {
    if sentences.is_empty() {
        return 0;
    }
    sentences
        .partition_point(|span| span.start <= position + 0.001)
        .saturating_sub(1)
        .min(sentences.len() - 1)
}

/// Target position for a sentence skip. Forward jumps to the next sentence
/// start (the last sentence restarts); backward restarts the current sentence
/// when more than the grace period into it, otherwise jumps to the previous.
pub fn sentence_target(sentences: &[SentenceSpan], position: f32, direction: i32) -> f32 {
    if sentences.is_empty() {
        return 0.0;
    }
    let current = sentence_index_at(sentences, position);
    if direction >= 0 {
        let next = (current + 1).min(sentences.len() - 1);
        sentences[next].start
    } else {
        let start = sentences[current].start;
        if position - start > SENTENCE_RESTART_GRACE || current == 0 {
            start
        } else {
            sentences[current - 1].start
        }
    }
}

/// Index of the word being spoken at `position`, or -1 before the first word.
pub fn word_index_at(words: &[WordTiming], position: f32) -> i32 {
    words.partition_point(|timing| timing.start <= position + 0.001) as i32 - 1
}

/// Compact `m:ss` clock for the playback deck.
pub fn format_clock(seconds: f32) -> String {
    let total = seconds.max(0.0).floor() as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timeline() -> Vec<WordTiming> {
        // "Hello there. This is two. And three!" spread over 9 seconds.
        let words = ["Hello", "there.", "This", "is", "two.", "And", "three!"];
        words
            .iter()
            .enumerate()
            .map(|(index, word)| WordTiming {
                word: word.to_string(),
                start: index as f32 * 1.3,
            })
            .collect()
    }

    #[test]
    fn sentence_spans_group_words_by_delimiters() {
        let spans = sentence_spans(&timeline());
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].text, "Hello there.");
        assert_eq!(spans[0].start, 0.0);
        assert_eq!(spans[1].text, "This is two.");
        assert!((spans[1].start - 2.6).abs() < 0.001);
        assert_eq!(spans[2].text, "And three!");
        // No delimiter: the whole text is one navigable sentence.
        let single = sentence_spans(&[WordTiming {
            word: "no punctuation here".into(),
            start: 0.0,
        }]);
        assert_eq!(single.len(), 1);
        assert!(sentence_spans(&[]).is_empty());
    }

    #[test]
    fn sentence_navigation_steps_forward_and_back() {
        let spans = sentence_spans(&timeline());
        // From inside sentence 0, forward lands on sentence 1's start.
        assert!((sentence_target(&spans, 0.5, 1) - spans[1].start).abs() < 0.001);
        // Early in sentence 1, backward returns to sentence 0.
        assert!((sentence_target(&spans, spans[1].start + 0.2, -1) - spans[0].start).abs() < 0.001);
        // Deep into sentence 1, backward restarts sentence 1 itself.
        assert!((sentence_target(&spans, spans[1].start + 2.0, -1) - spans[1].start).abs() < 0.001);
        // Backward at the very first sentence stays at zero.
        assert_eq!(sentence_target(&spans, 0.2, -1), 0.0);
        // Forward past the last sentence restarts the last one.
        let last = spans.len() - 1;
        assert!((sentence_target(&spans, spans[last].start + 0.5, 1) - spans[last].start).abs() < 0.001);
        assert_eq!(sentence_target(&[], 3.0, 1), 0.0);
    }

    #[test]
    fn word_index_tracks_position() {
        let words = timeline();
        assert_eq!(word_index_at(&words, 0.0), 0);
        assert_eq!(word_index_at(&words, 2.7), 2);
        assert_eq!(word_index_at(&words, 100.0), words.len() as i32 - 1);
        assert_eq!(word_index_at(&[], 1.0), -1);
    }

    #[test]
    fn clock_formats_minutes_and_seconds() {
        assert_eq!(format_clock(0.0), "0:00");
        assert_eq!(format_clock(9.6), "0:09");
        assert_eq!(format_clock(75.0), "1:15");
        assert_eq!(format_clock(-3.0), "0:00");
    }

    #[test]
    #[ignore = "needs espeak-ng, ffplay, and a working audio device"]
    fn playback_lifecycle_with_real_audio() {
        use crate::config::AppConfig;
        use std::sync::mpsc;

        let config = AppConfig {
            tts_language: "en".into(),
            auto_save_directory: String::new(),
            ..AppConfig::default()
        };
        let path = std::env::temp_dir().join(format!(
            "hivetts-playback-test-{}.wav",
            std::process::id()
        ));
        crate::tts::synthesize_to_file(
            "Hello there. This is a second sentence. And a third one.",
            &path,
            &config,
        )
        .expect("synthesis");
        let duration = crate::tts::wav_duration_seconds(&path).expect("duration");
        let words = crate::preprocess::segment_words(
            "Hello there. This is a second sentence. And a third one.",
        );
        let timeline = crate::tts::word_timeline(&words, duration);
        let sentences = sentence_spans(&timeline);
        let (sender, receiver) = mpsc::channel();
        let controller = PlaybackController::new(move |snapshot| {
            let _ = sender.send(snapshot);
        });
        controller
            .start(
                Clip {
                    path: path.clone(),
                    duration,
                    words: timeline,
                    sentences,
                },
                1.0,
            )
            .expect("start");
        thread::sleep(Duration::from_millis(600));
        let snapshot = receiver.try_iter().last().expect("snapshot");
        assert_eq!(snapshot.status, PlaybackStatus::Playing);
        assert!(snapshot.position > 0.0);
        controller.pause().expect("pause");
        let paused = receiver.try_iter().last().expect("paused snapshot");
        assert_eq!(paused.status, PlaybackStatus::Paused);
        let paused_at = paused.position;
        thread::sleep(Duration::from_millis(300));
        // The position clock must stay frozen while paused: no new snapshots.
        assert!(
            receiver.try_iter().next().is_none(),
            "position advanced while paused (was {paused_at})"
        );
        controller.resume().expect("resume");
        controller.skip(SKIP_SECONDS).expect("skip");
        controller.step_sentence(-1).expect("sentence");
        controller.set_speed(1.5).expect("speed");
        assert_eq!(receiver.try_iter().last().unwrap().speed, 1.5);
        controller.stop().expect("stop");
        let stopped = receiver.try_iter().last().unwrap();
        assert_eq!(stopped.status, PlaybackStatus::Idle);
        assert_eq!(stopped.position, 0.0);
        let _ = fs::remove_file(&path);
    }
}
