use crate::{
    config::AppConfig,
    error::{AppError, Result},
};
use std::{
    env,
    fmt,
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Emotional tone applied to synthesized speech. Engines that lack native
/// emotion control approximate it through prosody (espeak-ng) or punctuation
/// styling (Piper). The TTS page's Voice Style selector maps onto these
/// variants (standard → Neutral, calm → Calm, formal → Formal,
/// friendly → Friendly); legacy values such as `happy` still parse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Emotion {
    Neutral,
    Happy,
    Sad,
    Formal,
    Friendly,
    Excited,
    Calm,
}

impl Emotion {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "happy" => Self::Happy,
            "sad" => Self::Sad,
            "formal" => Self::Formal,
            "friendly" => Self::Friendly,
            "excited" => Self::Excited,
            "calm" => Self::Calm,
            // The Voice Style default is `standard`, which behaves neutrally.
            _ => Self::Neutral,
        }
    }
}

/// Speech Tone layered on top of the Voice Style. Chosen from the TTS page
/// (neutral, gentle, informative, conversational) and applied as small prosody
/// shifts so the same engine can produce different delivery flavors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechTone {
    Neutral,
    Gentle,
    Informative,
    Conversational,
}

impl SpeechTone {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "gentle" => Self::Gentle,
            "informative" => Self::Informative,
            "conversational" => Self::Conversational,
            _ => Self::Neutral,
        }
    }

    /// Additional pitch offset (espeak units) contributed by the tone.
    pub fn pitch_shift(self) -> f32 {
        match self {
            Self::Neutral => 0.0,
            Self::Gentle => -6.0,
            Self::Informative => -3.0,
            Self::Conversational => 5.0,
        }
    }

    /// Additional words-per-minute contributed by the tone.
    pub fn rate_shift(self) -> f32 {
        match self {
            Self::Neutral => 0.0,
            Self::Gentle => -10.0,
            Self::Informative => -5.0,
            Self::Conversational => 12.0,
        }
    }
}

impl fmt::Display for Emotion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Neutral => "neutral",
            Self::Happy => "happy",
            Self::Sad => "sad",
            Self::Formal => "formal",
            Self::Friendly => "friendly",
            Self::Excited => "excited",
            Self::Calm => "calm",
        };
        formatter.write_str(label)
    }
}

const SENTENCE_DELIMITERS: [char; 7] = ['.', '!', '?', '۔', '؟', '।', '॥'];
const CLAUSE_DELIMITERS: [char; 5] = [',', ';', '،', '؛', ':'];

pub trait SpeechSynthesizer {
    /// Speak the text; the `Ok` payload is the auto-saved download file, if any.
    fn speak(&self, text: &str, config: &AppConfig) -> Result<Option<PathBuf>>;
    fn export_wav(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()>;
    fn export_mp3(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()>;
    fn export_ogg(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()>;
}

pub struct LocalSynthesizer;

/// Container formats the TTS page can export and batch-convert into.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExportFormat {
    Wav,
    Mp3,
    Ogg,
}

impl ExportFormat {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "mp3" => Self::Mp3,
            "ogg" => Self::Ogg,
            _ => Self::Wav,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Ogg => "ogg",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Wav => "WAV",
            Self::Mp3 => "MP3",
            Self::Ogg => "OGG",
        }
    }
}

/// Output quality presets. MP3 picks a bitrate, OGG a Vorbis quality level,
/// and WAV the sample rate the clip is delivered at (16-bit PCM throughout).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputQuality {
    Low,
    Standard,
    High,
    Maximum,
}

impl OutputQuality {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "low" => Self::Low,
            "standard" => Self::Standard,
            "maximum" => Self::Maximum,
            _ => Self::High,
        }
    }

    pub fn mp3_bitrate(self) -> &'static str {
        match self {
            Self::Low => "96k",
            Self::Standard => "160k",
            Self::High => "256k",
            Self::Maximum => "320k",
        }
    }

    pub fn ogg_quality(self) -> &'static str {
        match self {
            Self::Low => "3",
            Self::Standard => "5",
            Self::High => "7",
            Self::Maximum => "10",
        }
    }

    pub fn wav_sample_rate(self) -> u32 {
        match self {
            Self::Low => 16000,
            Self::Standard => 22050,
            Self::High => 44100,
            Self::Maximum => 48000,
        }
    }
}

/// One entry of the read-along timeline: a display word and the second, counted
/// from playback start, at which it is expected to begin. Timings are derived
/// from the finished clip's duration and each word's relative length, so they
/// approximate the true audio rather than tracking phoneme boundaries.
#[derive(Clone, Debug)]
pub struct WordTiming {
    pub word: String,
    pub start: f32,
}

fn piper_configured(config: &AppConfig) -> bool {
    !config.piper_executable.is_empty() && !config.piper_voice.is_empty()
}

/// A downloaded model voice requires Piper; selecting one without a configured
/// executable must fail loudly instead of silently falling back to eSpeak NG.
fn use_piper(config: &AppConfig) -> Result<bool> {
    if config.piper_voice.trim().is_empty() {
        return Ok(false);
    }
    if config.piper_executable.trim().is_empty() {
        return Err(AppError::Message(
            "Model voices need a Piper executable. Configure it under Speech to Text settings."
                .into(),
        ));
    }
    // Downloaded models are listed even when their weights are not Piper
    // format; report that up front instead of failing inside piper.
    if !config.piper_voice.to_lowercase().ends_with(".onnx") {
        return Err(AppError::Message(format!(
            "The selected voice model is not a Piper (.onnx) voice, so it cannot be spoken: {}",
            config.piper_voice
        )));
    }
    Ok(piper_configured(config))
}

impl SpeechSynthesizer for LocalSynthesizer {
    fn speak(&self, text: &str, config: &AppConfig) -> Result<Option<PathBuf>> {
        validate_text(text)?;
        if use_piper(config)? {
            let temp_path = unique_path(&temp_wav_directory(), "hivetts-speak", "wav");
            piper_synthesize(text, &temp_path, config)?;
            let result = play_wav(&temp_path);
            let spoken_file = move_auto_save(&temp_path, config);
            if let Err(error) = fs::remove_file(&temp_path) {
                eprintln!("Could not remove temporary speech file: {error}");
            }
            return result.map(|()| spoken_file);
        }
        native_speak(text, config)
    }

    fn export_wav(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()> {
        validate_text(text)?;
        if destination.to_string_lossy().trim().is_empty() {
            return Err(AppError::Message("Enter a WAV export path.".into()));
        }
        let quality = OutputQuality::parse(&config.output_quality);
        let temp_wav = unique_path(&temp_wav_directory(), "hivetts-export", "wav");
        synthesize_to_file(text, &temp_wav, config)?;
        let result = write_wav_with_quality(&temp_wav, destination, quality);
        if let Err(error) = fs::remove_file(&temp_wav) {
            eprintln!("Could not remove temporary WAV file: {error}");
        }
        result.and_then(|()| save_copy(destination, config))
    }

    fn export_mp3(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()> {
        validate_text(text)?;
        if destination.to_string_lossy().trim().is_empty() {
            return Err(AppError::Message("Enter an MP3 export path.".into()));
        }
        export_compressed(text, destination, config, ExportFormat::Mp3)
    }

    fn export_ogg(
        &self,
        text: &str,
        destination: &std::path::Path,
        config: &AppConfig,
    ) -> Result<()> {
        validate_text(text)?;
        if destination.to_string_lossy().trim().is_empty() {
            return Err(AppError::Message("Enter an OGG export path.".into()));
        }
        export_compressed(text, destination, config, ExportFormat::Ogg)
    }
}

/// Shared path for the lossy formats: synthesize once, then encode with the
/// quality preset's bitrate (MP3) or Vorbis quality level (OGG).
fn export_compressed(
    text: &str,
    destination: &std::path::Path,
    config: &AppConfig,
    format: ExportFormat,
) -> Result<()> {
    let temp_wav = unique_path(&temp_wav_directory(), "hivetts-export", "wav");
    synthesize_to_file(text, &temp_wav, config)?;
    let result = transcode(
        &temp_wav,
        destination,
        format,
        OutputQuality::parse(&config.output_quality),
    );
    if let Err(error) = fs::remove_file(&temp_wav) {
        eprintln!("Could not remove temporary WAV file: {error}");
    }
    result.and_then(|()| save_copy(destination, config))
}

/// Deliver the WAV at the quality preset's sample rate: copy untouched when
/// the engine already produced that rate, otherwise resample with FFmpeg.
fn write_wav_with_quality(
    source: &Path,
    destination: &Path,
    quality: OutputQuality,
) -> Result<()> {
    ensure_parent_dir(destination)?;
    let native_rate = fs::read(source)
        .ok()
        .and_then(|bytes| read_wav_sample_rate(&bytes));
    match native_rate {
        Some(rate) if rate != quality.wav_sample_rate() => {
            resample_wav(source, destination, quality.wav_sample_rate())
        }
        _ => fs::copy(source, destination).map(|_| ()).map_err(AppError::from),
    }
}

impl LocalSynthesizer {
    /// Synthesize the text to a temporary WAV and measure its duration, ready
    /// for the playback controller. The caller owns the returned file.
    pub fn prepare_playback(text: &str, config: &AppConfig) -> Result<(PathBuf, f32)> {
        validate_text(text)?;
        let path = unique_path(&temp_wav_directory(), "hivetts-playback", "wav");
        synthesize_to_file(text, &path, config)?;
        match wav_duration_seconds(&path) {
            Ok(duration) => Ok((path, duration)),
            Err(error) => {
                let _ = fs::remove_file(&path);
                Err(error)
            }
        }
    }

    /// Synthesize the whole text in a single pass (so the voice stays
    /// consistent), then play it back while driving read-along highlighting.
    /// `words` is the display list (built with `preprocess::segment_words`);
    /// the clip duration is measured after synthesis, so `on_word(index)`
    /// fires at each word's estimated start. Falls back to the plain blocking
    /// `speak` when there is nothing to highlight or playback is unsupported.
    pub fn speak_highlighted(
        &self,
        text: &str,
        config: &AppConfig,
        words: &[String],
        mut on_word: impl FnMut(usize) + Send,
    ) -> Result<Option<PathBuf>> {
        validate_text(text)?;
        if words.is_empty() {
            return self.speak(text, config);
        }
        let temp_path = unique_path(&temp_wav_directory(), "hivetts-speak", "wav");
        synthesize_to_file(text, &temp_path, config)?;
        let timeline = word_timeline(words, wav_duration_seconds(&temp_path).unwrap_or(0.0));
        if timeline.is_empty() {
            // Duration could not be measured; play without live highlighting.
            let result = play_wav(&temp_path);
            let spoken_file = move_auto_save(&temp_path, config);
            let _ = fs::remove_file(&temp_path);
            return result.map(|()| spoken_file);
        }
        let mut child = play_wav_child(&temp_path)?;
        let start = Instant::now();
        for (index, timing) in timeline.iter().enumerate() {
            let target = timing.start.max(0.0);
            let elapsed = start.elapsed().as_secs_f32();
            if target > elapsed {
                thread::sleep(Duration::from_secs_f32(target - elapsed));
            }
            on_word(index);
        }
        let status = child.wait()?;
        let spoken_file = move_auto_save(&temp_path, config);
        if let Err(error) = fs::remove_file(&temp_path) {
            eprintln!("Could not remove temporary speech file: {error}");
        }
        wait_message(status, "Audio player")?;
        Ok(spoken_file)
    }
}

/// Read the duration of a PCM WAV file in seconds from its RIFF header.
pub fn wav_duration_seconds(path: &Path) -> Result<f32> {
    let bytes = fs::read(path)?;
    read_wav_duration(&bytes)
        .ok_or_else(|| AppError::Message("Could not determine the audio duration.".into()))
}

/// Locate the `fmt ` and `data` chunks and divide the data size by the byte
/// rate. Returns `None` for anything that is not a parsable PCM WAV.
fn read_wav_duration(bytes: &[u8]) -> Option<f32> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut position = 12usize;
    let mut byte_rate = 0u32;
    let mut data_size = 0usize;
    while position + 8 <= bytes.len() {
        let id = bytes.get(position..position + 4)?;
        let size = u32::from_le_bytes(bytes.get(position + 4..position + 8)?.try_into().ok()?)
            as usize;
        if id == b"fmt " {
            // Byte rate lives at offset 8 inside the fmt chunk (after the
            // 2-byte audio format and 2-byte channel count and 4-byte rate).
            let offset = position + 8 + 8;
            byte_rate = u32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?);
        } else if id == b"data" {
            data_size = size;
            break;
        }
        // Chunks are word-aligned, so an odd size is followed by a pad byte.
        position += 8 + size + (size % 2);
    }
    if byte_rate == 0 {
        return None;
    }
    Some(data_size as f32 / byte_rate as f32)
}

/// Distribute a clip's duration across its words proportionally to each word's
/// character length, adding a bonus for words that end a clause or sentence so
/// the highlight lags a little during the natural pauses the engine inserts.
pub fn word_timeline(words: &[String], duration: f32) -> Vec<WordTiming> {
    if words.is_empty() || !(duration > 0.0) {
        return Vec::new();
    }
    const SENTENCE_ENDS: [char; 7] = ['.', '!', '?', '۔', '؟', '।', '॥'];
    const CLAUSE_ENDS: [char; 5] = [',', ';', ':', '،', '؛'];
    let weights: Vec<f32> = words
        .iter()
        .map(|word| {
            let mut weight = word.chars().count().max(1) as f32;
            if let Some(last) = word.chars().last() {
                if SENTENCE_ENDS.contains(&last) {
                    weight += 6.0;
                } else if CLAUSE_ENDS.contains(&last) {
                    weight += 3.0;
                }
            }
            weight
        })
        .collect();
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let mut timings = Vec::with_capacity(words.len());
    let mut consumed = 0.0f32;
    let mut previous = 0.0f32;
    for (word, weight) in words.iter().zip(&weights) {
        let start = (duration * (consumed / total)).max(previous);
        previous = start;
        timings.push(WordTiming {
            word: word.clone(),
            start,
        });
        consumed += weight;
    }
    timings
}

/// Start playback without waiting, so the highlight loop can advance while the
/// audio plays. Returns the running player process to be waited on later.
fn play_wav_child(path: &Path) -> Result<Child> {
    #[cfg(target_os = "macos")]
    {
        Ok(Command::new("afplay").arg(path).spawn()?)
    }
    #[cfg(target_os = "linux")]
    {
        Ok(Command::new("ffplay")
            .args(["-nodisp", "-autoexit", "-loglevel", "error"])
            .arg(path)
            .spawn()?)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = path;
        Err(AppError::Message(
            "Read-along highlighting is supported on Linux and macOS.".into(),
        ))
    }
}

/// Copy an exported file into the auto-save directory so a download copy
/// always exists, unless it is already there.
fn save_copy(exported: &Path, config: &AppConfig) -> Result<()> {
    let Some(directory) = auto_save_directory(config) else {
        return Ok(());
    };
    if exported.parent().is_some_and(|parent| parent == directory) {
        return Ok(());
    }
    let extension = exported
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("wav");
    let destination = unique_path(&directory, "hivetts-speech", extension);
    if let Some(parent) = destination.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::copy(exported, &destination) {
        Ok(_) => {
            eprintln!("Download copy saved to {}", destination.display());
            Ok(())
        }
        // The copy is a convenience; never fail the export over it.
        Err(error) => {
            eprintln!("Could not save the download copy: {error}");
            Ok(())
        }
    }
}

pub fn synthesize_to_file(
    text: &str,
    destination: &std::path::Path,
    config: &AppConfig,
) -> Result<()> {
    if use_piper(config)? {
        return piper_synthesize(text, destination, config);
    }
    #[cfg(target_os = "linux")]
    {
        let mut command = espeak_command(text, Some(destination), config)?;
        let status = command.status()?;
        return wait_message(status, "Speech engine");
    }
    #[cfg(target_os = "macos")]
    {
        let mut command = Command::new("say");
        command
            .arg("-r")
            .arg(words_per_minute(config))
            .arg("-o")
            .arg(destination)
            .arg("--file-format=WAVE")
            .arg("--data-format=LEI16@22050")
            .arg(plain_text(text, config));
        let status = command.status()?;
        return wait_message(status, "Speech engine");
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    Err(AppError::Message(
        "Audio file generation requires Piper; configure it in the STT settings.".into(),
    ))
}

/// Generate a timestamped file path inside the configured auto-save directory.
pub fn auto_save_path(config: &AppConfig, extension: &str) -> Option<PathBuf> {
    auto_save_directory(config).map(|directory| unique_path(&directory, "hivetts-speech", extension))
}

fn auto_save_directory(config: &AppConfig) -> Option<PathBuf> {
    let directory = config.auto_save_directory.trim();
    (!directory.is_empty()).then(|| PathBuf::from(directory))
}

fn move_auto_save(temp_path: &Path, config: &AppConfig) -> Option<PathBuf> {
    let destination = auto_save_path(config, "wav")?;
    if let Some(parent) = destination.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::rename(temp_path, &destination)
        .or_else(|_| fs::copy(temp_path, &destination).map(|_| ()))
    {
        Ok(()) => Some(destination),
        Err(error) => {
            eprintln!("Could not save the speech file: {error}");
            None
        }
    }
}

fn unique_path(directory: &Path, stem: &str, extension: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut candidate =
        directory.join(format!("{stem}-{}-{timestamp}.{extension}", std::process::id()));
    let mut counter = 1u32;
    while candidate.exists() {
        candidate =
            directory.join(format!("{stem}-{}-{timestamp}-{counter}.{extension}", std::process::id()));
        counter += 1;
    }
    candidate
}

fn temp_wav_directory() -> PathBuf {
    env::temp_dir()
}

/// Encode a synthesized WAV into the requested container with FFmpeg, using
/// the quality preset's sample rate and encoder settings.
fn transcode(
    source: &Path,
    destination: &Path,
    format: ExportFormat,
    quality: OutputQuality,
) -> Result<()> {
    ensure_parent_dir(destination)?;
    let ffmpeg = env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let mut command = Command::new(ffmpeg);
    command
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(source)
        // Encode at the preset's rate: MPEG-2 Layer III (<= 24 kHz) caps at
        // 160 kbps, so the High/Maximum bitrates need MPEG-1 sample rates.
        .args(["-ar", &quality.wav_sample_rate().to_string()]);
    match format {
        ExportFormat::Mp3 => {
            command.args(["-codec:a", "libmp3lame", "-b:a", quality.mp3_bitrate()]);
        }
        ExportFormat::Ogg => {
            command.args(["-codec:a", "libvorbis", "-q:a", quality.ogg_quality()]);
        }
        ExportFormat::Wav => {
            command.args(["-codec:a", "pcm_s16le"]);
        }
    }
    let status = command.arg(destination).status().map_err(|error| {
        AppError::Message(format!(
            "Could not run FFmpeg to encode {}: {error}. Install FFmpeg for MP3/OGG export.",
            format.label()
        ))
    })?;
    wait_message(status, &format!("{} encoder", format.label()))
}

/// Resample a WAV to the quality preset's rate, keeping 16-bit PCM.
fn resample_wav(source: &Path, destination: &Path, sample_rate: u32) -> Result<()> {
    let ffmpeg = env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let status = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(source)
        .args(["-ar", &sample_rate.to_string(), "-codec:a", "pcm_s16le"])
        .arg(destination)
        .status()
        .map_err(|error| {
            AppError::Message(format!(
                "Could not run FFmpeg to resample the WAV: {error}. Install FFmpeg to change WAV quality."
            ))
        })?;
    wait_message(status, "WAV resampler")
}

fn ensure_parent_dir(destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Sample rate from the `fmt ` chunk, for deciding whether a WAV export needs
/// resampling to reach the chosen quality.
fn read_wav_sample_rate(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut position = 12usize;
    while position + 8 <= bytes.len() {
        let id = bytes.get(position..position + 4)?;
        let size = u32::from_le_bytes(bytes.get(position + 4..position + 8)?.try_into().ok()?)
            as usize;
        if id == b"fmt " {
            // Sample rate sits 4 bytes into the fmt chunk body.
            let offset = position + 8 + 4;
            return Some(u32::from_le_bytes(
                bytes.get(offset..offset + 4)?.try_into().ok()?,
            ));
        }
        // Chunks are word-aligned, so an odd size is followed by a pad byte.
        position += 8 + size + (size % 2);
    }
    None
}

fn piper_synthesize(text: &str, destination: &std::path::Path, config: &AppConfig) -> Result<()> {
    let styled = piper_text(text, config);
    let mut command = Command::new(&config.piper_executable);
    command
        .args(["--model", &config.piper_voice, "--output_file"])
        .arg(destination)
        .args(["--length_scale"])
        .arg((1.0 / config.speech_rate.clamp(0.5, 2.0)).to_string());
    // Speaker ids address voices bundled inside multi-speaker Piper models.
    if let Some(index) = speaker_id_index(&config.voice) {
        command.arg("--speaker").arg(index);
    }
    if (config.volume - 1.0).abs() > f32::EPSILON {
        command
            .args(["--volume"])
            .arg(config.volume.clamp(0.2, 3.0).to_string());
    }
    let mut child = command.stdin(Stdio::piped()).spawn()?;
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| AppError::Message("Piper input stream unavailable.".into()))?;
    use std::io::Write;
    input.write_all(styled.as_bytes())?;
    drop(input);
    let status = child.wait()?;
    wait_message(status, "Piper")
}

fn speaker_id_index(voice: &str) -> Option<String> {
    let digits: String = voice
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect();
    (!digits.is_empty()).then_some(digits)
}

/// Piper has no prosody control, so voice style is approximated by restyling
/// punctuation, which the phonemizer uses to shape intonation.
fn piper_text(text: &str, config: &AppConfig) -> String {
    let mut styled = with_pause_styling(text, config.pause_sentence, config.pause_paragraph);
    match Emotion::parse(&config.voice_style) {
        Emotion::Neutral => {}
        Emotion::Happy => styled = push_upbeat(&styled, "!"),
        Emotion::Excited => styled = push_upbeat(&styled, "!!"),
        Emotion::Sad => styled = push_suffix(&styled, "…"),
        Emotion::Calm => styled = push_suffix(&styled, "."),
        Emotion::Friendly => styled = push_suffix(&styled, "!"),
        Emotion::Formal => styled = ensure_terminal(&styled, "."),
    }
    styled
}

fn push_upbeat(text: &str, marker: &str) -> String {
    let trimmed = text.trim_end();
    let base = trim_terminal_markers(trimmed);
    format!("{base}{marker}")
}

fn push_suffix(text: &str, marker: &str) -> String {
    let trimmed = text.trim_end();
    if trimmed.ends_with(marker) {
        trimmed.to_string()
    } else {
        format!("{trimmed}{marker}")
    }
}

fn ensure_terminal(text: &str, marker: &str) -> String {
    let trimmed = text.trim_end();
    if trimmed
        .chars()
        .last()
        .is_some_and(|character| SENTENCE_DELIMITERS.contains(&character))
    {
        trimmed.to_string()
    } else {
        format!("{trimmed}{marker}")
    }
}

fn trim_terminal_markers(text: &str) -> String {
    let mut result = text.to_string();
    while result
        .chars()
        .last()
        .is_some_and(|character| SENTENCE_DELIMITERS.contains(&character) || character == '…')
    {
        result.pop();
    }
    result
}

/// Scale inter-sentence, inter-clause, and inter-paragraph pauses by repeating
/// espeak pause characters. Each additional second beyond the natural pause
/// (~0.5 s) inserts one more delimiter, and paragraph breaks (`\n`) receive
/// their own stretch through `pause_paragraph`.
fn with_pause_styling(text: &str, pause_sentence: f32, pause_paragraph: f32) -> String {
    let sentence_input = if pause_sentence.is_finite() { pause_sentence } else { 0.5 };
    let paragraph_input = if pause_paragraph.is_finite() { pause_paragraph } else { 0.5 };
    let extra_sentences = (sentence_input - 0.5).max(0.0).clamp(0.0, 4.0).round() as usize;
    let extra_clauses = ((sentence_input - 0.5) / 2.0).max(0.0).clamp(0.0, 2.0).round() as usize;
    let extra_paragraphs = (paragraph_input - 0.5).max(0.0).clamp(0.0, 5.0).round() as usize;
    if extra_sentences == 0 && extra_clauses == 0 && extra_paragraphs == 0 {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut sentence: Vec<char> = Vec::new();
    for character in text.chars() {
        if character == '\n' {
            // Flush the pending sentence and stretch the paragraph gap.
            while sentence
                .last()
                .is_some_and(|previous| SENTENCE_DELIMITERS.contains(previous))
            {
                sentence.pop();
            }
            result.extend(sentence.iter());
            sentence.clear();
            for _ in 0..extra_paragraphs {
                result.push('.');
            }
            result.push('\n');
        } else if SENTENCE_DELIMITERS.contains(&character) {
            // Trailing punctuation like "!!" belongs to one sentence end.
            while sentence
                .last()
                .is_some_and(|previous| SENTENCE_DELIMITERS.contains(previous))
            {
                sentence.pop();
            }
            result.extend(sentence.iter());
            sentence.clear();
            for _ in 0..=extra_sentences {
                result.push(character);
            }
        } else if CLAUSE_DELIMITERS.contains(&character) {
            result.extend(sentence.iter());
            sentence.clear();
            for _ in 0..=extra_clauses {
                result.push(character);
            }
        } else {
            sentence.push(character);
        }
    }
    result.extend(sentence.iter());
    result
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
        return wait_message(status, "Audio player");
    }
    #[cfg(not(target_os = "windows"))]
    {
        let mut command = Command::new(player);
        #[cfg(target_os = "linux")]
        command.args(["-nodisp", "-autoexit", "-loglevel", "error"]);
        let status = command.arg(path).status()?;
        wait_message(status, "Audio player")
    }
}

fn native_speak(text: &str, config: &AppConfig) -> Result<Option<PathBuf>> {
    #[cfg(target_os = "linux")]
    {
        let temp_path = unique_path(&temp_wav_directory(), "hivetts-speak", "wav");
        let mut command = espeak_command(text, Some(&temp_path), config)?;
        let status = command.status()?;
        wait_message(status, "Speech engine")?;
        let result = play_wav(&temp_path);
        let spoken_file = move_auto_save(&temp_path, config);
        if let Err(error) = fs::remove_file(&temp_path) {
            eprintln!("Could not remove temporary speech file: {error}");
        }
        return result.map(|()| spoken_file);
    }
    #[cfg(target_os = "macos")]
    {
        let mut command = Command::new("say");
        command.arg("-r").arg(words_per_minute(config)).arg(plain_text(text, config));
        let status = command.status()?;
        wait_message(status, "Speech engine").map(|()| None)
    }
    #[cfg(target_os = "windows")]
    {
        let _ = (text, config);
        Err(AppError::Message(
            "Real-time speech requires Piper on Windows; configure it in the STT settings.".into(),
        ))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    Err(AppError::Message(
        "No native speech engine for this platform.".into(),
    ))
}

#[cfg(target_os = "linux")]
fn espeak_command(
    text: &str,
    destination: Option<&Path>,
    config: &AppConfig,
) -> Result<Command> {
    let exe = find_executable("espeak-ng")
        .or_else(|| find_executable("espeak"))
        .ok_or_else(|| AppError::Message("Install eSpeak NG to synthesize speech.".into()))?;
    let mut command = Command::new(exe);
    command
        .args(["-s", &words_per_minute(config).to_string()])
        .args(["-p", &pitch_parameter(config).to_string()])
        .args(["-a", &amplitude_parameter(config).to_string()])
        .arg("-v")
        .arg(espeak_voice(config));
    if let Some(path) = destination {
        command.args(["-w"]).arg(path);
    }
    command.arg(plain_text(text, config));
    Ok(command)
}

/// Text passed to the engine with pause styling applied.
fn plain_text(text: &str, config: &AppConfig) -> String {
    with_pause_styling(text, config.pause_sentence, config.pause_paragraph)
}

/// espeak-ng range: -100 (low) … 100 (high); default 50 sits at the midpoint.
/// Voice Style provides the base shift and Speech Tone layers a smaller
/// adjustment on top so both controls remain audible.
fn pitch_parameter(config: &AppConfig) -> i32 {
    let style_shift: f32 = match Emotion::parse(&config.voice_style) {
        Emotion::Neutral => 0.0,
        Emotion::Happy => 12.0,
        Emotion::Excited => 18.0,
        Emotion::Friendly => 8.0,
        Emotion::Sad => -18.0,
        Emotion::Formal => -8.0,
        Emotion::Calm => -10.0,
    };
    let tone_shift = SpeechTone::parse(&config.speech_tone).pitch_shift();
    (50.0 + config.pitch.clamp(-50.0, 50.0) + style_shift + tone_shift).clamp(0.0, 100.0) as i32
}

/// espeak-ng amplitude (volume and intelligibility): 0 … 200, default 100.
fn amplitude_parameter(config: &AppConfig) -> i32 {
    (config.volume.clamp(0.0, 2.0) * 100.0).round() as i32
}

/// Words per minute, including the per-style tempo espeak-ng supports and a
/// smaller contribution from the chosen Speech Tone.
fn words_per_minute(config: &AppConfig) -> u32 {
    let style_rate = match Emotion::parse(&config.voice_style) {
        Emotion::Neutral => 0.0,
        Emotion::Happy => 10.0,
        Emotion::Excited => 25.0,
        Emotion::Friendly => 5.0,
        Emotion::Sad => -25.0,
        Emotion::Formal => -10.0,
        Emotion::Calm => -20.0,
    };
    let tone_rate = SpeechTone::parse(&config.speech_tone).rate_shift();
    ((175.0 * config.speech_rate.clamp(0.5, 2.0) + style_rate + tone_rate)
        .clamp(80.0, 450.0)) as u32
}

/// Build an espeak-ng voice string such as "en-us+f3" or "fa+m1".
#[cfg(target_os = "linux")]
fn espeak_voice(config: &AppConfig) -> String {
    let language = normalize_language(&config.tts_language);
    let variant = voice_variant(&config.voice, Emotion::parse(&config.voice_style));
    match (language, variant.is_empty()) {
        (None, true) => "en".to_string(),
        (Some(language), true) => language,
        (None, false) => format!("en{variant}"),
        (Some(language), false) => format!("{language}{variant}"),
    }
}

fn normalize_language(language: &str) -> Option<String> {
    let language = language.trim().to_lowercase();
    if language.is_empty() || language == "auto" {
        return None;
    }
    // Full espeak voice names with a regional subtag are accepted as-is.
    if matches!(language.as_str(), "en-us" | "en-gb" | "pt-br") {
        return Some(language);
    }
    Some(
        language
            .split(['-', '_', '+'])
            .next()
            .unwrap_or(language.as_str())
            .to_string(),
    )
}

fn voice_variant(voice: &str, emotion: Emotion) -> String {
    let voice = voice.trim().to_lowercase();
    let base = match voice.as_str() {
        "female" => "", // espeak's built-in default is a female-sounding voice.
        "male" => "+m1",
        "child" => "+c",
        "whisper" => "+whisper",
        "grotesque" => "+grotesque",
        // Custom espeak variants are passed through, e.g. "+f3" or "+fast".
        other if other.starts_with('+') => other,
        _ => "",
    };
    // Calm and similar tones stack an additional character on top.
    let extra = match emotion {
        Emotion::Calm if !base.contains("whisper") => "+whisper",
        _ => "",
    };
    format!("{base}{extra}")
}

/// Open a file or folder with the desktop's default handler.
pub fn open_path(path: &Path) -> Result<()> {
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("explorer");
        command.arg(path);
        command
    };
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("open");
        command.arg(path);
        command
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut command = Command::new("xdg-open");
        command.arg(path);
        command
    };
    #[cfg(not(any(unix, target_os = "windows")))]
    return Err(AppError::Message("Opening paths is unsupported.".into()));
    let status = command.status()?;
    // xdg-open and explorer often return non-zero even on success, so callers
    // treat launch failures only as errors.
    let _ = status;
    Ok(())
}

fn wait_message(status: std::process::ExitStatus, label: &str) -> Result<()> {
    if status.success() {
        Ok(())
    } else {
        Err(AppError::Message(format!(
            "{label} exited with {status}."
        )))
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
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emotion_parsing_is_case_insensitive() {
        assert_eq!(Emotion::parse("  HaPPy "), Emotion::Happy);
        // The new default voice-style value maps to Neutral.
        assert_eq!(Emotion::parse("standard"), Emotion::Neutral);
        assert_eq!(Emotion::parse("unknown"), Emotion::Neutral);
        assert_eq!(Emotion::Happy.to_string(), "happy");
        assert_eq!(SpeechTone::parse("Gentle"), SpeechTone::Gentle);
        assert_eq!(SpeechTone::parse("unknown"), SpeechTone::Neutral);
    }

    #[test]
    fn pause_styling_scales_sentence_and_clause_delimiters() {
        // The default sentence pause (0.5 s) leaves the text untouched.
        assert_eq!(with_pause_styling("Hello world! Fine, thanks.", 0.5, 0.5), "Hello world! Fine, thanks.");
        // A 1.5 s sentence pause doubles each delimiter and adds one comma per clause.
        let doubled = with_pause_styling("Hello world! Fine, thanks.", 1.5, 0.5);
        assert_eq!(doubled, "Hello world!! Fine,, thanks..");
        assert_eq!(with_pause_styling("", 1.5, 1.5), "");
        // Shorter pauses never drop the original delimiter.
        assert_eq!(with_pause_styling("Hi there.", 0.2, 0.2), "Hi there.");
    }

    #[test]
    fn pause_styling_stretches_paragraph_breaks() {
        // Default (0.5 s) sentence and paragraph pauses leave the text alone.
        assert_eq!(with_pause_styling("Hello.\nWorld.", 0.5, 0.5), "Hello.\nWorld.");
        // A 1.5 s paragraph pause stretches the gap before each `\n`.
        let styled = with_pause_styling("Hello.\nWorld.", 0.5, 1.5);
        assert_eq!(styled, "Hello..\nWorld.");
        // A bare newline without a preceding delimiter still gets a pause.
        let styled = with_pause_styling("Hello\nWorld", 0.5, 1.5);
        assert_eq!(styled, "Hello.\nWorld");
    }

    #[test]
    fn piper_text_restyles_terminal_punctuation_per_voice_style() {
        let friendly = AppConfig {
            voice_style: "friendly".into(),
            pause_sentence: 0.5,
            pause_paragraph: 0.5,
            ..AppConfig::default()
        };
        assert_eq!(piper_text("It is done", &friendly), "It is done!");
        let formal = AppConfig {
            voice_style: "formal".into(),
            ..AppConfig::default()
        };
        assert_eq!(piper_text("It is done", &formal), "It is done.");
        let calm = AppConfig {
            voice_style: "calm".into(),
            ..AppConfig::default()
        };
        assert_eq!(piper_text("It is done", &calm), "It is done.");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn espeak_voice_combines_language_gender_and_style() {
        let config = AppConfig {
            tts_language: "fas".into(),
            voice: "male".into(),
            voice_style: "standard".into(),
            ..AppConfig::default()
        };
        assert_eq!(espeak_voice(&config), "fas+m1");
        let config = AppConfig {
            tts_language: "auto".into(),
            voice: "female".into(),
            voice_style: "standard".into(),
            ..AppConfig::default()
        };
        assert_eq!(espeak_voice(&config), "en");
        let config = AppConfig {
            tts_language: "en-us".into(),
            voice: "whisper".into(),
            voice_style: "calm".into(),
            ..AppConfig::default()
        };
        assert_eq!(espeak_voice(&config), "en-us+whisper");
    }

    #[test]
    fn pitch_and_amplitude_stay_inside_engine_ranges() {
        let config = AppConfig {
            pitch: 50.0,
            volume: 2.0,
            voice_style: "friendly".into(),
            speech_tone: "conversational".into(),
            ..AppConfig::default()
        };
        assert_eq!(pitch_parameter(&config), 100);
        assert_eq!(amplitude_parameter(&config), 200);
        let config = AppConfig {
            pitch: -50.0,
            volume: 0.0,
            voice_style: "calm".into(),
            speech_tone: "gentle".into(),
            ..AppConfig::default()
        };
        assert_eq!(pitch_parameter(&config), 0); // clamps at the minimum.
        assert_eq!(amplitude_parameter(&config), 0);
        assert!((80..=450).contains(&words_per_minute(&config)));
    }

    #[test]
    fn speaker_ids_only_parse_digits() {
        assert_eq!(speaker_id_index("Male (2)").as_deref(), Some("2"));
        assert_eq!(speaker_id_index("female"), None);
    }

    #[test]
    fn auto_save_requires_a_directory() {
        let config = AppConfig {
            auto_save_directory: String::new(),
            ..AppConfig::default()
        };
        assert!(auto_save_path(&config, "wav").is_none());
        let config = AppConfig {
            auto_save_directory: env::temp_dir().display().to_string(),
            ..AppConfig::default()
        };
        let path = auto_save_path(&config, "wav").expect("path generated");
        assert_eq!(path.extension().unwrap(), "wav");
        assert!(path.file_name().unwrap().to_string_lossy().starts_with("hivetts-speech-"));
    }

    #[test]
    fn export_formats_and_quality_presets_parse() {
        assert_eq!(ExportFormat::parse("MP3"), ExportFormat::Mp3);
        assert_eq!(ExportFormat::parse(" ogg "), ExportFormat::Ogg);
        assert_eq!(ExportFormat::parse("unknown"), ExportFormat::Wav);
        assert_eq!(ExportFormat::Ogg.extension(), "ogg");
        assert_eq!(ExportFormat::Mp3.label(), "MP3");
        assert_eq!(OutputQuality::parse("low"), OutputQuality::Low);
        assert_eq!(OutputQuality::parse("standard"), OutputQuality::Standard);
        assert_eq!(OutputQuality::parse("maximum"), OutputQuality::Maximum);
        // The default preset is High.
        assert_eq!(OutputQuality::parse(""), OutputQuality::High);
        assert_eq!(OutputQuality::High.mp3_bitrate(), "256k");
        assert_eq!(OutputQuality::Maximum.ogg_quality(), "10");
        assert_eq!(OutputQuality::Low.wav_sample_rate(), 16000);
    }

    #[test]
    fn wav_sample_rate_reads_the_fmt_chunk() {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // one channel
        bytes.extend_from_slice(&22050u32.to_le_bytes()); // sample rate
        assert_eq!(read_wav_sample_rate(&bytes), Some(22050));
        assert_eq!(read_wav_sample_rate(b"not a wav at all"), None);
    }

    #[test]
    fn word_timeline_is_monotonic_and_within_duration() {
        let words = vec![
            "Hello".to_string(),
            "world.".to_string(),
            "Fine,".to_string(),
            "thanks".to_string(),
        ];
        let timeline = word_timeline(&words, 10.0);
        assert_eq!(timeline.len(), words.len());
        assert_eq!(timeline[0].start, 0.0);
        assert!(timeline.windows(2).all(|pair| pair[0].start <= pair[1].start));
        assert!(timeline.last().unwrap().start < 10.0);
        assert_eq!(timeline[0].word, "Hello");
    }

    #[test]
    fn word_timeline_is_empty_without_duration_or_words() {
        assert!(word_timeline(&[], 5.0).is_empty());
        assert!(word_timeline(&["Hi".to_string()], 0.0).is_empty());
    }

    #[test]
    fn wav_duration_reads_a_minimal_riff_header() {
        // 16-bit mono at 8000 Hz => byte rate 16000; 16000 data bytes => 1 s.
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&0u32.to_le_bytes()); // overall size (unused)
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // one channel
        bytes.extend_from_slice(&8000u32.to_le_bytes()); // sample rate
        bytes.extend_from_slice(&16000u32.to_le_bytes()); // byte rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&16000u32.to_le_bytes()); // data size
        bytes.extend_from_slice(&vec![0u8; 16000]);
        assert_eq!(read_wav_duration(&bytes), Some(1.0));
        assert_eq!(read_wav_duration(b"not a wav at all"), None);
    }
}
