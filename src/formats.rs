pub fn format_srt(segments: &[(f64, f64, String)]) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(index, (start, end, text))| {
            format!(
                "{}\n{} --> {}\n{}\n",
                index + 1,
                format_timestamp(*start, ','),
                format_timestamp(*end, ','),
                text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_vtt(segments: &[(f64, f64, String)]) -> String {
    let mut output = String::from("WEBVTT\n\n");
    for (start, end, text) in segments {
        output.push_str(&format!(
            "{} --> {}\n{}\n\n",
            format_timestamp(*start, '.'),
            format_timestamp(*end, '.'),
            text.trim()
        ));
    }
    output
}

fn format_timestamp(seconds: f64, decimal: char) -> String {
    let milliseconds = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds / 60_000) % 60;
    let seconds = (milliseconds / 1_000) % 60;
    let millis = milliseconds % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{decimal}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtitle_formats_have_correct_headers_and_timestamps() {
        let segments = vec![(1.25, 3.5, " Hello ".to_string())];
        assert_eq!(
            format_srt(&segments),
            "1\n00:00:01,250 --> 00:00:03,500\nHello\n"
        );
        assert_eq!(
            format_vtt(&segments),
            "WEBVTT\n\n00:00:01.250 --> 00:00:03.500\nHello\n\n"
        );
    }

    #[test]
    fn negative_timestamps_are_clamped_to_zero() {
        assert_eq!(format_timestamp(-2.0, ','), "00:00:00,000");
    }
}
