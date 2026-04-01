import logging
import math
import os
from fractions import Fraction
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TEXT_SUBTITLE_CODECS = {
    "ass",
    "mov_text",
    "srt",
    "ssa",
    "subrip",
    "text",
    "webvtt",
}

_AUDIO_STREAM_TARGETS = {
    "aac": {"ext": ".m4a", "format": "ipod"},
    "ac3": {"ext": ".ac3", "format": "ac3"},
    "alac": {"ext": ".m4a", "format": "ipod"},
    "eac3": {"ext": ".eac3", "format": "eac3"},
    "flac": {"ext": ".flac", "format": "flac"},
    "mp3": {"ext": ".mp3", "format": "mp3"},
    "opus": {"ext": ".ogg", "format": "ogg"},
    "pcm_s16le": {"ext": ".wav", "format": "wav"},
    "vorbis": {"ext": ".ogg", "format": "ogg"},
}

_SUBTITLE_STREAM_TARGETS = {
    "ass": {"ext": ".ass", "codec": "ass", "format": "ass"},
    "mov_text": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "srt": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "ssa": {"ext": ".ass", "codec": "ass", "format": "ass"},
    "subrip": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "text": {"ext": ".srt", "codec": "srt", "format": "srt"},
    "webvtt": {"ext": ".vtt", "codec": "webvtt", "format": "webvtt"},
}


def probe_media_streams(video_path: str) -> dict[str, Any]:
    import ffmpeg

    return ffmpeg.probe(video_path)


def probe_video_metadata(
    probe_data: dict[str, Any], frame_count: int
) -> tuple[float, float]:
    video_stream = next(
        stream
        for stream in probe_data["streams"]
        if stream["codec_type"] == "video"
    )
    duration = video_stream.get("duration") or probe_data["format"].get("duration")
    if duration is None:
        raise RuntimeError("Unable to determine video duration")
    duration_seconds = float(duration)

    fps = 0.0
    for rate_key in ("avg_frame_rate", "r_frame_rate"):
        raw_rate = video_stream.get(rate_key)
        if not raw_rate:
            continue
        try:
            fps = float(Fraction(raw_rate))
        except (ValueError, ZeroDivisionError):
            continue
        if fps > 0:
            break

    if fps <= 0 and frame_count > 0 and duration_seconds > 0:
        fps = frame_count / duration_seconds

    return duration_seconds, fps


def extract_audio_stream(
    video_path: str, output_dir: str, task_id: str, probe_data: dict[str, Any]
) -> Optional[dict[str, Any]]:
    import ffmpeg

    audio_streams = [
        stream
        for stream in probe_data.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if not audio_streams:
        return None

    stream = audio_streams[0]
    codec_name = (stream.get("codec_name") or "").lower()
    target = _AUDIO_STREAM_TARGETS.get(
        codec_name, {"ext": ".mka", "format": "matroska"}
    )
    output_path = os.path.join(output_dir, f"{task_id}{target['ext']}")

    try:
        (
            ffmpeg.input(video_path)
            .output(
                output_path,
                map="0:a:0",
                acodec="copy",
                format=target["format"],
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(
            f"Failed to extract audio stream from {video_path}: {stderr}"
        ) from exc

    return {
        "stream_index": stream.get("index"),
        "codec_name": stream.get("codec_name"),
        "channels": stream.get("channels"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
        "language": (stream.get("tags") or {}).get("language"),
        "local_path": output_path,
        "file_ext": target["ext"],
    }


def extract_subtitle_streams(
    video_path: str, output_dir: str, task_id: str, probe_data: dict[str, Any]
) -> list[dict[str, Any]]:
    import ffmpeg

    subtitle_streams = [
        stream
        for stream in probe_data.get("streams", [])
        if stream.get("codec_type") == "subtitle"
    ]

    extracted_subtitles = []
    for subtitle_order, stream in enumerate(subtitle_streams):
        codec_name = (stream.get("codec_name") or "").lower()
        disposition = stream.get("disposition") or {}
        stream_info = {
            "stream_index": stream.get("index"),
            "codec_name": stream.get("codec_name"),
            "language": (stream.get("tags") or {}).get("language"),
            "title": (stream.get("tags") or {}).get("title"),
            "default": bool(disposition.get("default")),
            "forced": bool(disposition.get("forced")),
            "extractable": codec_name in _TEXT_SUBTITLE_CODECS,
        }

        target = _SUBTITLE_STREAM_TARGETS.get(codec_name)
        if target is None:
            extracted_subtitles.append(stream_info)
            continue

        output_path = os.path.join(
            output_dir,
            f"{task_id}-subtitle-{subtitle_order}{target['ext']}",
        )

        try:
            (
                ffmpeg.input(video_path)
                .output(
                    output_path,
                    map=f"0:s:{subtitle_order}",
                    format=target["format"],
                    **{"c:s": target["codec"]},
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = (
                exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            )
            raise RuntimeError(
                "Failed to extract subtitle stream "
                f"{stream.get('index')} from {video_path}: {stderr}"
            ) from exc

        stream_info["local_path"] = output_path
        stream_info["file_ext"] = target["ext"]
        extracted_subtitles.append(stream_info)

    return extracted_subtitles


def build_scene_sampling_plan(
    scenes: list[list[int]],
    frame_count: int,
    duration_seconds: float,
    max_interval_seconds: float,
) -> list[dict[str, Any]]:
    if frame_count <= 0 or duration_seconds <= 0:
        return []

    average_fps = frame_count / duration_seconds
    max_frames_per_sample = max(1, math.ceil(average_fps * max_interval_seconds))
    scene_previews = []

    for scene_index, (start_frame, end_frame) in enumerate(scenes):
        scene_frame_count = end_frame - start_frame + 1
        sample_count = max(1, math.ceil(scene_frame_count / max_frames_per_sample))
        sampled_frames = []

        for sample_index in range(sample_count):
            segment_start = sample_index * scene_frame_count / sample_count
            segment_end = (sample_index + 1) * scene_frame_count / sample_count
            middle_offset = int((segment_start + segment_end - 1) / 2)
            frame_id = start_frame + middle_offset
            frame_id = min(max(frame_id, start_frame), end_frame)
            sampled_frames.append(
                {
                    "sample_index": sample_index,
                    "frame_id": frame_id,
                }
            )

        scene_previews.append(
            {
                "scene_index": scene_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "sample_count": sample_count,
                "sampled_frames": sampled_frames,
            }
        )

    return scene_previews
