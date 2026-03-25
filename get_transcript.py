#!/usr/bin/env python3
"""Fetch a YouTube transcript with timestamps and save it as a .txt file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def is_valid_video_id(video_id: str) -> bool:
    return bool(VIDEO_ID_PATTERN.fullmatch(video_id))


def extract_video_id(value: str) -> str:
    value = value.strip()

    if is_valid_video_id(value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        if is_valid_video_id(candidate):
            return candidate

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        query_params = parse_qs(parsed.query)
        v_param = query_params.get("v", [""])[0]
        if is_valid_video_id(v_param):
            return v_param

        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "v", "live"}:
            candidate = path_parts[1]
            if is_valid_video_id(candidate):
                return candidate

    raise ValueError(
        "Could not extract a valid video ID from the provided URL/input."
    )


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_output_lines(transcript) -> list[str]:
    lines: list[str] = []
    for snippet in transcript:
        text = " ".join(snippet.text.split())
        lines.append(f"[{format_timestamp(snippet.start)}] {text}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a YouTube transcript and save it as a timestamped .txt file."
    )
    parser.add_argument(
        "url",
        help="YouTube URL (or a direct 11-character video ID).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Preferred language codes in priority order, for example: --languages en es",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        video_id = extract_video_id(args.url)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    api = YouTubeTranscriptApi()

    try:
        if args.languages:
            transcript = api.fetch(video_id, languages=args.languages)
        else:
            transcript = api.fetch(video_id)
    except Exception as error:  # noqa: BLE001
        print(f"Error fetching transcript: {error}", file=sys.stderr)
        return 1

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / f"{video_id}_transcript.txt"
    output_text = "\n".join(build_output_lines(transcript)) + "\n"

    output_path.write_text(output_text, encoding="utf-8")
    print(f"Transcript saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
