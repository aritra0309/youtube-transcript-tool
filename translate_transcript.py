#!/usr/bin/env python3
"""Translate transcript .txt files to English while preserving timestamps."""

from __future__ import annotations

import argparse
import asyncio
import html
import os
import random
import re
import sys
from collections import Counter
from contextlib import AsyncExitStack
from pathlib import Path
from typing import NamedTuple

import httpx
from googletrans import Translator


TIMESTAMP_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")
GOOGLE_TRANSLATE_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"


class TranslationJob(NamedTuple):
    index: int
    original_line: str
    text_to_translate: str
    timestamp: str | None


class ChunkOutcome(NamedTuple):
    updates: list[tuple[int, str]]
    translated_count: int
    fallback_count: int
    retries_used: int
    provider_used: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate a transcript .txt file to English and save it as another .txt file."
        )
    )
    parser.add_argument("input_file", help="Path to input .txt transcript file.")
    parser.add_argument(
        "--output",
        help=(
            "Optional output .txt path. Defaults to <input_basename>_en.txt in the same "
            "folder as the input file."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "google", "googletrans"],
        default="auto",
        help=(
            "Translation provider mode. 'auto' uses official Google API when "
            "GOOGLE_TRANSLATE_API_KEY is present, otherwise uses googletrans."
        ),
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback provider and only use the selected primary provider.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of lines to translate per batch request (default: 50).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum total characters per batch request (default: 12000).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of batch requests to run in parallel (default: 3).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per provider for each batch (default: 3).",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=0.6,
        help="Base delay in seconds for exponential retry backoff (default: 0.6).",
    )
    return parser.parse_args()


def resolve_input_path(path_value: str) -> Path:
    input_path = Path(path_value).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() != ".txt":
        raise ValueError("Input file must be a .txt file")

    return input_path


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg).expanduser().resolve()
        if output_path.suffix.lower() != ".txt":
            raise ValueError("Output file must have a .txt extension")
        return output_path

    return input_path.with_name(f"{input_path.stem}_en.txt")


def resolve_provider_chain(
    provider_mode: str, api_key: str | None, allow_fallback: bool
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []

    if provider_mode == "auto":
        if api_key:
            providers = ["google", "googletrans"] if allow_fallback else ["google"]
        else:
            providers = ["googletrans"]
            warnings.append(
                "GOOGLE_TRANSLATE_API_KEY not set; using googletrans as primary provider."
            )
        return providers, warnings

    if provider_mode == "google":
        if api_key:
            providers = ["google", "googletrans"] if allow_fallback else ["google"]
            return providers, warnings

        if allow_fallback:
            warnings.append(
                "Google provider selected but GOOGLE_TRANSLATE_API_KEY is missing; "
                "falling back to googletrans."
            )
            return ["googletrans"], warnings

        raise ValueError(
            "Google provider selected but GOOGLE_TRANSLATE_API_KEY is not set."
        )

    return ["googletrans"], warnings


def build_translation_jobs(input_lines: list[str]) -> list[TranslationJob]:
    jobs: list[TranslationJob] = []

    for i, line in enumerate(input_lines):
        if not line.strip():
            continue

        timestamp_match = TIMESTAMP_LINE_RE.match(line)
        if timestamp_match:
            timestamp, text = timestamp_match.groups()
            if text.strip():
                jobs.append(
                    TranslationJob(
                        index=i,
                        original_line=line,
                        text_to_translate=text,
                        timestamp=timestamp,
                    )
                )
            continue

        jobs.append(
            TranslationJob(
                index=i,
                original_line=line,
                text_to_translate=line,
                timestamp=None,
            )
        )

    return jobs


def build_chunks(
    jobs: list[TranslationJob], batch_size: int, max_chars: int
) -> list[list[TranslationJob]]:
    chunks: list[list[TranslationJob]] = []
    current_chunk: list[TranslationJob] = []
    current_chars = 0

    for job in jobs:
        job_chars = max(1, len(job.text_to_translate))
        should_flush = bool(current_chunk) and (
            len(current_chunk) >= batch_size or current_chars + job_chars > max_chars
        )

        if should_flush:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0

        current_chunk.append(job)
        current_chars += job_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def run_with_retries(coro_factory, retries: int, base_delay: float):
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await coro_factory(), attempt, None
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt == retries:
                break

            backoff = base_delay * (2**attempt)
            jitter = random.uniform(0, base_delay)
            await asyncio.sleep(backoff + jitter)

    return None, retries, last_error


async def translate_with_google(
    client: httpx.AsyncClient, api_key: str, texts: list[str]
) -> list[str]:
    response = await client.post(
        GOOGLE_TRANSLATE_ENDPOINT,
        params={"key": api_key},
        json={"q": texts, "target": "en", "format": "text"},
    )
    response.raise_for_status()

    payload = response.json()
    translations = payload.get("data", {}).get("translations", [])
    if len(translations) != len(texts):
        raise RuntimeError("Google Translate API returned unexpected response length")

    return [html.unescape(item.get("translatedText", "")) for item in translations]


async def translate_with_googletrans(
    translator: Translator, texts: list[str]
) -> list[str]:
    result = await translator.translate(texts, dest="en")
    if isinstance(result, list):
        translations = result
    else:
        translations = [result]

    if len(translations) != len(texts):
        raise RuntimeError("googletrans returned unexpected response length")

    return [item.text for item in translations]


def format_updates(chunk: list[TranslationJob], translated_texts: list[str]) -> list[tuple[int, str]]:
    updates: list[tuple[int, str]] = []
    for job, translated_text in zip(chunk, translated_texts):
        if job.timestamp is None:
            updates.append((job.index, translated_text))
        else:
            updates.append((job.index, f"[{job.timestamp}] {translated_text}"))
    return updates


async def process_chunk(
    chunk: list[TranslationJob],
    provider_chain: list[str],
    retries: int,
    retry_base_delay: float,
    google_client: httpx.AsyncClient | None,
    google_api_key: str | None,
    translator: Translator | None,
    semaphore: asyncio.Semaphore,
) -> ChunkOutcome:
    texts = [job.text_to_translate for job in chunk]
    retries_used = 0

    async with semaphore:
        for provider in provider_chain:
            if provider == "google":
                if google_client is None or not google_api_key:
                    continue

                translated, attempts, error = await run_with_retries(
                    lambda: translate_with_google(google_client, google_api_key, texts),
                    retries,
                    retry_base_delay,
                )
                retries_used += attempts
                if error is None and translated is not None:
                    return ChunkOutcome(
                        updates=format_updates(chunk, translated),
                        translated_count=len(chunk),
                        fallback_count=0,
                        retries_used=retries_used,
                        provider_used="google",
                    )
                continue

            if provider == "googletrans":
                if translator is None:
                    continue

                translated, attempts, error = await run_with_retries(
                    lambda: translate_with_googletrans(translator, texts),
                    retries,
                    retry_base_delay,
                )
                retries_used += attempts
                if error is None and translated is not None:
                    return ChunkOutcome(
                        updates=format_updates(chunk, translated),
                        translated_count=len(chunk),
                        fallback_count=0,
                        retries_used=retries_used,
                        provider_used="googletrans",
                    )

    return ChunkOutcome(
        updates=[(job.index, job.original_line) for job in chunk],
        translated_count=0,
        fallback_count=len(chunk),
        retries_used=retries_used,
        provider_used="original",
    )


async def translate_file(
    input_path: Path,
    output_path: Path,
    provider_chain: list[str],
    google_api_key: str | None,
    batch_size: int,
    max_chars: int,
    concurrency: int,
    retries: int,
    retry_base_delay: float,
) -> tuple[int, int, int, int, int, Counter, Counter]:
    input_text = input_path.read_text(encoding="utf-8")
    if not input_text.strip():
        raise ValueError("Input file is empty")

    input_had_trailing_newline = input_text.endswith("\n")
    input_lines = input_text.splitlines()
    jobs = build_translation_jobs(input_lines)
    chunks = build_chunks(jobs, batch_size, max_chars)

    output_lines: list[str] = list(input_lines)
    translated_count = 0
    fallback_count = 0
    retries_used_total = 0
    provider_chunk_counts: Counter = Counter()
    provider_line_counts: Counter = Counter()

    async with AsyncExitStack() as stack:
        google_client = None
        translator = None

        if "google" in provider_chain:
            google_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=30.0)
            )

        if "googletrans" in provider_chain:
            translator = await stack.enter_async_context(
                Translator(service_urls=["translate.googleapis.com"])
            )

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            asyncio.create_task(
                process_chunk(
                    chunk=chunk,
                    provider_chain=provider_chain,
                    retries=retries,
                    retry_base_delay=retry_base_delay,
                    google_client=google_client,
                    google_api_key=google_api_key,
                    translator=translator,
                    semaphore=semaphore,
                )
            )
            for chunk in chunks
        ]

        outcomes = await asyncio.gather(*tasks)

    for outcome in outcomes:
        for index, line in outcome.updates:
            output_lines[index] = line

        translated_count += outcome.translated_count
        fallback_count += outcome.fallback_count
        retries_used_total += outcome.retries_used
        provider_chunk_counts[outcome.provider_used] += 1
        provider_line_counts[outcome.provider_used] += (
            outcome.translated_count + outcome.fallback_count
        )

    output_text = "\n".join(output_lines)
    if input_had_trailing_newline:
        output_text += "\n"

    output_path.write_text(output_text, encoding="utf-8")

    return (
        len(input_lines),
        translated_count,
        fallback_count,
        retries_used_total,
        len(chunks),
        provider_chunk_counts,
        provider_line_counts,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.max_chars <= 0:
        raise ValueError("--max-chars must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.retry_base_delay <= 0:
        raise ValueError("--retry-base-delay must be greater than 0")


def format_counter(counter: Counter) -> str:
    if not counter:
        return "none"
    parts = [f"{key}={counter[key]}" for key in sorted(counter.keys())]
    return ", ".join(parts)


def main() -> int:
    args = parse_args()
    google_api_key = None

    try:
        validate_args(args)
        input_path = resolve_input_path(args.input_file)
        output_path = resolve_output_path(input_path, args.output)
        google_api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip() or None
        provider_chain, warnings = resolve_provider_chain(
            provider_mode=args.provider,
            api_key=google_api_key,
            allow_fallback=not args.no_fallback,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    try:
        (
            total_lines,
            translated_count,
            fallback_count,
            retries_used_total,
            chunk_count,
            provider_chunk_counts,
            provider_line_counts,
        ) = asyncio.run(
            translate_file(
                input_path=input_path,
                output_path=output_path,
                provider_chain=provider_chain,
                google_api_key=google_api_key,
                batch_size=args.batch_size,
                max_chars=args.max_chars,
                concurrency=args.concurrency,
                retries=args.retries,
                retry_base_delay=args.retry_base_delay,
            )
        )
    except Exception as error:  # noqa: BLE001
        print(f"Error during translation: {error}", file=sys.stderr)
        return 1

    print(f"Translated file saved to: {output_path}")
    print(
        "Lines processed: "
        f"{total_lines} | translated: {translated_count} | kept original (fallback): {fallback_count}"
    )
    print(
        "Batches processed: "
        f"{chunk_count} | total retries used: {retries_used_total} | concurrency: {args.concurrency}"
    )
    print(f"Providers used (chunks): {format_counter(provider_chunk_counts)}")
    print(f"Providers used (lines): {format_counter(provider_line_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
