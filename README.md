# YouTube Transcript + Translation + QA Evidence Toolkit

Small standalone toolkit for:

- pulling timestamped YouTube transcripts,
- translating transcript files to English,
- and running automated QA evidence scoring for RAG-style evaluation.

## Files In This Repo

- `get_transcript.py`: Fetches YouTube transcript and saves timestamped `.txt`
- `translate_transcript.py`: Translates transcript text to English while preserving timestamps
- `build_qa_evidence.py`: Scores QA pairs against transcript windows and exports evidence artifacts

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install youtube-transcript-api googletrans==4.0.2 httpx
```

## 1) Fetch Transcript

```bash
python3 get_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Optional language priority:

```bash
python3 get_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID" --languages en
```

Output:

- `<video_id>_transcript.txt`
- line format: `[HH:MM:SS] transcript text`

## 2) Translate Transcript To English

Default output in same folder:

```bash
python3 translate_transcript.py "input_transcript.txt"
```

Custom output path:

```bash
python3 translate_transcript.py "input_transcript.txt" --output "translated_output.txt"
```

Provider behavior:

- `--provider auto` (default): uses Google API when `GOOGLE_TRANSLATE_API_KEY` is set, otherwise `googletrans`
- `--provider google`: only Google API (unless fallback allowed and key missing)
- `--provider googletrans`: only googletrans

Example with API key:

```bash
export GOOGLE_TRANSLATE_API_KEY="YOUR_API_KEY"
python3 translate_transcript.py "input_transcript.txt" --provider auto --concurrency 3
```

Notes:

- timestamp tags are preserved (`[HH:MM:SS]`)
- only transcript text after timestamps is translated
- failed lines are kept as original text and processing continues

## 3) Build QA Evidence + Scoring

This script expects a user-provided `qa_pairs.json` file in the same folder.

Run:

```bash
python3 build_qa_evidence.py
```

Generated artifacts:

- `qa_evidence.json`
- `retrieval_checks.csv`
- `method_log.txt`

### `qa_pairs.json` schema

```json
[
  {
    "id": "qa1",
    "question": "...",
    "answer": "...",
    "source": {
      "title": "...",
      "video_id": "...",
      "transcript_file": "video_transcript.txt",
      "start": "00:00:00",
      "end": "00:01:00"
    }
  }
]
```

## Security / Public Repo Tips

- Never hardcode API keys in scripts.
- Use environment variables (for example `GOOGLE_TRANSLATE_API_KEY`).
- Add generated outputs (`*.txt`, `*.json`, `*.csv`, logs, transcripts) to `.gitignore` if needed.
- Rotate any key that was ever shared in chat, screenshots, or commit history.
