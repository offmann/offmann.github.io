#!/usr/bin/env python3
"""Generate OpenAI TTS audio for article pages and embed audio players."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List
from urllib import error, request


DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_SUMMARY_MODEL = "gpt-5"
DEFAULT_VOICE = "alloy"
DEFAULT_FORMAT = "mp3"
MAX_CHARS_PER_CHUNK = 3500
DEFAULT_SUMMARY_WORDS = 140


class ArticleTextExtractor(HTMLParser):
    """Extract readable text from the <article> block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_article = False
        self._ignore = False
        self._text_parts: List[str] = []
        self._title_parts: List[str] = []
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "article":
            self._in_article = True
        if not self._in_article:
            return
        if tag in {"script", "style"}:
            self._ignore = True
        if tag == "h1":
            self._capture_title = True
        if tag in {"p", "li", "h1", "h2", "h3", "blockquote"}:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self._in_article = False
        if tag in {"script", "style"}:
            self._ignore = False
        if tag == "h1":
            self._capture_title = False
        if self._in_article and tag in {"p", "li", "h1", "h2", "h3", "blockquote"}:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._in_article or self._ignore:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._capture_title:
            self._title_parts.append(text)
        self._text_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        raw = "\n".join(self._text_parts)
        cleaned = re.sub(r"\n{3,}", "\n\n", raw).strip()
        return html.unescape(cleaned)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        add_len = len(paragraph) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunks.append("\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
            continue

        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + max_chars])
                start += max_chars
            continue

        current.append(paragraph)
        current_len += add_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def openai_json_request(api_key: str, endpoint: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed ({exc.code}): {detail}") from exc


def summarize_article(
    api_key: str,
    title: str,
    article_text: str,
    summary_model: str,
    summary_word_count: int,
) -> str:
    system_prompt = (
        "You summarize personal blog articles for spoken audio narration. "
        "Your output must sound natural when listened to, not read. "
        "Keep the core premise and main idea behind the article. "
        "Use plain language, short flowing paragraphs, and no bullet points. "
        "Do not invent facts."
    )
    user_prompt = (
        f"Title: {title}\n\n"
        f"Target length: about {summary_word_count} words (roughly one minute of audio).\n\n"
        "Create a concise spoken summary that lets a listener quickly grasp the premise and central idea. "
        "No bullets or numbered lists.\n\n"
        f"Article:\n{article_text}"
    )
    payload = {
        "model": summary_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = openai_json_request(api_key, "https://api.openai.com/v1/responses", payload)

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output", [])
    extracted: List[str] = []
    for item in output:
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                extracted.append(content["text"])

    summary = "\n".join(extracted).strip()
    if not summary:
        raise RuntimeError("Summary response did not contain text output.")
    return summary


def synthesize_speech(api_key: str, text: str, model: str, voice: str, audio_format: str) -> bytes:
    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "format": audio_format,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/audio/speech",
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TTS request failed ({exc.code}): {detail}") from exc


def extract_article_content(article_path: Path) -> tuple[str, str]:
    parser = ArticleTextExtractor()
    parser.feed(article_path.read_text(encoding="utf-8"))
    title = parser.title or article_path.stem.replace("-", " ").title()
    body = parser.text
    full_text = f"{title}\n\n{body}".strip()
    return title, full_text


def generate_audio_for_articles(
    repo_root: Path,
    article_paths: Iterable[Path],
    force: bool,
    model: str,
    summary_model: str,
    summary_word_count: int,
    voice: str,
    audio_format: str,
) -> None:
    load_env_file(repo_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in environment or .env file.")

    audio_dir = repo_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    ext = "mp3" if audio_format == "mp3" else audio_format

    for article_path in article_paths:
        slug = article_path.stem
        output_path = audio_dir / f"{slug}.{ext}"
        if output_path.exists() and not force:
            print(f"[skip] {output_path} already exists")
            continue

        title, article_text = extract_article_content(article_path)
        print(f"[summary] {article_path.name}")
        try:
            summary_text = summarize_article(
                api_key=api_key,
                title=title,
                article_text=article_text,
                summary_model=summary_model,
                summary_word_count=summary_word_count,
            )
        except RuntimeError as exc:
            print(f"[error] failed to summarize {article_path.name}: {exc}")
            continue

        chunks = chunk_text(summary_text)
        print(f"[tts] {article_path.name} summary ({len(chunks)} chunk(s))")

        temp_chunks: List[bytes] = []
        try:
            for idx, chunk in enumerate(chunks, start=1):
                print(f"  - chunk {idx}/{len(chunks)}")
                audio_bytes = synthesize_speech(
                    api_key=api_key,
                    text=chunk,
                    model=model,
                    voice=voice,
                    audio_format=audio_format,
                )
                temp_chunks.append(audio_bytes)

            output_path.write_bytes(b"".join(temp_chunks))
            print(f"[ok] wrote {output_path} for '{title}'")
        except RuntimeError as exc:
            if output_path.exists():
                output_path.unlink()
            print(f"[error] failed to generate {article_path.name}: {exc}")


def embed_audio_players(article_paths: Iterable[Path], audio_extension: str = "mp3") -> None:
    marker = 'data-article-audio="true"'
    section_anchor = '<section class="space-y-6 leading-relaxed">'

    for article_path in article_paths:
        html_text = article_path.read_text(encoding="utf-8")
        if marker in html_text:
            print(f"[skip] player already embedded in {article_path.name}")
            continue
        if section_anchor not in html_text:
            print(f"[warn] unable to find article section in {article_path.name}")
            continue

        slug = article_path.stem
        player_block = (
            '      <div class="mb-8 p-4 rounded-2xl border border-gray-200 bg-gray-50" data-article-audio="true">\n'
            '        <p class="text-sm text-gray-700 mb-2">Listen to summary</p>\n'
            f'        <audio controls preload="none" class="w-full">\n'
            f'          <source src="../audio/{slug}.{audio_extension}" type="audio/mpeg" />\n'
            "          Your browser does not support the audio element.\n"
            "        </audio>\n"
            f'        <p class="text-xs text-gray-500 mt-2"><a class="underline" href="../audio/{slug}.{audio_extension}">Download audio</a></p>\n'
            "      </div>\n\n"
        )

        updated = html_text.replace(f"      {section_anchor}", player_block + f"      {section_anchor}", 1)
        article_path.write_text(updated, encoding="utf-8")
        print(f"[ok] embedded player in {article_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate article TTS and embed audio players.")
    parser.add_argument(
        "--articles-dir",
        default="articles",
        help="Path to the articles directory (default: articles)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of article slugs (without .html) to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate audio even if output file already exists",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"TTS model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--summary-model",
        default=DEFAULT_SUMMARY_MODEL,
        help=f"Summary model (default: {DEFAULT_SUMMARY_MODEL})",
    )
    parser.add_argument(
        "--summary-words",
        type=int,
        default=DEFAULT_SUMMARY_WORDS,
        help=f"Target summary length in words (default: {DEFAULT_SUMMARY_WORDS})",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"TTS voice (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        choices=["mp3", "opus", "aac", "flac", "wav", "pcm"],
        help=f"Audio output format (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip TTS generation and only embed players",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip HTML embedding and only generate audio",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    articles_dir = (repo_root / args.articles_dir).resolve()

    if not articles_dir.exists():
        print(f"Articles directory not found: {articles_dir}", file=sys.stderr)
        return 1

    article_paths = sorted(articles_dir.glob("*.html"))
    article_paths = [p for p in article_paths if p.name != "index.html"]

    if args.only:
        wanted = {slug.strip() for slug in args.only if slug.strip()}
        article_paths = [p for p in article_paths if p.stem in wanted]

    if not article_paths:
        print("No article files matched the selection.")
        return 1

    if not args.skip_generate:
        generate_audio_for_articles(
            repo_root=repo_root,
            article_paths=article_paths,
            force=args.force,
            model=args.model,
            summary_model=args.summary_model,
            summary_word_count=args.summary_words,
            voice=args.voice,
            audio_format=args.format,
        )

    if not args.skip_embed:
        audio_extension = "mp3" if args.format == "mp3" else args.format
        embed_audio_players(article_paths=article_paths, audio_extension=audio_extension)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
