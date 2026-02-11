#!/usr/bin/env python3
"""
Fetch the latest earnings-related 8-K exhibits (EX-99.1 / EX-99.2) from SEC EDGAR.

Usage:
  python sec_earnings_8k.py --ticker COST
  python sec_earnings_8k.py --ticker COST --date 2025-12-11 --outdir ./downloads
"""

from __future__ import annotations

import argparse
import gzip
import html as html_lib
import json
import os
import re
import sys
import time
import shutil
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import datetime
from html.parser import HTMLParser
from typing import Iterable, Optional

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
DUCKDUCKGO_HTML_SEARCH = "https://duckduckgo.com/html/?q={query}"
BING_HTML_SEARCH = "https://www.bing.com/search?q={query}&setlang=en-us"
SEARCH_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

DEFAULT_OUTDIR = "./sec_earnings_8k"
DEFAULT_TRANSCRIPT_COOKIE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".investing_cookie.txt")
DEFAULT_TRANSCRIPT_STEM = "earnings_call_transcript"


class TranscriptSearchError(RuntimeError):
    pass


def _http_get(
    url: str,
    user_agent: str,
    ssl_context: Optional[ssl.SSLContext],
    extra_headers: Optional[dict[str, str]] = None,
) -> bytes:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30, context=ssl_context) as resp:
        data = resp.read()
        encoding = (resp.headers.get("Content-Encoding") or "").lower()
        if encoding == "gzip":
            return gzip.decompress(data)
        if encoding == "deflate":
            return zlib.decompress(data)
        return data


def _load_json(url: str, user_agent: str, ssl_context: Optional[ssl.SSLContext]) -> dict:
    return json.loads(_http_get(url, user_agent, ssl_context))


def _normalize_ticker(ticker: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", ticker.upper())


def _ticker_to_cik(ticker: str, user_agent: str, ssl_context: Optional[ssl.SSLContext]) -> str:
    data = _load_json(SEC_TICKER_URL, user_agent, ssl_context)
    norm = _normalize_ticker(ticker)
    for _, entry in data.items():
        if _normalize_ticker(entry.get("ticker", "")) == norm:
            cik_num = int(entry["cik_str"])
            return f"{cik_num:010d}"
    raise ValueError(f"Ticker not found: {ticker}")


def _is_earnings_related(items: str) -> bool:
    items_norm = items.replace(" ", "")
    return "2.02" in items_norm


def _matches_exhibit(name: str, ex_code: str) -> bool:
    name_norm = re.sub(r"[^a-z0-9]", "", name.lower())
    ex_code_norm = re.sub(r"[^a-z0-9]", "", ex_code.lower())
    patterns = [
        ex_code_norm,
        ex_code_norm.replace("ex", "exhibit"),
    ]
    return any(p in name_norm for p in patterns)


def _infer_exhibit_code_from_name(name: str) -> Optional[str]:
    base = os.path.splitext(name)[0].lower()
    match = re.search(r"(?:ex|exhibit)[-_]?99[\\._-]?([12])", base)
    if match:
        return f"EX-99.{match.group(1)}"
    digits_only = re.sub(r"[^0-9]", "", base)
    match = re.search(r"(99[12])$", digits_only)
    if match:
        return f"EX-99.{match.group(1)[-1]}"
    return None


def _is_html_doc(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in {".htm", ".html"}


def _exhibit_score(name: str) -> int:
    ext = os.path.splitext(name)[1].lower()
    if ext in {".htm", ".html"}:
        return 4
    if ext == ".pdf":
        return 3
    if ext == ".txt":
        return 2
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
        return 1
    return 0


def _find_exhibit_files(index_json: dict, primary_document: Optional[str] = None) -> dict:
    found = {"EX-99.1": None, "EX-99.2": None}
    scores = {"EX-99.1": -1, "EX-99.2": -1}
    items = index_json.get("directory", {}).get("item", [])
    for item in items:
        file_type = (item.get("type") or "").upper()
        name = item.get("name", "")
        if not name:
            continue
        for ex_code in ("EX-99.1", "EX-99.2"):
            if file_type == ex_code or _matches_exhibit(name, ex_code):
                sc = _exhibit_score(name)
                if sc > scores[ex_code]:
                    found[ex_code] = name
                    scores[ex_code] = sc
        inferred = _infer_exhibit_code_from_name(name)
        if inferred in {"EX-99.1", "EX-99.2"}:
            sc = _exhibit_score(name)
            if sc > scores[inferred]:
                found[inferred] = name
                scores[inferred] = sc
        if not found["EX-99.1"] or not found["EX-99.2"]:
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            match = re.search(r"ex99([12])", normalized)
            if match:
                ex_code = f"EX-99.{match.group(1)}"
                if not found[ex_code]:
                    sc = _exhibit_score(name)
                    if sc > scores[ex_code]:
                        found[ex_code] = name
                        scores[ex_code] = sc

    # Fallback: some filings omit explicit EX-99 labels. Heuristically pick HTML docs besides the primary.
    if not found["EX-99.1"] or not found["EX-99.2"]:
        candidates = []
        for item in items:
            name = item.get("name", "") or ""
            if not _is_html_doc(name):
                continue
            name_lower = name.lower()
            if primary_document and name_lower == primary_document.lower():
                continue
            if any(token in name_lower for token in ["xsd", "xml", "xbrl", "json", "js", "css", "schema", "cal", "lab", "pre", "def", "summary"]):
                continue
            candidates.append(name)
        if candidates:
            # Score candidates to prefer earnings press releases/presentations.
            def score(n: str) -> int:
                s = 0
                nl = n.lower()
                if "earnings" in nl or "release" in nl or "press" in nl:
                    s += 3
                if "presentation" in nl or "slides" in nl or "deck" in nl:
                    s += 2
                if "99" in re.sub(r"[^0-9]", "", nl):
                    s += 2
                return s

            candidates.sort(key=lambda n: (score(n), n), reverse=True)
            if not found["EX-99.1"]:
                found["EX-99.1"] = candidates[0]
            if not found["EX-99.2"] and len(candidates) > 1:
                found["EX-99.2"] = candidates[1]
    return found


def _debug_print_exhibit_mapping(index_json: dict, exhibits: dict) -> None:
    print("Debug: exhibit mapping")
    print(f"  EX-99.1 -> {exhibits.get('EX-99.1')}")
    print(f"  EX-99.2 -> {exhibits.get('EX-99.2')}")
    items = index_json.get("directory", {}).get("item", [])
    print("Debug: index items with possible exhibit hints:")
    for item in items:
        name = item.get("name", "")
        file_type = (item.get("type") or "").upper()
        if not name:
            continue
        inferred = _infer_exhibit_code_from_name(name)
        if file_type.startswith("EX-99") or inferred or "99" in name.lower():
            print(f"  type={file_type or '-'} name={name} inferred={inferred or '-'}")


def _filing_candidates(submissions: dict, date_filter: Optional[str]) -> Iterable[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    items = recent.get("items", [])
    primary_docs = recent.get("primaryDocument", [])

    for idx, form in enumerate(forms):
        if form != "8-K":
            continue
        filing_date = filing_dates[idx] if idx < len(filing_dates) else None
        if date_filter and filing_date != date_filter:
            continue
        items_str = items[idx] if idx < len(items) else ""
        if not _is_earnings_related(items_str):
            continue
        accession = accession_numbers[idx] if idx < len(accession_numbers) else None
        if not accession:
            continue
        yield {
            "filing_date": filing_date,
            "items": items_str,
            "accession": accession,
            "primary_document": primary_docs[idx] if idx < len(primary_docs) else None,
        }


def _find_prior_report(submissions: dict, before_date: Optional[str]) -> Optional[dict]:
    if not before_date:
        return None
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    best: Optional[dict] = None
    for idx, form in enumerate(forms):
        if form not in {"10-Q", "10-K"}:
            continue
        filing_date = filing_dates[idx] if idx < len(filing_dates) else None
        if not filing_date or filing_date >= before_date:
            continue
        accession = accession_numbers[idx] if idx < len(accession_numbers) else None
        if not accession:
            continue
        primary_doc = primary_docs[idx] if idx < len(primary_docs) else None
        if not primary_doc:
            continue
        if not best or filing_date > best["filing_date"]:
            best = {
                "form": form,
                "filing_date": filing_date,
                "accession": accession,
                "primary_document": primary_doc,
            }
    return best


def _date_or_none(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Date must be YYYY-MM-DD") from exc
    return date_str


def _download_file(
    url: str,
    dest_path: str,
    user_agent: str,
    ssl_context: Optional[ssl.SSLContext],
) -> None:
    data = _http_get(url, user_agent, ssl_context)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)


def _acceptance_datetime(accession: str, base_url: str, user_agent: str, ssl_context: Optional[ssl.SSLContext]) -> Optional[str]:
    url = f"{base_url}/{accession}.txt"
    try:
        data = _http_get(url, user_agent, ssl_context)
    except urllib.error.HTTPError:
        return None
    text = data.decode("utf-8", errors="replace")
    match = re.search(r"ACCEPTANCE-DATETIME[>:]\s*(\d{14})", text)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{raw[8:10]}:{raw[10:12]}:{raw[12:14]}"


class _ImageSrcParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "img":
            return
        for key, value in attrs:
            if key.lower() == "src" and value:
                self.srcs.append(value)


def _rewrite_html_image_srcs(html_path: str, replacements: dict[str, str]) -> None:
    if not replacements:
        return
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    for original, new_value in replacements.items():
        content = content.replace(original, new_value)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def _download_linked_images(
    html_path: str,
    base_url: str,
    out_dir: str,
    user_agent: str,
    ssl_context: Optional[ssl.SSLContext],
) -> None:
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    parser = _ImageSrcParser()
    parser.feed(content)
    replacements: dict[str, str] = {}
    images_dir = os.path.join(out_dir, "img")
    os.makedirs(images_dir, exist_ok=True)
    for src in parser.srcs:
        if src.startswith("data:"):
            continue
        full_url = urllib.parse.urljoin(f"{base_url}/", src)
        parsed = urllib.parse.urlparse(full_url)
        filename = os.path.basename(parsed.path)
        if not filename:
            continue
        dest_path = os.path.join(images_dir, filename)
        if os.path.exists(dest_path):
            replacements[src] = f"img/{filename}"
            continue
        _download_file(full_url, dest_path, user_agent, ssl_context)
        replacements[src] = f"img/{filename}"
        time.sleep(0.1)
    _rewrite_html_image_srcs(html_path, replacements)






def _convert_html_to_pdf(
    html_path: str,
    pdf_path: str,
    extra_args: Optional[list[str]] = None,
    allow_failure_if_output: bool = False,
) -> None:
    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if not wkhtmltopdf:
        raise RuntimeError("wkhtmltopdf not found in PATH")
    cmd = [wkhtmltopdf, "--quiet", "--enable-local-file-access"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([html_path, pdf_path])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if allow_failure_if_output and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            print(f"wkhtmltopdf exited with code {result.returncode}; keeping generated PDF.")
            return
        raise RuntimeError(f"wkhtmltopdf failed with exit code {result.returncode}")


def _validate_quarter_inputs(q: Optional[int], fy: Optional[int]) -> Optional[tuple[int, int]]:
    if q is None and fy is None:
        return None
    if q is None or fy is None:
        raise ValueError("Both --q and --fy must be provided together.")
    if q not in {1, 2, 3, 4}:
        raise ValueError("--q must be 1, 2, 3, or 4.")
    if fy < 1900:
        raise ValueError("--fy must be a 4-digit year.")
    return q, fy


def _folder_name(ticker: str, q: Optional[int], fy: Optional[int]) -> str:
    if q is None or fy is None:
        return _normalize_ticker(ticker)
    return f"{_normalize_ticker(ticker)}_Q{q}_{fy}"


def _file_prefix(ticker: str, q: Optional[int], fy: Optional[int]) -> str:
    ticker_norm = _normalize_ticker(ticker).lower()
    if q is None or fy is None:
        return f"{ticker_norm}_"
    return f"{ticker_norm}_q{q}_{fy}_"


def _prior_quarter_label(q: int, fy: int) -> tuple[int, int, str]:
    if q == 1:
        prev_q = 4
        prev_fy = fy - 1
    else:
        prev_q = q - 1
        prev_fy = fy
    label = "10k" if prev_q == 4 else "10q"
    return prev_q, prev_fy, label


def _trim_pdf_pages(pdf_path: str, trim_first: int, trim_last: int) -> bool:
    if trim_first < 0 or trim_last < 0:
        return False
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        except Exception:
            print("PDF trim skipped: install 'pypdf' to remove the first/last pages.")
            return False

    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    if total <= (trim_first + trim_last):
        print(f"PDF trim skipped: transcript has {total} pages.")
        return False
    start_idx = trim_first
    end_idx = total - trim_last
    writer = PdfWriter()
    for i in range(start_idx, end_idx):
        writer.add_page(reader.pages[i])
    tmp_path = pdf_path + ".tmp"
    with open(tmp_path, "wb") as f:
        writer.write(f)
    os.replace(tmp_path, pdf_path)
    return True


def _build_transcript_query(ticker: str, q: Optional[int], fy: Optional[int]) -> str:
    parts = [ticker, "earnings call transcript"]
    if q is not None and fy is not None:
        parts.append(f"Q{q} FY{fy}")
    parts.append("investing.com")
    return " ".join(parts)


class _DDGResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        href = None
        class_attr = ""
        for key, value in attrs:
            if key.lower() == "href":
                href = value
            elif key.lower() == "class" and value:
                class_attr = value
        if href and "result__a" in class_attr:
            self.urls.append(href)


def _extract_ddg_result_urls(html: str) -> list[str]:
    parser = _DDGResultParser()
    parser.feed(html)
    return parser.urls


def _extract_bing_result_urls(html: str) -> list[str]:
    urls: list[str] = []
    for block in re.findall(r'<li[^>]+class="[^"]*b_algo[^"]*"[^>]*>.*?</li>', html, re.I | re.S):
        match = re.search(r'<h2>\s*<a[^>]+href="([^"]+)"', block, re.I | re.S)
        if match:
            urls.append(match.group(1))
    return urls


def _normalize_ddg_url(url: str) -> str:
    url = html_lib.unescape(url)
    if url.startswith("/l/?") or "duckduckgo.com/l/?" in url:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return qs["uddg"][0]
    return url


def _normalize_bing_url(url: str) -> str:
    return html_lib.unescape(url)


def _is_investing_transcript_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc or "investing.com" not in parsed.netloc:
        return False
    return "/news/transcripts/" in parsed.path


def _looks_like_ddg_block_page(html: str) -> bool:
    lower = html.lower()
    markers = [
        "bots use duckduckgo too",
        "anomaly-modal",
        "please complete the following challenge",
        "anomaly.js",
    ]
    return any(marker in lower for marker in markers)


def _looks_like_bing_block_page(html: str) -> bool:
    lower = html.lower()
    markers = [
        "our systems have detected unusual traffic",
        "verify you are a human",
        "captcha",
        "bing.com/challenge",
    ]
    return any(marker in lower for marker in markers)


def _run_search_attempt(
    *,
    engine: str,
    search_url: str,
    query: str,
    ssl_context: Optional[ssl.SSLContext],
    result_extractor,
    url_normalizer,
    block_detector,
    no_results_markers: list[str],
) -> dict:
    attempt: dict[str, object] = {
        "engine": engine,
        "query": query,
        "search_url": search_url,
        "status": "network_error",
        "detail": "",
        "transcript_url": None,
        "raw_result_count": 0,
    }
    try:
        html = _http_get(search_url, SEARCH_USER_AGENT, ssl_context).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        attempt["status"] = "network_error"
        attempt["detail"] = f"HTTP {exc.code}"
        return attempt
    except urllib.error.URLError as exc:
        attempt["status"] = "network_error"
        attempt["detail"] = str(exc.reason)
        return attempt
    except Exception as exc:
        attempt["status"] = "network_error"
        attempt["detail"] = str(exc)
        return attempt

    if block_detector(html):
        attempt["status"] = "bot_blocked"
        attempt["detail"] = "Challenge/anti-bot page detected."
        return attempt

    raw_urls = result_extractor(html)
    attempt["raw_result_count"] = len(raw_urls)
    normalized_urls = [url_normalizer(u) for u in raw_urls]

    for normalized in normalized_urls:
        if _is_investing_transcript_url(normalized):
            attempt["status"] = "found"
            attempt["transcript_url"] = normalized
            attempt["detail"] = "Found Investing.com transcript URL."
            return attempt

    if normalized_urls:
        attempt["status"] = "no_relevant_result"
        attempt["detail"] = (
            f"Parsed {len(normalized_urls)} results but none matched an Investing.com transcript URL."
        )
        return attempt

    lower = html.lower()
    if any(marker in lower for marker in no_results_markers):
        attempt["status"] = "no_relevant_result"
        attempt["detail"] = "Search page loaded but reported no matching results."
    else:
        attempt["status"] = "parse_failed"
        attempt["detail"] = "Search page did not expose parseable result links."
    return attempt


def _log_search_attempt(attempt: dict, debug: bool) -> None:
    engine = str(attempt.get("engine"))
    status = str(attempt.get("status"))
    detail = str(attempt.get("detail") or "")
    if status == "found":
        print(f"Transcript search ({engine}): found candidate URL.")
        if debug:
            print(f"  URL: {attempt.get('transcript_url')}")
        return
    if status == "no_relevant_result":
        print(f"Transcript search ({engine}): results loaded, but no Investing.com transcript URL was found.")
    elif status == "bot_blocked":
        print(f"Transcript search ({engine}): blocked by anti-bot challenge.")
    elif status == "parse_failed":
        print(f"Transcript search ({engine}): could not parse usable search results.")
    else:
        print(f"Transcript search ({engine}): request failed.")
    if detail and (debug or status in {"bot_blocked", "network_error"}):
        print(f"  Detail: {detail}")
    if debug:
        print(f"  Query: {attempt.get('query')}")
        print(f"  Search URL: {attempt.get('search_url')}")
        print(f"  Raw results parsed: {attempt.get('raw_result_count')}")


def _format_transcript_search_failure(query: str, attempts: list[dict]) -> str:
    lines = [
        "Transcript search failed after trying multiple engines.",
        f"Query: {query}",
    ]
    for attempt in attempts:
        lines.append(
            f"- {attempt.get('engine')}: {attempt.get('status')} ({attempt.get('detail')})"
        )
    lines.append("Try again later, pass --transcript-url, or run with --debug for more detail.")
    return "\n".join(lines)


def _search_investing_transcript_url(
    ticker: str,
    q: Optional[int],
    fy: Optional[int],
    ssl_context: Optional[ssl.SSLContext],
    debug: bool = False,
) -> str:
    query = _build_transcript_query(ticker, q, fy)
    encoded_query = urllib.parse.quote_plus(query)
    attempts: list[dict] = []

    ddg_attempt = _run_search_attempt(
        engine="DuckDuckGo",
        search_url=DUCKDUCKGO_HTML_SEARCH.format(query=encoded_query),
        query=query,
        ssl_context=ssl_context,
        result_extractor=_extract_ddg_result_urls,
        url_normalizer=_normalize_ddg_url,
        block_detector=_looks_like_ddg_block_page,
        no_results_markers=["no results found", "no more results"],
    )
    attempts.append(ddg_attempt)
    _log_search_attempt(ddg_attempt, debug)
    if ddg_attempt.get("status") == "found":
        return str(ddg_attempt["transcript_url"])

    print("Transcript search: trying fallback engine (Bing).")
    bing_attempt = _run_search_attempt(
        engine="Bing",
        search_url=BING_HTML_SEARCH.format(query=encoded_query),
        query=query,
        ssl_context=ssl_context,
        result_extractor=_extract_bing_result_urls,
        url_normalizer=_normalize_bing_url,
        block_detector=_looks_like_bing_block_page,
        no_results_markers=["there are no results for", "did not match any documents"],
    )
    attempts.append(bing_attempt)
    _log_search_attempt(bing_attempt, debug)
    if bing_attempt.get("status") == "found":
        return str(bing_attempt["transcript_url"])

    raise TranscriptSearchError(_format_transcript_search_failure(query, attempts))


def _read_cookie_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cookie = f.read().strip()
    except FileNotFoundError:
        return None
    if not cookie:
        return None
    return cookie


def _write_cookie_file(path: str, cookie: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(cookie.strip())


def _prompt_for_cookie() -> Optional[str]:
    help_text = (
        "To get the Investing.com cookie from your browser:\n"
        "  1) Open the transcript page in your browser.\n"
        "  2) Open DevTools (F12) and go to the Network tab.\n"
        "  3) Refresh the page, click the main document request.\n"
        "  4) In Request Headers, copy the full 'Cookie' value.\n"
        "You can pass it once via --transcript-cookie or paste it here.\n"
    )
    if not sys.stdin.isatty():
        print(help_text)
        print("Non-interactive session: pass --transcript-cookie or use --transcript-cookie-file.")
        return None
    print(help_text)
    try:
        cookie = input("Enter Investing.com cookie (leave blank to skip): ").strip()
    except EOFError:
        return None
    return cookie or None


def _looks_like_auth_wall(html: str) -> bool:
    lower = html.lower()
    if "transcript" in lower:
        return False
    if "sign in" in lower or "log in" in lower or "subscribe" in lower:
        return True
    return False


def _download_investing_transcript(
    ticker: str,
    q: Optional[int],
    fy: Optional[int],
    out_base: str,
    user_agent: str,
    ssl_context: Optional[ssl.SSLContext],
    transcript_url: Optional[str],
    cookie_file: str,
    cookie_value: Optional[str],
    save_pdf: bool,
    debug: bool,
) -> bool:
    url = transcript_url
    if not url:
        url = _search_investing_transcript_url(
            ticker=ticker,
            q=q,
            fy=fy,
            ssl_context=ssl_context,
            debug=debug,
        )

    cookie = cookie_value
    if cookie:
        _write_cookie_file(cookie_file, cookie)
    if not cookie:
        cookie = _read_cookie_file(cookie_file)
    if not cookie:
        cookie = _prompt_for_cookie()
        if cookie:
            _write_cookie_file(cookie_file, cookie)
    if not cookie:
        print("Transcript download skipped (no cookie provided).")
        return False

    last_error: Optional[str] = None
    for attempt in range(2):
        try:
            html_bytes = _http_get(
                url,
                user_agent,
                ssl_context,
                extra_headers={
                    "Cookie": cookie,
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                last_error = f"HTTP {exc.code}"
            else:
                print(f"Transcript download failed: HTTP {exc.code}")
                return False
        else:
            html_text = html_bytes.decode("utf-8", errors="replace")
            if not _looks_like_auth_wall(html_text):
                file_prefix = _file_prefix(ticker, q, fy)
                html_path = os.path.join(out_base, f"{file_prefix}{DEFAULT_TRANSCRIPT_STEM}.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_text)
                if save_pdf:
                    pdf_path = os.path.splitext(html_path)[0] + ".pdf"
                    try:
                        _convert_html_to_pdf(
                            html_path,
                            pdf_path,
                            extra_args=[
                                "--load-error-handling",
                                "ignore",
                                "--load-media-error-handling",
                                "ignore",
                            ],
                            allow_failure_if_output=True,
                        )
                        try:
                            _trim_pdf_pages(pdf_path, trim_first=1, trim_last=2)
                        except Exception as exc:
                            print(f"Transcript PDF trim failed: {exc}")
                        print(f"Transcript saved: {pdf_path}")
                        try:
                            os.remove(html_path)
                        except FileNotFoundError:
                            pass
                        except OSError:
                            print(f"Could not remove transcript HTML: {html_path}")
                    except Exception as exc:
                        print(f"Transcript PDF conversion failed: {exc}")
                        print(f"Transcript HTML retained: {html_path}")
                else:
                    print(f"Transcript saved: {html_path}")
                return True
            last_error = "auth wall detected"

        if attempt == 0:
            print(f"Transcript access failed ({last_error}). Please refresh cookie.")
            cookie = _prompt_for_cookie()
            if cookie:
                _write_cookie_file(cookie_file, cookie)
                continue
        break

    print("Transcript download failed; cookie may need refresh.")
    return False


def fetch_latest_earnings_8k(
    ticker: str,
    date_filter: Optional[str],
    outdir: str,
    user_agent: str,
    ssl_context: Optional[ssl.SSLContext],
    save_pdf: bool,
    q: Optional[int] = None,
    fy: Optional[int] = None,
    transcript: bool = False,
    transcript_cookie_file: Optional[str] = None,
    transcript_cookie: Optional[str] = None,
    transcript_url: Optional[str] = None,
    debug: bool = False,
) -> int:
    cik = _ticker_to_cik(ticker, user_agent, ssl_context)
    submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
    submissions = _load_json(submissions_url, user_agent, ssl_context)

    candidates = list(_filing_candidates(submissions, date_filter))
    if not candidates:
        print("8-K not available for the requested date or earnings criteria.")
        return 1

    candidates.sort(key=lambda x: x["filing_date"] or "", reverse=True)

    for candidate in candidates:
        accession = candidate["accession"]
        accession_nodash = accession.replace("-", "")
        index_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_nodash}/index.json"
        try:
            index_json = _load_json(index_url, user_agent, ssl_context)
        except urllib.error.HTTPError:
            continue

        exhibits = _find_exhibit_files(index_json, primary_document=candidate.get("primary_document"))
        if debug:
            _debug_print_exhibit_mapping(index_json, exhibits)
        if not exhibits["EX-99.1"] and not exhibits["EX-99.2"]:
            continue

        saved = []
        ex99_1_paths: list[str] = []
        html_paths: list[str] = []
        base_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_nodash}"
        out_base = os.path.join(outdir, _folder_name(ticker, q, fy))
        file_prefix = _file_prefix(ticker, q, fy)
        os.makedirs(out_base, exist_ok=True)
        sec_filing_dt = _acceptance_datetime(accession, base_url, user_agent, ssl_context)
        meta_path = os.path.join(out_base, "sec_filing_info.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ticker": _normalize_ticker(ticker),
                    "q": q,
                    "fy": fy,
                    "sec_filing_datetime": sec_filing_dt,
                },
                f,
                indent=2,
            )

        for ex_code, filename in exhibits.items():
            if not filename:
                continue
            url = f"{base_url}/{filename}"
            ext = os.path.splitext(filename)[1] or ".htm"
            ex_suffix = "991" if ex_code == "EX-99.1" else "992"
            dest_path = os.path.join(out_base, f"{file_prefix}8k_{ex_suffix}{ext}")
            _download_file(url, dest_path, user_agent, ssl_context)
            saved.append(dest_path)
            if ex_code == "EX-99.1":
                ex99_1_paths.append(dest_path)
            if os.path.splitext(dest_path)[1].lower() in {".htm", ".html"}:
                html_paths.append(dest_path)
                _download_linked_images(
                    dest_path,
                    base_url=base_url,
                    out_dir=out_base,
                    user_agent=user_agent,
                    ssl_context=ssl_context,
                )
                if save_pdf and os.path.splitext(dest_path)[1].lower() in {".htm", ".html"}:
                    pdf_path = os.path.splitext(dest_path)[0] + ".pdf"
                    try:
                        _convert_html_to_pdf(dest_path, pdf_path)
                        saved.append(pdf_path)
                    except Exception as exc:
                        print(f"PDF conversion failed for {dest_path}: {exc}")
            time.sleep(0.2)

        prior_report = _find_prior_report(submissions, candidate["filing_date"])
        if prior_report:
            report_accession = prior_report["accession"]
            report_accession_nodash = report_accession.replace("-", "")
            report_base_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{report_accession_nodash}"
            report_doc = prior_report["primary_document"]
            report_ext = os.path.splitext(report_doc)[1] or ".htm"
            report_label = prior_report["form"].replace("-", "").lower()
            report_prefix = file_prefix
            if q is not None and fy is not None:
                prev_q, prev_fy, report_label = _prior_quarter_label(q, fy)
                report_prefix = _file_prefix(ticker, prev_q, prev_fy)
            report_path = os.path.join(out_base, f"{report_prefix}{report_label}{report_ext}")
            report_url = f"{report_base_url}/{report_doc}"
            _download_file(report_url, report_path, user_agent, ssl_context)
            saved.append(report_path)
            if os.path.splitext(report_path)[1].lower() in {".htm", ".html"}:
                html_paths.append(report_path)
                _download_linked_images(
                    report_path,
                    base_url=report_base_url,
                    out_dir=out_base,
                    user_agent=user_agent,
                    ssl_context=ssl_context,
                )
                if save_pdf:
                    pdf_path = os.path.splitext(report_path)[0] + ".pdf"
                    try:
                        _convert_html_to_pdf(report_path, pdf_path)
                        saved.append(pdf_path)
                    except Exception as exc:
                        print(f"PDF conversion failed for {report_path}: {exc}")

        if save_pdf and html_paths:
            retained_html = []
            for html_path in html_paths:
                pdf_path = os.path.splitext(html_path)[0] + ".pdf"
                if not os.path.exists(pdf_path):
                    retained_html.append(html_path)
                    continue
                try:
                    os.remove(html_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    retained_html.append(html_path)
                if html_path in saved:
                    saved.remove(html_path)
            if not retained_html:
                images_dir = os.path.join(out_base, "img")
                if os.path.isdir(images_dir):
                    shutil.rmtree(images_dir, ignore_errors=True)

        # Quarter date extraction removed; keep downloads only.

        if saved:
            print("Downloaded:")
            for path in saved:
                print(f"  {path}")
            missing = [k for k, v in exhibits.items() if v is None]
            if missing:
                print("Missing exhibits:", ", ".join(missing))
            if transcript:
                cookie_file = transcript_cookie_file or DEFAULT_TRANSCRIPT_COOKIE_FILE
                try:
                    _download_investing_transcript(
                        ticker=ticker,
                        q=q,
                        fy=fy,
                        out_base=out_base,
                        user_agent=user_agent,
                        ssl_context=ssl_context,
                        transcript_url=transcript_url,
                        cookie_file=cookie_file,
                        cookie_value=transcript_cookie,
                        save_pdf=save_pdf,
                        debug=debug,
                    )
                except TranscriptSearchError as exc:
                    print(str(exc))
                    return 3
                except Exception as exc:
                    print(f"Transcript download error: {exc}")
                    return 3
            return 0

    print("8-K not available for the requested date or earnings criteria.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the latest earnings-related 8-K exhibits (EX-99.1 / EX-99.2)."
    )
    parser.add_argument("--ticker", required=True, help="Company ticker, e.g. COST")
    parser.add_argument(
        "--date",
        help="Optional filing date filter (YYYY-MM-DD).",
        default=None,
    )
    parser.add_argument(
        "--q",
        type=int,
        help="Optional fiscal quarter (1-4). Must be used with --fy.",
        default=None,
    )
    parser.add_argument(
        "--fy",
        type=int,
        help="Optional fiscal year (YYYY). Must be used with --q.",
        default=None,
    )
    parser.add_argument(
        "--outdir",
        help=f"Directory to save files (default: {DEFAULT_OUTDIR}).",
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--user-agent",
        help="SEC requires a User-Agent with contact info, e.g. 'Your Name you@email.com'.",
        default="CodexSecFetcher/1.0 (youremail@example.com)",
    )
    parser.add_argument(
        "--ca-bundle",
        help="Path to a CA bundle (PEM). Use if your system certs are missing.",
        default=None,
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (not recommended).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save HTML exhibits as PDF (requires wkhtmltopdf in PATH).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info about exhibit selection and index items.",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Also download the Investing.com earnings call transcript as PDF.",
    )
    parser.add_argument(
        "--transcript-cookie",
        default=None,
        help="Investing.com cookie string (optional; saved for future use).",
    )
    parser.add_argument(
        "--transcript-cookie-file",
        default=None,
        help=f"Path to store Investing.com cookie (default: {DEFAULT_TRANSCRIPT_COOKIE_FILE}).",
    )
    parser.add_argument(
        "--transcript-url",
        default=None,
        help="Optional direct transcript URL to skip search.",
    )
    args = parser.parse_args()

    date_filter = _date_or_none(args.date)
    quarter_info = _validate_quarter_inputs(args.q, args.fy)
    q = quarter_info[0] if quarter_info else None
    fy = quarter_info[1] if quarter_info else None

    ssl_context: Optional[ssl.SSLContext]
    if args.insecure:
        ssl_context = ssl._create_unverified_context()
    else:
        ssl_context = ssl.create_default_context(cafile=args.ca_bundle)

    try:
        return fetch_latest_earnings_8k(
            ticker=args.ticker,
            date_filter=date_filter,
            outdir=args.outdir,
            user_agent=args.user_agent,
            ssl_context=ssl_context,
            save_pdf=args.pdf,
            q=q,
            fy=fy,
            transcript=args.transcript,
            transcript_cookie_file=args.transcript_cookie_file,
            transcript_cookie=args.transcript_cookie,
            transcript_url=args.transcript_url,
            debug=args.debug,
        )
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
            print("SSL verification failed. Try one of:")
            print("  - Provide a CA bundle with --ca-bundle /path/to/cacert.pem")
            print("  - Use --insecure to disable verification (not recommended)")
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
