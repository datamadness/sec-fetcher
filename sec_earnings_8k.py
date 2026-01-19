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

DEFAULT_OUTDIR = "./sec_earnings_8k"


def _http_get(url: str, user_agent: str, ssl_context: Optional[ssl.SSLContext]) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
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
    name_norm = name.lower()
    ex_code_norm = ex_code.lower().replace(".", "")
    patterns = [
        ex_code_norm,
        ex_code_norm.replace("ex", "ex-"),
        ex_code_norm.replace("ex", "ex_"),
        ex_code_norm.replace("ex", "exhibit"),
    ]
    return any(p in name_norm.replace(".", "").replace("-", "").replace("_", "") for p in patterns)


def _find_exhibit_files(index_json: dict) -> dict:
    found = {"EX-99.1": None, "EX-99.2": None}
    items = index_json.get("directory", {}).get("item", [])
    for item in items:
        file_type = (item.get("type") or "").upper()
        name = item.get("name", "")
        if not name:
            continue
        if file_type == "EX-99.1" or _matches_exhibit(name, "EX-99.1"):
            found["EX-99.1"] = name
        if file_type == "EX-99.2" or _matches_exhibit(name, "EX-99.2"):
            found["EX-99.2"] = name
        if not found["EX-99.1"] or not found["EX-99.2"]:
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            match = re.search(r"ex99([12])", normalized)
            if match:
                ex_code = f"EX-99.{match.group(1)}"
                if not found[ex_code]:
                    found[ex_code] = name
    return found


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






def _convert_html_to_pdf(html_path: str, pdf_path: str) -> None:
    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if not wkhtmltopdf:
        raise RuntimeError("wkhtmltopdf not found in PATH")
    subprocess.run(
        [wkhtmltopdf, "--quiet", "--enable-local-file-access", html_path, pdf_path],
        check=True,
    )


def fetch_latest_earnings_8k(
    ticker: str,
    date_filter: Optional[str],
    outdir: str,
    user_agent: str,
    ssl_context: Optional[ssl.SSLContext],
    save_pdf: bool,
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

        exhibits = _find_exhibit_files(index_json)
        if not exhibits["EX-99.1"] and not exhibits["EX-99.2"]:
            continue

        saved = []
        ex99_1_paths: list[str] = []
        base_url = f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_nodash}"
        safe_ticker = _normalize_ticker(ticker)
        out_base = os.path.join(outdir, f"{safe_ticker}_{candidate['filing_date']}_{accession}")

        for ex_code, filename in exhibits.items():
            if not filename:
                continue
            url = f"{base_url}/{filename}"
            dest_path = os.path.join(out_base, f"{ex_code}_{filename}")
            _download_file(url, dest_path, user_agent, ssl_context)
            saved.append(dest_path)
            if ex_code == "EX-99.1":
                ex99_1_paths.append(dest_path)
            if os.path.splitext(dest_path)[1].lower() in {".htm", ".html"}:
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
            report_form = prior_report["form"].replace("-", "")
            report_name = f"PRIOR_{report_form}_{prior_report['filing_date']}_{report_doc}"
            report_path = os.path.join(out_base, report_name)
            report_url = f"{report_base_url}/{report_doc}"
            _download_file(report_url, report_path, user_agent, ssl_context)
            saved.append(report_path)
            if os.path.splitext(report_path)[1].lower() in {".htm", ".html"}:
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

        # Quarter date extraction removed; keep downloads only.

        if saved:
            print("Downloaded:")
            for path in saved:
                print(f"  {path}")
            missing = [k for k, v in exhibits.items() if v is None]
            if missing:
                print("Missing exhibits:", ", ".join(missing))
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
    args = parser.parse_args()

    date_filter = _date_or_none(args.date)

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
