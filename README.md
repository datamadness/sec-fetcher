# SEC Earnings 8-K Fetcher

Fetch the latest earnings-related 8-K exhibits (EX-99.1 / EX-99.2) from SEC EDGAR,
optionally filtered by filing date. The tool saves the exhibit HTML files and any
linked images (e.g., slideshow-style EX-99.2) so the documents render correctly
offline. PDF conversion is optional.

## Purpose

- Download the most recent earnings-related 8-K (Item 2.02) for a ticker.
- Also download the most recent 10-Q or 10-K filed before that 8-K.
- Optionally filter by a specific filing date (`YYYY-MM-DD`).
- Save EX-99.1 / EX-99.2 exhibits, plus linked image assets.
- Optionally convert HTML exhibits to PDF.
- Optionally auto-search Investing.com and save the earnings call transcript as PDF.

## Install dependencies

SEC downloads use the Python standard library. Reliable Investing.com transcript
retrieval uses Playwright as a browser fallback:
```bash
pip install -r requirements.txt
playwright install chromium
```

Optional PDF conversion requires `wkhtmltopdf`:
- Install `wkhtmltopdf` and ensure it is available in your `PATH`.

Transcript PDF trimming (remove first page + last 2 pages) requires `pypdf`:
- `pip install pypdf`

Examples:
```bash
# Verify wkhtmltopdf is available
wkhtmltopdf --version
```

## Windows installation

1. Install Python 3.10+ (from https://www.python.org/downloads/windows/) and ensure `python` is on your PATH.
2. (Optional) Install `wkhtmltopdf` if you want PDF output. Add it to PATH so `wkhtmltopdf --version` works in PowerShell.
3. (Optional) Install `pypdf` if you want transcript page trimming.
4. For transcript browser fallback, run `pip install playwright` followed by `playwright install chromium`.
5. On Windows, use `python` instead of `python3` in the CLI examples below.

## Usage

```bash
python3 sec_earnings_8k.py --ticker COST
```

## Flags

- `--ticker` (required): Company ticker symbol, e.g. `COST`.
- `--date` (optional): Filing date filter in `YYYY-MM-DD`.
- `--q` (optional): Fiscal quarter (1-4). Must be used with `--fy`; both are required with `--transcript`.
- `--fy` (optional): Fiscal year (YYYY). Must be used with `--q`; both are required with `--transcript`.
- `--outdir` (optional): Output directory (default: `./sec_earnings_8k`).
- `--user-agent` (optional): SEC requires a User-Agent with contact info.
- `--ca-bundle` (optional): Path to a CA bundle (PEM) if your system certs are missing.
- `--insecure` (optional): Disable TLS verification (not recommended).
- `--pdf` (optional): Also save HTML exhibits as PDF (requires `wkhtmltopdf`). When PDF conversion succeeds, HTML files and the `img/` folder are removed.
- `--debug` (optional): Print exhibit selection details and transcript search diagnostics.
- `--transcript` (optional): Discover and download the exact Investing.com earnings call transcript.
- Transcript PDFs can be trimmed with `--transcript-trim-first` and `--transcript-trim-last` (requires `pypdf`).
- `--transcript-cookie` (optional): Investing.com cookie string. Saved to the cookie file for reuse.
- `--transcript-cookie-file` (optional): Override path for Investing.com cookie file (default: `.secrets/investing.cookie.txt`).
- `--transcript-url` (optional): Direct transcript URL to skip search.

## Examples

Download latest earnings-related 8-K exhibits:
```bash
python3 sec_earnings_8k.py --ticker COST
```

Include fiscal quarter/year for naming:
```bash
python3 sec_earnings_8k.py --ticker COST --q 4 --fy 2025
```

Filter by a specific filing date:
```bash
python3 sec_earnings_8k.py --ticker COST --date 2025-12-11
```

Save to a custom folder:
```bash
python3 sec_earnings_8k.py --ticker COST --outdir ./downloads
```

Download and also create PDFs:
```bash
python3 sec_earnings_8k.py --ticker COST --pdf
```

Download SEC files and transcript (auto-search Investing.com):
```bash
python3 sec_earnings_8k.py --ticker COST --q 1 --fy 2026 --transcript
```

Provide/update cookie manually (saved for future runs):
```bash
python3 sec_earnings_8k.py --ticker COST --q 1 --fy 2026 --transcript --transcript-cookie "YOUR_COOKIE_STRING"
```

## Earnings call transcript guide

When `--transcript` is used, the tool:
- Searches Google News RSS for Investing.com transcript candidates using the SEC company name and exact fiscal period. It does not scrape DuckDuckGo.
- Requires the Investing.com `Full transcript - Company (TICKER) Q# YYYY` heading to match the requested ticker, quarter, and fiscal year.
- Rejects ambiguous or previous-quarter pages instead of saving a best-effort result.
- Resolves the Google News result to its canonical Investing.com URL, tries browser-impersonated HTTP first, then uses a persistent Playwright Chromium profile when Investing.com returns a challenge or incomplete page.
- Saves the transcript into the same output folder as the SEC filings.
- If `--pdf` is set, it creates a PDF and removes the HTML (same behavior as SEC filings).
- If `--pdf` is not set, it keeps the HTML only.
- Trims the PDF to remove first/last pages (defaults: first=1, last=2; configurable).
  - If `wkhtmltopdf` reports external resource load errors but still creates the PDF, the file is kept.

With `--debug`, transcript logs include:
- Accepted and rejected RSS candidates.
- The resolved Investing.com URL.
- Whether browser-impersonated HTTP, direct HTTP, or Playwright retrieved the page.
- The exact transcript identity heading that passed validation.

If no exact candidate passes validation, the command exits non-zero.

### Browser fallback

Playwright stores its reusable Investing.com browser session in
`.secrets/investing-browser`. Headless Chromium is attempted first. If
Investing.com still presents a security check in an interactive terminal, the
tool opens a visible browser once and asks you to complete it. Later runs reuse
that browser profile.

### Cookies

The default cookie file is `.secrets/investing.cookie.txt`.

You can paste either:
- A raw `Cookie:` header value, or
- Netscape cookie file format (for example from "Get cookies.txt locally").

The `--transcript-cookie-file` flag overrides this path.

To obtain cookies from your browser:

1. Open the transcript page in your browser (Brave/Chrome/Edge).
2. Open DevTools (F12) and go to the **Network** tab.
3. Click the **Doc** filter (to show the main document requests).
4. Refresh the page and click the top transcript document entry.
5. In the right pane, open **Headers** → **Request Headers**.
6. Copy the full value of **Cookie:** (everything after `Cookie:`).

Pass the cookie once via `--transcript-cookie`; it is saved for later direct
HTTP attempts. The persistent Playwright profile is the preferred fallback.

### Transcript output naming

Transcript files include the substring `earnings_call_transcript`:
```
<ticker>_q<q>_<fy>_earnings_call_transcript.html
<ticker>_q<q>_<fy>_earnings_call_transcript.pdf
```

Example:
```
aapl_q1_2026_earnings_call_transcript.pdf
```

Work around TLS issues (not recommended):
```bash
python3 sec_earnings_8k.py --ticker COST --insecure
```

### Windows TLS troubleshooting

If you see SSL verification errors on Windows, try one of these:
- Run with `--ca-bundle` pointing to a PEM CA bundle.
- As a last resort, run with `--insecure` (not recommended).

### macOS TLS troubleshooting (uv/venv)

If TLS verification fails on macOS, your Python may not have a CA bundle.
Quick fix using `certifi`:

```bash
uv pip install certifi
python -c "import certifi; print(certifi.where())"
python3 sec_earnings_8k.py --ticker COST --ca-bundle "$(python -c 'import certifi; print(certifi.where())')"
```

If you installed Python from python.org, you can also run the bundled
`Install Certificates.command` in the Python Applications folder.

## Python script usage (Windows or macOS)

Example: download SEC files for `COST` into a specified folder.

```python
import ssl

from sec_earnings_8k import fetch_latest_earnings_8k

ticker = "COST"
outdir = r"C:\sec-downloads"  # Use a raw string on Windows.
user_agent = "Your Name you@email.com"

exit_code = fetch_latest_earnings_8k(
    ticker=ticker,
    date_filter=None,
    outdir=outdir,
    user_agent=user_agent,
    ssl_context=ssl.create_default_context(),
    save_pdf=False,
    q=4,
    fy=2025,
)

print("Done, exit code:", exit_code)
```

## Optional install (pip)

You can install the module locally for easier imports:

```bash
pip install -e .
```

## Output layout

If `--q`/`--fy` are provided, files are saved to a folder named:
```
COST_Q4_2025/
```

If `--q`/`--fy` are omitted, files are saved to:
```
./sec_earnings_8k/COST/
```

Inside the folder:
- `cost_q4_2025_8k_991.htm` / `cost_q4_2025_8k_991.pdf`
- `cost_q4_2025_8k_992.htm` / `cost_q4_2025_8k_992.pdf`
- `cost_q3_2025_10q.htm` / `cost_q3_2025_10q.pdf`
- `cost_q4_2025_earnings_call_transcript.html` / `cost_q4_2025_earnings_call_transcript.pdf`
- `img/` (linked images referenced by the HTML)

If `--q`/`--fy` are omitted, the filenames drop the quarter/year prefix:
- `cost_8k_991.htm`, `cost_8k_992.htm`, `cost_10q.htm` (or `cost_10k.htm`)
