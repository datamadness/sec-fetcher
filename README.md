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

## Install dependencies

This tool uses the Python standard library by default.

Optional PDF conversion requires `wkhtmltopdf`:
- Install `wkhtmltopdf` and ensure it is available in your `PATH`.

Examples:
```bash
# Verify wkhtmltopdf is available
wkhtmltopdf --version
```

## Windows installation

1. Install Python 3 (from https://www.python.org/downloads/windows/) and ensure `python` is on your PATH.
2. (Optional) Install `wkhtmltopdf` if you want PDF output. Add it to PATH so `wkhtmltopdf --version` works in PowerShell.
3. No Python packages are required; `requirements.txt` is empty.
4. On Windows, use `python` instead of `python3` in the CLI examples below.

## Usage

```bash
python3 sec_earnings_8k.py --ticker COST
```

## Flags

- `--ticker` (required): Company ticker symbol, e.g. `COST`.
- `--date` (optional): Filing date filter in `YYYY-MM-DD`.
- `--q` (optional): Fiscal quarter (1-4). Must be used with `--fy`.
- `--fy` (optional): Fiscal year (YYYY). Must be used with `--q`.
- `--outdir` (optional): Output directory (default: `./sec_earnings_8k`).
- `--user-agent` (optional): SEC requires a User-Agent with contact info.
- `--ca-bundle` (optional): Path to a CA bundle (PEM) if your system certs are missing.
- `--insecure` (optional): Disable TLS verification (not recommended).
- `--pdf` (optional): Also save HTML exhibits as PDF (requires `wkhtmltopdf`).

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
- `img/` (linked images referenced by the HTML)

If `--q`/`--fy` are omitted, the filenames drop the quarter/year prefix:
- `cost_8k_991.htm`, `cost_8k_992.htm`, `cost_10q.htm` (or `cost_10k.htm`)
