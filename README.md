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

## Usage

```bash
python3 sec_earnings_8k.py --ticker COST
```

## Flags

- `--ticker` (required): Company ticker symbol, e.g. `COST`.
- `--date` (optional): Filing date filter in `YYYY-MM-DD`.
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

## Output layout

Each filing is stored in a folder like:
```
./sec_earnings_8k/COST_2025-12-11_0000909832-25-000164/
```

Inside the folder:
- `EX-99.1_*.htm`
- `EX-99.2_*.htm`
- `PRIOR_10Q_*` or `PRIOR_10K_*` (the most recent report before the 8-K)
- `img/` (linked images referenced by the HTML)
- `EX-99.*.pdf` (only if `--pdf` is used)
