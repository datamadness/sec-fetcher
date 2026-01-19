Project status (as of today)

We built a Python CLI tool that downloads the latest earnings-related 8-K
exhibits (EX-99.1 / EX-99.2) for a given ticker, optionally filtered by a
specific filing date. It also downloads the most recent 10-Q or 10-K filed
before that 8-K and saves the primary document for the prior report.

Current behavior
- Resolves ticker -> CIK via SEC company_tickers.json.
- Filters 8-K filings to Item 2.02 (earnings).
- Downloads EX-99.1 and EX-99.2 from the SEC archives.
- For HTML exhibits, downloads linked images into an img/ subfolder and
  rewrites HTML to reference img/ so slides render offline.
- Optional PDF conversion via wkhtmltopdf (flag --pdf). HTML remains default.
- Downloads prior 10-Q/10-K primary document filed before the 8-K.

Important notes
- Quarter date extraction was attempted but removed due to unreliable parsing.
- PDF conversion works only if wkhtmltopdf is installed and in PATH.
- TLS verification fails on this machine, so runs currently require --insecure.
  Future improvement: fix CA bundle / trust store so --insecure is not needed.

Key files
- sec_earnings_8k.py (main script)
- README.md (usage, flags, and setup)
- requirements.txt (no third-party dependencies)

Example usage
- python3 sec_earnings_8k.py --ticker COST --insecure
- python3 sec_earnings_8k.py --ticker COST --pdf --insecure

Next possible work
- Consider adding a robust quarter-date solution using authoritative metadata,
  if available, without regex parsing.
