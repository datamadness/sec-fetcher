import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import sec_earnings_8k as fetcher

URL = 'https://www.investing.com/news/transcripts/earnings-call-transcript-test-12345'


class TranscriptDiagnosticsTests(unittest.TestCase):
    def test_normal_run_reports_failure_without_debug_or_cookie_access(self):
        for reason in ['access challenge detected', 'rendered Full transcript heading was not found', 'ticker is OTHER, expected DASH', 'Playwright failed: browser unavailable']:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder, patch.object(fetcher, '_fetch_transcript_page', return_value=(None, URL, reason)), patch.object(fetcher, '_read_cookie_file', return_value=None), redirect_stdout(io.StringIO()) as output:
                saved = fetcher._download_transcript(ticker='DASH', company_title='DoorDash, Inc.', q=3, fy=2026, out_base=folder, user_agent='test', ssl_context=None, transcript_url=URL, transcript_cookie_file=None, cookie_value=None, save_pdf=False, trim_first=0, trim_last=0, debug=False)
                self.assertFalse(saved)
                self.assertIn(reason, output.getvalue())
                self.assertIn('no candidate yielded a validated transcript', output.getvalue())


    def test_ticker_only_failure_reports_verified_alias_recovery_without_saving(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            fetcher, '_fetch_transcript_page', return_value=(None, URL, 'ticker is DASH.A, expected DASH')
        ), patch.object(fetcher, '_read_cookie_file', return_value=None), redirect_stdout(io.StringIO()) as output:
            saved = fetcher._download_transcript(ticker='DASH', company_title='DoorDash, Inc.',
                q=3, fy=2026, out_base=folder, user_agent='test', ssl_context=None,
                transcript_url=URL, transcript_cookie_file=None, cookie_value=None,
                save_pdf=False, trim_first=0, trim_last=0, debug=False)
            self.assertFalse(saved)
            self.assertEqual(list(Path(folder).iterdir()), [])
        record = next(line for line in output.getvalue().splitlines()
                      if line.startswith('TRANSCRIPT_TICKER_MISMATCH '))
        details = json.loads(record.split(' ', 1)[1])
        self.assertEqual(details['requested_ticker'], 'DASH')
        self.assertEqual(details['source_ticker'], 'DASH.A')
        self.assertEqual(details['url'], URL)
        self.assertIn('verify the source ticker', output.getvalue())

    def test_company_and_period_conflicts_are_not_ticker_soft_failures(self):
        for company, quarter in [('Other Business', 3), ('DoorDash, Inc.', 2)]:
            html = (f'<h2>Full transcript - {company} (OTHER) Q{quarter} 2026:</h2>')
            valid, reason, _ = fetcher._validate_transcript_identity(html, 'DASH', 'DoorDash, Inc.', 3, 2026)
            self.assertFalse(valid)
            self.assertFalse(reason.startswith('ticker is '), reason)

    def test_ticker_match_retry_still_requires_company_and_period(self):
        valid, _, _ = fetcher._validate_transcript_identity(
            '<h2>Full transcript - DoorDash, Inc. (DASH.A) Q3 2026:</h2>',
            'DASH.A', 'DoorDash, Inc.', 3, 2026)
        self.assertTrue(valid)


if __name__ == '__main__':
    unittest.main()
