import io
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


if __name__ == '__main__':
    unittest.main()
