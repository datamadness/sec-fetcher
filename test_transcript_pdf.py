import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import sec_earnings_8k as f

HEADER = '<h2>Full transcript - DoorDash Inc (DASH) Q3 2026:</h2>'
CONTENT = '<p>Operator: Welcome to the call.</p>' + '<p>Tony Xu: Complete example answer with business details.</p>' * 70 + '<p>Operator: You may now disconnect.</p>'
ARTICLE = '<div class="article_WYSIWYG__example">' + HEADER + CONTENT + '</div>'
URL = 'https://www.investing.com/news/transcripts/earnings-call-transcript-example-123'


class TranscriptPdfTests(unittest.TestCase):
    def test_article_extraction_removes_navigation_resources_and_preserves_call(self):
        raw = '<nav>EUR/USD</nav><link rel="stylesheet" href="https://example.com/a.css">' + ARTICLE.replace(HEADER, HEADER + '<script>bad()</script><aside>Sidebar noise</aside>') + '<footer>Footer noise</footer>'
        clean, expected = f._prepare_transcript_print_html(raw, 'DASH', 'DoorDash, Inc.', 3, 2026, URL)
        for text in ['EUR/USD', 'Sidebar noise', 'Footer noise', '<script', '<link', 'bad()']:
            self.assertNotIn(text, clean)
        self.assertIn('You may now disconnect.', expected)
        self.assertEqual(expected.count('Complete example answer'), 70)
        self.assertIn('Complete example answer', clean)
        self.assertIn('Source:', clean)

    def test_wrong_or_missing_article_is_rejected_instead_of_printing_site(self):
        for raw in [HEADER + CONTENT, ARTICLE.replace('Q3 2026','Q2 2026'), '<article>'+HEADER+'<p>Too short</p></article>']:
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                f._prepare_transcript_print_html(raw, 'DASH', 'DoorDash, Inc.', 3, 2026, URL)

    def test_pdf_checks_reject_missing_text_and_sparse_page_explosion(self):
        api = ModuleType('pypdf')
        for pages, message in [(['Only a title'], 'complete'), (['Complete call text'] + ['Indices'] * 300, 'pagination')]:
            api.PdfReader = lambda path, pages=pages: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in pages])
            with patch.dict(sys.modules, {'pypdf': api}), self.assertRaisesRegex(ValueError, message):
                f._validate_transcript_pdf('unused.pdf', 'Complete call text')

    def test_bad_linux_pdf_stops_download_and_removes_partial_pdf(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(f.sys, 'platform', 'linux'), patch.object(f, '_read_cookie_file', return_value=None), patch.object(f, '_fetch_transcript_page', return_value=(ARTICLE, URL, 'test')), patch.object(f, '_convert_html_to_pdf'), patch.object(f, '_validate_transcript_pdf', side_effect=ValueError('pagination rejected')), patch('builtins.print'):
            pdf = Path(folder) / 'dash_q3_2026_earnings_call_transcript.pdf'
            pdf.write_bytes(b'bad PDF')
            result = f._download_transcript(ticker='DASH', company_title='DoorDash, Inc.', q=3, fy=2026, out_base=folder, user_agent='test', ssl_context=None, transcript_url=URL, transcript_cookie_file=None, cookie_value=None, save_pdf=True, trim_first=0, trim_last=0, debug=False)
            self.assertFalse(result)
            self.assertFalse(pdf.exists())


if __name__ == '__main__':
    unittest.main()
