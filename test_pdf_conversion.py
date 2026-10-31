import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import sec_earnings_8k as fetcher


class PdfConversionTests(unittest.TestCase):
    def test_linux_uses_chromium_and_closes_browser(self):
        api = ModuleType('playwright.sync_api')
        api.sync_playwright = MagicMock()
        browser = api.sync_playwright.return_value.__enter__.return_value.chromium.launch.return_value
        page = browser.new_page.return_value
        with patch.dict(sys.modules, {'playwright': ModuleType('playwright'), 'playwright.sync_api': api}), patch.object(fetcher.sys, 'platform', 'linux'), patch.object(fetcher.subprocess, 'run') as run:
            fetcher._convert_html_to_pdf('input file.html', 'output.pdf')
            page.goto.assert_called_once_with(Path('input file.html').resolve().as_uri(), wait_until='load')
            self.assertEqual(page.pdf.call_args.kwargs['path'], 'output.pdf')
            browser.close.assert_called_once()
            run.assert_not_called()

    def test_linux_conversion_failure_closes_browser_and_propagates(self):
        api = ModuleType('playwright.sync_api')
        api.sync_playwright = MagicMock()
        browser = api.sync_playwright.return_value.__enter__.return_value.chromium.launch.return_value
        browser.new_page.return_value.pdf.side_effect = RuntimeError('PDF failed')
        with patch.dict(sys.modules, {'playwright': ModuleType('playwright'), 'playwright.sync_api': api}), patch.object(fetcher.sys, 'platform', 'linux'):
            with self.assertRaisesRegex(RuntimeError, 'PDF failed'):
                fetcher._convert_html_to_pdf('input.html', 'output.pdf')
            browser.close.assert_called_once()

    def test_windows_preserves_converter_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            target = str(Path(folder) / 'output.pdf')
            with patch.object(fetcher.sys, 'platform', 'win32'), patch.object(fetcher.shutil, 'which', return_value='wkhtmltopdf.exe'), patch.object(fetcher.subprocess, 'run', return_value=MagicMock(returncode=0)) as run:
                fetcher._convert_html_to_pdf('input.html', target, extra_args=['--load-error-handling', 'ignore'])
                run.assert_called_once_with(['wkhtmltopdf.exe', '--quiet', '--enable-local-file-access', '--load-error-handling', 'ignore', 'input.html', target], check=False)


if __name__ == '__main__':
    unittest.main()
