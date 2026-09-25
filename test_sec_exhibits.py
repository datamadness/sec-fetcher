import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import sec_earnings_8k as fetcher


def document(file_type: str, filename: str) -> str:
    return f"""
<DOCUMENT>
<TYPE>{file_type}
<SEQUENCE>2
<FILENAME>{filename}
<DESCRIPTION>{file_type}
<TEXT>
</DOCUMENT>
"""


class SubmissionExhibitTests(unittest.TestCase):
    def test_finds_opaque_nvidia_exhibit_names_from_document_types(self) -> None:
        submission = document("8-K", "nvda-20260826.htm")
        submission += document("EX-99.1", "q2fy27pr.htm")
        submission += document("EX-99.2", "q2fy27cfocommentary.htm")

        self.assertEqual(
            {
                "EX-99.1": "q2fy27pr.htm",
                "EX-99.2": "q2fy27cfocommentary.htm",
            },
            fetcher._find_exhibit_files_from_submission(submission),
        )

    def test_normalizes_zero_padded_exhibit_types(self) -> None:
        submission = document("EX-99.01", "release.pdf")
        submission += document("EX-99.02", "commentary.txt")

        self.assertEqual(
            {"EX-99.1": "release.pdf", "EX-99.2": "commentary.txt"},
            fetcher._find_exhibit_files_from_submission(submission),
        )

    def test_prefers_html_when_an_exhibit_has_multiple_formats(self) -> None:
        submission = document("EX-99.1", "release.pdf")
        submission += document("EX-99.1", "release.htm")

        exhibits = fetcher._find_exhibit_files_from_submission(submission)

        self.assertEqual("release.htm", exhibits["EX-99.1"])


class DirectoryFallbackTests(unittest.TestCase):
    def test_does_not_treat_sec_metadata_pages_as_exhibits(self) -> None:
        index_json = {
            "directory": {
                "item": [
                    {"name": "0001045810-26-000073-index-headers.html", "type": "text.gif"},
                    {"name": "0001045810-26-000073-index.html", "type": "text.gif"},
                    {"name": "0001045810-26-000073.txt", "type": "text.gif"},
                ]
            }
        }

        self.assertEqual(
            {"EX-99.1": None, "EX-99.2": None},
            fetcher._find_exhibit_files(index_json, "nvda-20260826.htm"),
        )

    def test_retains_filename_based_fallback(self) -> None:
        index_json = {
            "directory": {
                "item": [
                    {"name": "amzn-20260630xex991.htm", "type": "text.gif"},
                    {"name": "amzn-20260630xex992.htm", "type": "text.gif"},
                ]
            }
        }

        self.assertEqual(
            {
                "EX-99.1": "amzn-20260630xex991.htm",
                "EX-99.2": "amzn-20260630xex992.htm",
            },
            fetcher._find_exhibit_files(index_json),
        )


class FilingFetchTests(unittest.TestCase):
    def test_keeps_newly_generated_pdf_when_renderer_reports_missing_resource(self) -> None:
        with TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "filing.pdf"
            pdf_path.write_bytes(b"stale")

            def render(*args, **kwargs):
                self.assertFalse(pdf_path.exists())
                pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
                return SimpleNamespace(returncode=1)

            with patch.object(fetcher.shutil, "which", return_value="wkhtmltopdf"), patch.object(
                fetcher.subprocess, "run", side_effect=render
            ):
                fetcher._convert_html_to_pdf("filing.htm", str(pdf_path), allow_failure_if_output=True)

            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-"))

    def test_does_not_scan_older_filing_dates(self) -> None:
        submissions = {"filings": {"recent": {
            "form": ["8-K", "8-K"],
            "filingDate": ["2026-09-24", "2020-03-31"],
            "accessionNumber": ["0000000123-26-000001", "0000000123-20-000001"],
            "items": ["2.02", "2.02"],
            "primaryDocument": ["current.htm", "old.htm"],
        }}}
        no_exhibits = {"EX-99.1": None, "EX-99.2": None}

        with patch.object(fetcher, "_ticker_to_cik_and_title", return_value=("123", "Test")), patch.object(
            fetcher, "_load_json", side_effect=[submissions, {}]
        ) as load_json, patch.object(fetcher, "_http_get", return_value=b""), patch.object(
            fetcher, "_find_exhibit_files_from_submission", return_value=no_exhibits
        ), patch.object(fetcher, "_find_exhibit_files", return_value=no_exhibits):
            result = fetcher.fetch_latest_earnings_8k("TEST", None, ".", "agent", None, False)

        self.assertEqual(1, result)
        self.assertEqual(2, load_json.call_count)


if __name__ == "__main__":
    unittest.main()
