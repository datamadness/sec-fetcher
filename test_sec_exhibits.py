import unittest

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


if __name__ == "__main__":
    unittest.main()
