import tempfile
import unittest
import urllib.error
import json
from unittest import mock

import sec_earnings_8k as fetcher


INVESTING_URL = (
    "https://www.investing.com/news/transcripts/"
    "earnings-call-transcript-costco-q1-2026-results-1234567"
)
GOOGLE_URL = "https://news.google.com/rss/articles/example?oc=5"


def transcript_html(
    *,
    company: str = "Costco Wholesale Corp",
    ticker: str = "COST",
    q: int = 1,
    fy: int = 2026,
    related_text: str = "",
) -> str:
    return f"""
    <html>
      <head>
        <title>Earnings call transcript: Costco Q{q} {fy}</title>
        <link rel="canonical" href="{INVESTING_URL}">
      </head>
      <body>
        <aside>{related_text}</aside>
        <article>
          <h1>Earnings call transcript: Costco Q{q} {fy}</h1>
          <h2>Full transcript - {company} ({ticker}) Q{q} {fy}:</h2>
          <p>Operator: Good afternoon and welcome to the earnings call.</p>
        </article>
      </body>
    </html>
    """


class TranscriptUrlTests(unittest.TestCase):
    def test_accepts_stock_market_transcript(self) -> None:
        self.assertTrue(fetcher._is_investing_transcript_url(
            "https://uk.investing.com/news/stock-market-news/"
            "earnings-call-transcript-carmax-tops-q2-2026-estimates-93CH-4887699"
        ))

    def test_rejects_news_and_other_hosts(self) -> None:
        for url in (
            "https://www.investing.com/news/stock-market-news/carmax-results-123",
            "https://example.com/news/stock-market-news/earnings-call-transcript-carmax-123",
        ):
            with self.subTest(url=url):
                self.assertFalse(fetcher._is_investing_transcript_url(url))


class GoogleNewsDiscoveryTests(unittest.TestCase):
    def test_selects_only_exact_company_and_period(self) -> None:
        rss = b"""<?xml version="1.0"?>
        <rss><channel>
          <item>
            <title>Earnings call transcript: Costco Q1 2026 beats forecasts - Investing.com</title>
            <link>https://news.google.com/rss/articles/correct?oc=5</link>
            <pubDate>Thu, 11 Dec 2025 08:00:00 GMT</pubDate>
            <source url="https://www.investing.com">Investing.com</source>
          </item>
          <item>
            <title>Earnings call transcript: Costco Q4 2025 beats forecasts - Investing.com</title>
            <link>https://news.google.com/rss/articles/old?oc=5</link>
            <source url="https://www.investing.com">Investing.com</source>
          </item>
          <item>
            <title>Earnings call transcript: Sleep Number Q1 2026 - Investing.com</title>
            <link>https://news.google.com/rss/articles/wrong-company?oc=5</link>
            <source url="https://www.investing.com">Investing.com</source>
          </item>
        </channel></rss>"""

        candidates = fetcher._parse_google_news_candidates(
            rss,
            ticker="COST",
            company_title="COSTCO WHOLESALE CORP /NEW",
            q=1,
            fy=2026,
        )

        self.assertEqual(1, len(candidates))
        self.assertIn("correct", candidates[0]["url"])

    def test_rejects_non_investing_source(self) -> None:
        rss = b"""<rss><channel><item>
          <title>Earnings call transcript: Costco Q1 2026</title>
          <link>https://example.com/wrong</link>
          <source url="https://example.com">Example</source>
        </item></channel></rss>"""

        candidates = fetcher._parse_google_news_candidates(
            rss, "COST", "COSTCO WHOLESALE CORP", 1, 2026
        )

        self.assertEqual([], candidates)

    def test_matches_hyphenated_company_name(self) -> None:
        rss = b"""<rss><channel><item>
          <title>Earnings call transcript: WD-40 beats Q3 2026 forecasts - Investing.com</title>
          <link>https://news.google.com/rss/articles/wdfc?oc=5</link>
          <source url="https://www.investing.com">Investing.com</source>
        </item></channel></rss>"""

        candidates = fetcher._parse_google_news_candidates(
            rss, "WDFC", "WD 40 CO", 3, 2026
        )

        self.assertEqual(1, len(candidates))

    def test_resolves_modern_google_news_url(self) -> None:
        page = b'<div data-n-a-sg="signature" data-n-a-ts="1717597091"></div>'
        payload = json.dumps(["garturlres", INVESTING_URL, 1])
        body = (")]}'\n" + json.dumps([["wrb.fr", "Fbv4je", payload]])).encode()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return body

        with mock.patch.object(fetcher, "_http_get", return_value=page), mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse()
        ):
            resolved = fetcher._resolve_google_news_url(GOOGLE_URL, None)

        self.assertEqual(INVESTING_URL, resolved)


class TranscriptIdentityTests(unittest.TestCase):
    def test_accepts_exact_full_transcript_heading(self) -> None:
        valid, reason, identity = fetcher._validate_transcript_identity(
            transcript_html(),
            ticker="COST",
            company_title="COSTCO WHOLESALE CORP /NEW",
            q=1,
            fy=2026,
        )

        self.assertTrue(valid, reason)
        self.assertEqual("COST", identity["ticker"])
        self.assertEqual(1, identity["q"])
        self.assertEqual(2026, identity["fy"])

    def test_accepts_short_company_name_when_ticker_and_period_match(self) -> None:
        valid, reason, identity = fetcher._validate_transcript_identity(
            transcript_html(company="HP Inc", ticker="HPQ", q=3, fy=2026),
            ticker="HPQ",
            company_title="HP INC",
            q=3,
            fy=2026,
        )

        self.assertTrue(valid, reason)
        self.assertEqual("HPQ", identity["ticker"])

    def test_rejects_previous_period_despite_related_page_text(self) -> None:
        html = transcript_html(
            q=4,
            fy=2025,
            related_text="Related: Costco Q1 2026 earnings call transcript",
        )

        valid, reason, _ = fetcher._validate_transcript_identity(
            html,
            ticker="COST",
            company_title="COSTCO WHOLESALE CORP /NEW",
            q=1,
            fy=2026,
        )

        self.assertFalse(valid)
        self.assertIn("period is Q4 FY2025", reason)

    def test_rejects_wrong_ticker(self) -> None:
        valid, reason, _ = fetcher._validate_transcript_identity(
            transcript_html(company="Costco Wholesale Corp", ticker="WRONG"),
            ticker="COST",
            company_title="COSTCO WHOLESALE CORP /NEW",
            q=1,
            fy=2026,
        )

        self.assertFalse(valid)
        self.assertIn("ticker is WRONG", reason)

    def test_detects_access_challenge(self) -> None:
        valid, reason, _ = fetcher._validate_transcript_identity(
            "<html><title>Just a moment...</title><div class='cf-chl-widget'></div></html>",
            ticker="COST",
            company_title="COSTCO WHOLESALE CORP",
            q=1,
            fy=2026,
        )

        self.assertFalse(valid)
        self.assertEqual("access challenge detected", reason)

    def test_cloudflare_asset_reference_is_not_a_challenge(self) -> None:
        html = transcript_html().replace(
            "</head>", "<script src='https://static.cloudflareinsights.com/beacon.js'></script></head>"
        )

        valid, reason, _ = fetcher._validate_transcript_identity(
            html,
            ticker="COST",
            company_title="COSTCO WHOLESALE CORP",
            q=1,
            fy=2026,
        )

        self.assertTrue(valid, reason)


class RetrievalTests(unittest.TestCase):
    def test_uses_impersonated_http_for_resolved_investing_url(self) -> None:
        with mock.patch.object(
            fetcher,
            "_impersonated_http_get",
            return_value=(transcript_html().encode(), INVESTING_URL),
        ), mock.patch.object(fetcher, "_http_get_with_final_url") as plain_http, mock.patch.object(
            fetcher, "_playwright_fetch_once"
        ) as browser:
            html, final_url, method = fetcher._fetch_transcript_page(
                INVESTING_URL,
                ticker="COST",
                company_title="COSTCO WHOLESALE CORP /NEW",
                q=1,
                fy=2026,
                user_agent="test",
                ssl_context=None,
                cookie=None,
                debug=False,
            )

        self.assertIsNotNone(html)
        self.assertEqual(INVESTING_URL, final_url)
        self.assertEqual("impersonated HTTP", method)
        plain_http.assert_not_called()
        browser.assert_not_called()

    def test_accepts_google_redirect_to_exact_investing_article(self) -> None:
        with mock.patch.object(
            fetcher,
            "_http_get_with_final_url",
            return_value=(transcript_html().encode(), INVESTING_URL),
        ), mock.patch.object(fetcher, "_playwright_fetch_once") as browser:
            html, final_url, method = fetcher._fetch_transcript_page(
                GOOGLE_URL,
                ticker="COST",
                company_title="COSTCO WHOLESALE CORP /NEW",
                q=1,
                fy=2026,
                user_agent="test",
                ssl_context=None,
                cookie=None,
                debug=False,
            )

        self.assertIsNotNone(html)
        self.assertEqual(INVESTING_URL, final_url)
        self.assertEqual("direct HTTP", method)
        browser.assert_not_called()

    def test_falls_back_to_playwright_after_http_403(self) -> None:
        error = urllib.error.HTTPError(
            INVESTING_URL, 403, "Forbidden", hdrs=None, fp=None
        )
        with mock.patch.object(
            fetcher, "_http_get_with_final_url", side_effect=error
        ), mock.patch.object(
            fetcher,
            "_playwright_fetch_once",
            return_value=(transcript_html(), INVESTING_URL),
        ) as browser:
            html, final_url, method = fetcher._fetch_transcript_page(
                GOOGLE_URL,
                ticker="COST",
                company_title="COSTCO WHOLESALE CORP /NEW",
                q=1,
                fy=2026,
                user_agent="test",
                ssl_context=None,
                cookie=None,
                debug=False,
            )

        self.assertIsNotNone(html)
        self.assertEqual(INVESTING_URL, final_url)
        self.assertEqual("Playwright", method)
        browser.assert_called_once()

    def test_direct_url_still_rejects_wrong_period(self) -> None:
        with tempfile.TemporaryDirectory() as outdir, mock.patch.object(
            fetcher,
            "_fetch_transcript_page",
            return_value=(None, INVESTING_URL, "period is Q4 FY2025, expected Q1 FY2026"),
        ), mock.patch("builtins.print"):
            saved = fetcher._download_transcript(
                ticker="COST",
                company_title="COSTCO WHOLESALE CORP /NEW",
                q=1,
                fy=2026,
                out_base=outdir,
                user_agent="test",
                ssl_context=None,
                transcript_url=INVESTING_URL,
                transcript_cookie_file=None,
                cookie_value=None,
                save_pdf=False,
                trim_first=0,
                trim_last=0,
                debug=False,
            )

        self.assertFalse(saved)


if __name__ == "__main__":
    unittest.main()
