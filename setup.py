from setuptools import setup


setup(
    name="sec-fetcher",
    version="0.1.0",
    py_modules=["sec_earnings_8k"],
    description="Fetch latest earnings-related SEC 8-K exhibits.",
    python_requires=">=3.10",
    extras_require={
        "transcript": ["curl_cffi", "playwright"],
        "pdf": ["pypdf>=4.0.0"],
        "all": ["curl_cffi", "playwright", "pypdf>=4.0.0"],
    },
)
