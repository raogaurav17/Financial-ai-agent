"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
STANDARD_API_KEY = os.getenv("STANDARD_API_KEY", "my-secret-standard-key")
PREMIUM_API_KEY = os.getenv("PREMIUM_API_KEY", "my-secret-premium-key")
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
