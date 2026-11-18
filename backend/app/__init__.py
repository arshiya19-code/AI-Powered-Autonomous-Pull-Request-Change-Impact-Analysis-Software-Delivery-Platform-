"""FastAPI application package."""
from pathlib import Path
from dotenv import load_dotenv

# Load before database, GitHub, CORS and provider instances read configuration.
load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
