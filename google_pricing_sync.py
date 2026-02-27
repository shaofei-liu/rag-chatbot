"""Get available free tier models from Google Gemini API"""

from typing import List

OFFICIAL_FREE_TIER_MODELS = {
    "gemini-2.5-flash": "Fast, efficient",
    "gemini-2.5-flash-lite": "Lightweight",
    "gemini-2.5-pro": "Advanced",
}


def get_free_tier_models() -> List[str]:
    """Get list of free tier models."""
    return list(OFFICIAL_FREE_TIER_MODELS.keys())

