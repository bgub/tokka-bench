"""
Constants and configuration for the visualization module.
"""

# Directory paths
RESULTS_DIR = "data/results"

# Chart configuration
CHART_HEIGHT = 500
LEGEND_CONFIG = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

# Script grouping mappings for language categorization
SCRIPT_GROUPS = {
    "Latn": "Latin",
    "Cyrl": "Cyrillic",
    "Arab": "Arabic",
    "Hani": "CJK",
    "Jpan": "CJK",
    "Hang": "CJK",
    "Deva": "Indic",
    "Beng": "Indic",
    "Guru": "Indic",
    "Taml": "Indic",
    "Telu": "Indic",
    "Knda": "Indic",
    "Mlym": "Indic",
    "Gujr": "Indic",
    "Orya": "Indic",
    "Thai": "Southeast Asian",
    "Laoo": "Southeast Asian",
    "Khmr": "Southeast Asian",
    "Mymr": "Southeast Asian",
    "Grek": "Other European/Middle Eastern",
    "Armn": "Other European/Middle Eastern",
    "Geor": "Other European/Middle Eastern",
    "Hebr": "Other European/Middle Eastern",
}
