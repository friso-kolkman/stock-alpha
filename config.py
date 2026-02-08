"""Configuration for Stock Alpha Scanner."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

# Scanner settings
TOP_N_STOCKS = 10  # Research top N per run
MIN_ALPHA_SCORE = 50  # Minimum score threshold

# Perplexity API
PERPLEXITY_API_BASE = "https://api.perplexity.ai"
PERPLEXITY_MODEL = "sonar-pro"

# Perplexity Budget (Revolut Ultra = $4.99/month)
PERPLEXITY_MONTHLY_BUDGET = 4.99  # USD
# Pricing for sonar-pro (as of 2024)
PERPLEXITY_PRICE_PER_1K_INPUT = 0.003   # $3 per 1M input tokens
PERPLEXITY_PRICE_PER_1K_OUTPUT = 0.015  # $15 per 1M output tokens

# Alpha scoring weights (total = 100)
SCORING_WEIGHTS = {
    "value": 25,
    "quality": 25,
    "momentum": 25,
    "catalyst": 15,
    "liquidity": 10,
}

# Stock Universe organized by index
STOCK_UNIVERSE = {
    "AEX": {
        "exchange": "AMS",
        "currency": "EUR",
        "tickers": [
            "ASML", "SHELL", "UNA", "PRX", "ADYEN",
            "INGA", "ASM", "HEIA", "DSM", "PHIA",
            "WKL", "ABN", "NN", "AKZA", "AD",
            "KPN", "RAND", "UMG", "IMCD", "BESI",
            "AGN", "AH", "URW", "LIGHT", "EXOR",
        ],
    },
    "DAX": {
        "exchange": "XETR",
        "currency": "EUR",
        "tickers": [
            "SAP", "SIE", "ALV", "DTE", "AIR",
            "MBG", "MUV2", "BAS", "BMW", "IFX",
            "DHL", "DB1", "BEI", "SHL", "ADS",
            "VOW3", "HEN3", "RWE", "MRK", "DPW",
            "FRE", "SY1", "MTX", "PAH3", "HEI",
            "CON", "ENR", "DBK", "ZAL", "QIA",
            "BNR", "RHM", "P911", "DTG", "1COV",
            "EOAN", "FME", "HAG", "HNR1", "TKA",
        ],
    },
    "CAC40": {
        "exchange": "EPA",
        "currency": "EUR",
        "tickers": [
            "MC", "OR", "TTE", "SAN", "AI",
            "AIR", "SU", "BN", "CS", "DG",
            "SAF", "RI", "KER", "BNP", "EL",
            "EN", "CAP", "SGO", "STM", "HO",
            "DSY", "VIV", "GLE", "ACA", "PUB",
            "LR", "ML", "ERF", "TEP", "RNO",
            "VIE", "ORA", "ATO", "SW", "CA",
            "STLAP", "RMS", "WLN", "MT", "URW",
        ],
    },
    "FTSE100": {
        "exchange": "LSE",
        "currency": "GBP",
        "tickers": [
            "SHEL", "AZN", "ULVR", "HSBA", "BP",
            "GSK", "RIO", "REL", "DGE", "LSEG",
            "AAL", "BA", "NG", "VOD", "CPG",
            "LLOY", "BARC", "NWG", "ABF", "CRH",
            "AHT", "BKG", "BNZL", "BME", "BATS",
            "BRBY", "CNA", "CCH", "CTEC", "DARK",
            "DPLM", "EXPN", "GLEN", "HLN", "HLMA",
            "HWDN", "IHG", "III", "IMB", "INF",
            "ITRK", "JD", "KGF", "LAND", "LGEN",
            "MNG", "MNDI", "PSON", "PSH", "PHNX",
            "PRU", "RKT", "RR", "RS1", "RTO",
            "SDR", "SGE", "SGRO", "SMDS", "SMIN",
            "SMT", "SN", "SSE", "STAN", "SVT",
            "TSCO", "TW", "WEIR", "WPP", "WTB",
            "AV", "ENT", "FLTR", "FRAS", "HIK",
            "HSX", "IAG", "ICG", "ICP", "JMAT",
            "MKS", "PSN", "REL", "SBRY", "SHEL",
            "SKG", "SPX", "STJ", "UU", "AUTO",
            "AVV", "ANTO", "ADM", "BDEV", "BT",
            "CRDA", "EDV", "EZJ", "FERG", "GROW",
        ],
    },
    "DJIA": {
        "exchange": "US",
        "currency": "USD",
        "tickers": [
            "AAPL", "MSFT", "UNH", "GS", "HD",
            "AMGN", "MCD", "CAT", "V", "CRM",
            "TRV", "AXP", "JPM", "BA", "IBM",
            "HON", "JNJ", "WMT", "PG", "CVX",
            "MRK", "DIS", "NKE", "KO", "CSCO",
            "INTC", "VZ", "DOW", "WBA", "MMM",
        ],
    },
    "NASDAQ_SELECT": {
        "exchange": "US",
        "currency": "USD",
        "tickers": [
            "NVDA", "AMZN", "GOOG", "META", "TSLA",
            "AVGO", "NFLX", "AMD", "ADBE", "PEP",
            "COST", "QCOM", "TMUS", "AMAT", "ISRG",
        ],
    },
}

# DeGiro free core selection ETFs
DEGIRO_FREE_ETFS = [
    {"ticker": "VWRL", "exchange": "AMS", "name": "Vanguard FTSE All-World"},
    {"ticker": "IWDA", "exchange": "AMS", "name": "iShares Core MSCI World"},
    {"ticker": "SXR8", "exchange": "XETR", "name": "iShares Core S&P 500"},
    {"ticker": "SPY", "exchange": "US", "name": "SPDR S&P 500 ETF"},
    {"ticker": "QQQ", "exchange": "US", "name": "Invesco QQQ Trust"},
    {"ticker": "VUSA", "exchange": "AMS", "name": "Vanguard S&P 500"},
    {"ticker": "IEMA", "exchange": "AMS", "name": "iShares MSCI EM"},
    {"ticker": "EQQQ", "exchange": "AMS", "name": "Invesco EQQQ Nasdaq-100"},
]

# yfinance ticker suffix by exchange
YFINANCE_SUFFIX = {
    "AMS": ".AS",
    "XETR": ".DE",
    "EPA": ".PA",
    "LSE": ".L",
    # US tickers need no suffix
}

# GICS sector mapping
SECTORS = {
    "Technology": ["software", "semiconductor", "hardware", "cloud", "SaaS"],
    "Financials": ["banking", "insurance", "asset management", "fintech"],
    "Healthcare": ["pharma", "biotech", "medical devices", "diagnostics"],
    "Consumer Staples": ["food", "beverage", "household", "personal care"],
    "Consumer Discretionary": ["retail", "automotive", "luxury", "e-commerce"],
    "Industrials": ["aerospace", "defense", "machinery", "logistics"],
    "Energy": ["oil", "gas", "renewable", "utilities"],
    "Materials": ["chemicals", "mining", "metals", "construction"],
    "Communication Services": ["telecom", "media", "entertainment", "advertising"],
    "Utilities": ["electric", "water", "gas distribution"],
    "Real Estate": ["REIT", "property", "real estate"],
}

# Sectors with historically stronger alpha signals
PRIORITY_SECTORS = ["Technology", "Industrials", "Healthcare", "Consumer Staples"]

# Historical tracking & backtesting
EVAL_TIMEFRAMES = {
    "30d": {
        "min_days": 30,
        "max_days": 45,
        "buy_correct": 2.0,
        "buy_incorrect": -2.0,
        "hold_upper": 4.0,
        "hold_lower": -2.0,
        "avoid_correct": -2.0,
        "avoid_incorrect": 2.0,
    },
    "60d": {
        "min_days": 60,
        "max_days": 75,
        "buy_correct": 3.5,
        "buy_incorrect": -3.5,
        "hold_upper": 7.0,
        "hold_lower": -3.5,
        "avoid_correct": -3.5,
        "avoid_incorrect": 3.5,
    },
    "90d": {
        "min_days": 90,
        "max_days": 180,
        "buy_correct": 5.0,
        "buy_incorrect": -5.0,
        "hold_upper": 10.0,
        "hold_lower": -5.0,
        "avoid_correct": -5.0,
        "avoid_incorrect": 5.0,
    },
}
MAX_WEIGHT_ADJUSTMENT = 3       # Max +/- offset per factor from base weight
MIN_EVALUATIONS_FOR_ADAPT = 10  # Minimum evaluated predictions before adapting
