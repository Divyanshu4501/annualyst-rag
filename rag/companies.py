"""Detect which company (or companies) a question is about, from the question text only.

Slugs must match the `company` field in the chunks (the PDF filename before "_").
Never use the eval file's "company" field for this — that would leak the answer.
"""
import re

COMPANY_ALIASES = {
    "asianpaints": ["asian paints", "asianpaints", "white teak", "weatherseal"],
    "eternal": ["eternal", "zomato", "blinkit", "hyperpure"],
    "hdfcbank": ["hdfc bank", "hdfcbank", "hdfc", "hdb financial", "hdb financial services"],
    "icicibank": ["icici bank", "icicibank", "icici"],
    "infosys": ["infosys", "infy", "edgeverve", "finacle"],
    "itc": ["itc", "aashirvaad", "sunfeast", "bingo", "classmate", "savlon", "fiama"],
    "nykaa": ["nykaa", "fsn e-commerce", "fsn ecommerce"],
    "titan": ["titan", "tanishq", "caratlane", "fastrack", "taneira", "zoya"],
}

_PATTERNS = {
    slug: re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)
    for slug, aliases in COMPANY_ALIASES.items()
}


def detect_companies(question):
    """Return the set of company slugs named in the question (empty set = no filter)."""
    return {slug for slug, pat in _PATTERNS.items() if pat.search(question)}


if __name__ == "__main__":
    import sys
    for q in sys.argv[1:] or ["What was Zomato's GOV growth?", "HDFC Bank net profit FY26",
                              "Compare Tanishq and Asian Paints margins", "What is the repo rate?"]:
        print(f"{sorted(detect_companies(q)) or '-'}  <-  {q}")