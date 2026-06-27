import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url: str) -> str:
    """Extract readable text from a web URL (insurance manual page)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RS-Seguros-Bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines)
