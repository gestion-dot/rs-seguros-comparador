import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", str(Path(__file__).parent.parent / "downloads")))


def list_subfolders(folder_id: str) -> list[dict]:
    """List all subfolders in a public Google Drive folder using embedded view."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    folders = []

    for entry in soup.find_all("div", class_="flip-entry"):
        title_div = entry.find("div", class_="flip-entry-title")
        link = entry.find("a", href=True)
        if not title_div or not link:
            continue
        href = link["href"]
        if "/folders/" in href:
            fid = href.split("/folders/")[1].split("?")[0].split("/")[0]
            folders.append({"id": fid, "name": title_div.get_text(strip=True)})

    return folders


def list_files_in_folder(folder_id: str) -> list[dict]:
    """List all files (non-folders) in a Google Drive folder using embedded view."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    files = []

    for entry in soup.find_all("div", class_="flip-entry"):
        title_div = entry.find("div", class_="flip-entry-title")
        link = entry.find("a", href=True)
        if not title_div or not link:
            continue
        href = link["href"]
        # Only non-folder items have /file/d/ in href or open?id=
        if "/folders/" not in href:
            # Extract file ID
            fid = None
            m = re.search(r"/file/d/([^/?]+)", href)
            if m:
                fid = m.group(1)
            else:
                m = re.search(r"[?&]id=([^&]+)", href)
                if m:
                    fid = m.group(1)
            if fid:
                name = title_div.get_text(strip=True)
                mime = "application/pdf" if name.lower().endswith(".pdf") else "application/octet-stream"
                files.append({"id": fid, "name": name, "mimeType": mime})

    return files


def download_file(file_id: str, filename: str) -> Path:
    """Download a file from Google Drive (public) to the downloads directory."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOADS_DIR / filename

    # Use Google's export/download URL (works for public files)
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    session = requests.Session()
    resp = session.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    # Handle Google's virus scan warning page for large files
    if "text/html" in resp.headers.get("Content-Type", ""):
        soup = BeautifulSoup(resp.text, "html.parser")
        form = soup.find("form")
        if form:
            action = form.get("action", url)
            inputs = {i.get("name"): i.get("value") for i in form.find_all("input") if i.get("name")}
            resp = session.get(action, params=inputs, timeout=120, stream=True)

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return dest


def get_file_content_as_pdf_path(folder_id: str, company_name: str) -> Path | None:
    """Get the first file from a Drive subfolder and download it."""
    files = list_files_in_folder(folder_id)
    if not files:
        return None

    # Prefer PDF first, then any file
    preferred = sorted(files, key=lambda f: 0 if "pdf" in f.get("mimeType", "") or f["name"].lower().endswith(".pdf") else 1)
    target = preferred[0]
    safe_name = f"{company_name.replace(' ', '_')}_{target['id'][:8]}.pdf"
    return download_file(target["id"], safe_name)
