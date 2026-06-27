import os
import requests
from pathlib import Path

DRIVE_API = "https://www.googleapis.com/drive/v3"
import os
DOWNLOADS_DIR = Path(os.getenv("DOWNLOADS_DIR", str(Path(__file__).parent.parent / "downloads")))


def list_subfolders(folder_id: str) -> list[dict]:
    """List all subfolders in a public Google Drive folder."""
    url = f"{DRIVE_API}/files"
    params = {
        "q": f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        "fields": "files(id,name,modifiedTime)",
        "pageSize": 200,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def list_files_in_folder(folder_id: str) -> list[dict]:
    """List all files (non-folders) in a Google Drive folder."""
    url = f"{DRIVE_API}/files"
    params = {
        "q": f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false",
        "fields": "files(id,name,mimeType,modifiedTime,size)",
        "pageSize": 50,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def download_file(file_id: str, filename: str) -> Path:
    """Download a file from Google Drive to the downloads directory."""
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    dest = DOWNLOADS_DIR / filename

    # Try export for Google Docs formats, otherwise direct download
    export_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    resp = requests.get(export_url, timeout=120, stream=True)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return dest


def get_file_content_as_pdf_path(folder_id: str, company_name: str) -> Path | None:
    """Get the first PDF or document from a Drive subfolder and download it."""
    files = list_files_in_folder(folder_id)
    if not files:
        return None

    # Prefer PDF, then Word, then any file
    preferred = sorted(files, key=lambda f: (
        0 if "pdf" in f.get("mimeType", "") else
        1 if "word" in f.get("mimeType", "") or f["name"].endswith(".docx") else 2
    ))

    target = preferred[0]
    safe_name = f"{company_name}_{target['id']}.pdf"
    return download_file(target["id"], safe_name)
