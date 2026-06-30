import json
import os
import re
import time
import requests
import pdfplumber
from pathlib import Path

# ─── Motor de IA (configurable por variables de entorno) ──────────────────────
# AI_PROVIDER = "groq" (recomendado, cuota gratis alta) o "gemini".
AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "8000"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Throttling / retry
MAX_RETRIES = 4
BASE_BACKOFF = 20          # segundos si la API no sugiere retry-after
INTER_CALL_DELAY = float(os.getenv("INTER_CALL_DELAY", "3"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "40000"))  # acota tokens por request

_last_call_ts = [0.0]
_gemini_model = [None]      # lazy init


class QuotaExhaustedError(Exception):
    """La cuota diaria del proveedor de IA está agotada (no reintentar hoy)."""


class RateLimitError(Exception):
    """Límite por minuto: conviene esperar y reintentar."""
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


class RequestTooLargeError(Exception):
    """La request excede el límite de tokens por minuto del tier (413). Reducir input."""
    def __init__(self, msg, requested=None, limit=None):
        super().__init__(msg)
        self.requested = requested
        self.limit = limit


def _is_rate_limit_text(s: str) -> bool:
    s = s.lower()
    return "429" in s or "rate limit" in s or "rate_limit" in s or "quota" in s or "exhausted" in s


def _is_daily_quota_text(s: str) -> bool:
    s = s.lower()
    return "per day" in s or "perday" in s or "per_day" in s or "tpd" in s or "rpd" in s or "tokens per day" in s


def _parse_retry_after(header_val, body_text) -> float | None:
    if header_val:
        try:
            return float(header_val)
        except ValueError:
            pass
    # Gemini: "retry_delay { seconds: 30 }"
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", body_text)
    if m:
        return float(m.group(1))
    # Groq: "try again in 1m2.5s" o "in 20.5s"
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", body_text, re.IGNORECASE)
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        return mins * 60 + float(m.group(2))
    return None


def _throttle():
    elapsed = time.time() - _last_call_ts[0]
    if elapsed < INTER_CALL_DELAY:
        time.sleep(INTER_CALL_DELAY - elapsed)


def _call_groq(system: str, user: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Falta configurar GROQ_API_KEY")
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": GROQ_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    def _post(b):
        return requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=b,
            timeout=180,
        )

    r = _post(body)
    # Si el modo JSON estricto rechaza la salida, reintentar sin response_format
    # (a veces el modelo razona en texto; sin el modo estricto suele devolver JSON limpio).
    if r.status_code == 400 and "json_validate_failed" in r.text:
        body2 = {k: v for k, v in body.items() if k != "response_format"}
        r = _post(body2)
    if r.status_code == 413 and "per minute" in r.text.lower():
        req = re.search(r"Requested (\d+)", r.text)
        lim = re.search(r"Limit (\d+)", r.text)
        raise RequestTooLargeError(
            r.text[:300],
            requested=int(req.group(1)) if req else None,
            limit=int(lim.group(1)) if lim else None,
        )
    if r.status_code == 429:
        if _is_daily_quota_text(r.text):
            raise QuotaExhaustedError(r.text[:400])
        raise RateLimitError(r.text[:400], retry_after=_parse_retry_after(r.headers.get("retry-after"), r.text))
    if r.status_code >= 400:
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
    _last_call_ts[0] = time.time()
    return r.json()["choices"][0]["message"]["content"]


def _call_gemini(full_prompt: str) -> str:
    if _gemini_model[0] is None:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _gemini_model[0] = genai.GenerativeModel(GEMINI_MODEL)
    try:
        resp = _gemini_model[0].generate_content(full_prompt)
        _last_call_ts[0] = time.time()
        return resp.text
    except Exception as e:  # noqa: BLE001
        _last_call_ts[0] = time.time()
        s = str(e)
        if _is_rate_limit_text(s) and _is_daily_quota_text(s):
            raise QuotaExhaustedError(s)
        if _is_rate_limit_text(s):
            raise RateLimitError(s, retry_after=_parse_retry_after(None, s))
        raise


def _generate_with_retry(system_prompt: str, user_prompt: str) -> str:
    """Llama al proveedor de IA configurado, con throttling y reintentos por rate-limit."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            if AI_PROVIDER == "gemini":
                return _call_gemini(f"{system_prompt}\n\n{user_prompt}")
            return _call_groq(system_prompt, user_prompt)
        except QuotaExhaustedError:
            raise
        except RateLimitError as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(e.retry_after or (BASE_BACKOFF * (attempt + 1)))
    raise last_err  # pragma: no cover

# Instrucciones de extracción EDITABLES por el usuario (desde Opciones).
# Definen QUÉ extraer. El formato de salida (JSON) lo fuerza JSON_FORMAT_BLOCK.
DEFAULT_INSTRUCTIONS = """Actúa como un experto analista de datos y especialista en pólizas de seguros. Te paso el contenido de un manual de productos de una compañía de seguros.

Extraé la información técnica de TODOS los planes/productos de seguro del documento, sin omitir ninguno (ej.: Plan Básico, Terceros, Terceros Completo, Todo Riesgo, etc.).

Para CADA plan, extraé el detalle de:
- COBERTURAS / RIESGOS: Robo Total, Robo Parcial, Incendio Total, Incendio Parcial, Responsabilidad Civil, Daños por Granizo, Cristales, Cerraduras, Ajuste por inflación, y cualquier otra cobertura o beneficio que figure.
- REQUISITOS: Antigüedad Máxima Aceptada del bien.
- OPERATIVO: Forma de realizar la Inspección Previa (IP).
- CONDICIONES COMERCIALES: Franquicias / Deducibles, Carencias y Exclusiones principales.

Para cada dato indicá de forma clara y concisa si está incluido y bajo qué condiciones (ej.: "Sí, al 100%", "No cubierto", "Sí, franquicia 5%"), o el detalle del proceso si es la Inspección Previa. Si el manual no especifica un dato para un plan, poné "No especificado".

Sé riguroso y preciso. No inventes datos."""


# Bloque de formato OBLIGATORIO (no editable) — garantiza JSON parseable por la app.
JSON_FORMAT_BLOCK = """
────────────────────────────────────────
FORMATO DE SALIDA OBLIGATORIO (ignorá cualquier instrucción previa que pida una tabla de texto):
Devolvé ÚNICAMENTE un objeto JSON válido, sin texto fuera del JSON, con esta estructura exacta:
{
  "compania": "Nombre de la compañía",
  "fecha_actualizacion_manual": "DD/MM/AAAA o null",
  "ramas": [
    {
      "rama": "Autos",
      "planes": [
        {
          "nombre_plan": "Nombre del plan (= columna de la matriz)",
          "variante": "A o null",
          "grupo": "RC | Garage | Todo/Total | Terceros Completo | Todo Riesgo (solo Autos/Motos; null en otras ramas)",
          "coberturas": {
            "clave_snake_case": "valor de la celda"
          },
          "particularidades": "Máximo 2 oraciones."
        }
      ]
    }
  ]
}

REGLAS DE MAPEO:
- Cada PLAN = una columna. Cada clave dentro de "coberturas" = una FILA de la matriz.
- Usá una clave snake_case por cada concepto. Para las coberturas estándar usá EXACTAMENTE: responsabilidad_civil, robo_total, robo_parcial, incendio_total, incendio_parcial, destruccion_total, danos_parciales, granizo, cristales, cerraduras, ajuste_inflacion, antiguedad_maxima, forma_inspeccion_previa, franquicia_deducible, carencias, exclusiones. Agregá cualquier otra cobertura/beneficio que figure con su propia clave.
- En cada valor poné lo que indique el manual (ej.: "Sí, al 100%", "No cubierto", "Sí, franquicia 5%"). Si no se especifica para ese plan, poné "No especificado".
- Dividí por rama (Autos, Motos, Hogar, etc.) según lo que ofrezca el manual.
- No inventes datos."""


def build_system_prompt(instructions: str | None = None) -> str:
    instr = (instructions or "").strip() or DEFAULT_INSTRUCTIONS
    return instr + "\n" + JSON_FORMAT_BLOCK


def _looks_like_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5).startswith(b"%PDF")
    except OSError:
        return False


def _extract_text_from_html_file(path: Path) -> str:
    from bs4 import BeautifulSoup
    raw = path.read_bytes().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    # The Drive download may have returned an HTML page (interstitial / Google Doc)
    # instead of a real PDF. Fall back to HTML text extraction in that case.
    if not _looks_like_pdf(pdf_path):
        text = _extract_text_from_html_file(pdf_path)
        if not text.strip():
            raise ValueError("El archivo descargado no es un PDF válido ni contiene texto legible.")
        return text

    limite = MAX_INPUT_CHARS * 2

    # Primario: pypdf (muy liviano en memoria). pdfplumber renderiza tablas/imágenes
    # y revienta los 512MB de Render free con PDFs pesados (el worker muere sin log).
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        parts = []
        total = 0
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
                total += len(t)
            if total >= limite:
                break
        text = "\n".join(parts)
        if text.strip():
            return text
    except Exception:
        pass  # cae al método pesado si pypdf falla

    # Fallback: pdfplumber (más preciso con tablas, pero pesado)
    text_parts = []
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
                total += len(text)
            page.flush_cache()
            if total >= limite:
                break
    return "\n".join(text_parts)


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip().rstrip("```").strip()


def extract_coverages_from_text(text: str, company_name: str, instructions: str | None = None) -> dict:
    # Quitar caracteres de control no imprimibles (PDFs corruptos descarrilan al modelo)
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    system_prompt = build_system_prompt(instructions)

    chars = MAX_INPUT_CHARS
    last_err = None
    for _ in range(4):
        truncated = clean[:chars]
        user_prompt = (
            f"Compañía: {company_name}\n\nContenido del manual:\n\n{truncated}\n\n"
            "IMPORTANTE: Respondé ÚNICAMENTE con el objeto JSON especificado, "
            "empezando con { y terminando con }. Nada de texto, tablas ni explicaciones fuera del JSON."
        )
        try:
            raw = _generate_with_retry(system_prompt, user_prompt)
            return json.loads(clean_json_response(raw))
        except RequestTooLargeError as e:
            last_err = e
            # Reducir el input según el exceso reportado por la API (con margen)
            if e.requested and e.limit:
                chars = int(chars * (e.limit / e.requested) * 0.85)
            else:
                chars = int(chars * 0.6)
            if chars < 3000:
                raise
    raise last_err  # pragma: no cover


def extract_from_pdf(pdf_path: Path, company_name: str, instructions: str | None = None) -> dict:
    text = extract_text_from_pdf(pdf_path)
    return extract_coverages_from_text(text, company_name, instructions)


def extract_from_text(text: str, company_name: str, instructions: str | None = None) -> dict:
    return extract_coverages_from_text(text, company_name, instructions)
