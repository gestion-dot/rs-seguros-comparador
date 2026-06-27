import json
import os
import re
import time
import pdfplumber
from pathlib import Path
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# El modelo se puede cambiar por env var. flash-lite tiene una cuota diaria
# gratuita separada (y más alta) que gemini-2.5-flash.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
model = genai.GenerativeModel(MODEL_NAME)

# Throttling / retry config (free-tier Gemini rate limits)
MAX_RETRIES = 3
BASE_BACKOFF = 20          # seconds, used if API doesn't suggest a retry delay
INTER_CALL_DELAY = 6       # seconds between successful calls (~10 req/min ceiling)
MAX_INPUT_CHARS = 45000    # cap input size to reduce tokens-per-minute pressure

_last_call_ts = [0.0]      # mutable holder for last successful call timestamp


class QuotaExhaustedError(Exception):
    """Raised when Gemini reports the daily/project quota is exhausted."""


def _is_daily_quota(err: Exception) -> bool:
    s = str(err).lower()
    return "perday" in s or "per day" in s or "per_day" in s


def _retry_delay_from_error(err: Exception) -> float | None:
    """Parse the suggested retry delay (seconds) from a Gemini 429 error, if present."""
    msg = str(err)
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
    if m:
        return float(m.group(1))
    m = re.search(r"retry[- ]after[\"']?\s*[:=]\s*(\d+)", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return "429" in s or "quota" in s or "exhausted" in s or "rate limit" in s


def _throttle():
    """Space out calls to stay under the per-minute request ceiling."""
    elapsed = time.time() - _last_call_ts[0]
    if elapsed < INTER_CALL_DELAY:
        time.sleep(INTER_CALL_DELAY - elapsed)


def _generate_with_retry(prompt: str) -> str:
    """Call Gemini with throttling and exponential backoff on rate-limit (429) errors."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            response = model.generate_content(prompt)
            _last_call_ts[0] = time.time()
            return response.text
        except Exception as e:  # noqa: BLE001 - SDK raises google.api_core exceptions
            last_err = e
            _last_call_ts[0] = time.time()
            # Daily/project quota: retrying within the same day is futile — fail fast.
            if _is_rate_limit(e) and _is_daily_quota(e):
                raise QuotaExhaustedError(str(e))
            if not _is_rate_limit(e) or attempt == MAX_RETRIES - 1:
                raise
            wait = _retry_delay_from_error(e) or (BASE_BACKOFF * (attempt + 1))
            time.sleep(wait)
    raise last_err  # pragma: no cover

SYSTEM_PROMPT = """Eres un analista experto en seguros de Argentina. Tu tarea es analizar manuales de pólizas de seguros y extraer la información clave de las coberturas para alimentar una base de datos comparativa.

REGLAS DE EXTRACCIÓN:
1. RAMAS: Divide la información por ramo (Autos, Motos, Hogar, Comercio, Vida, etc.) según lo que ofrezca cada compañía.
2. PLANES: Dentro de cada rama, identifica todos los planes comerciales (Terceros, Terceros Completo, Todo Riesgo, etc.).
3. VARIANTES: Si un plan tiene variantes (A, B, C, D, Garage u otras), crea una entrada por variante.
4. GRUPO (solo Autos y Motos): Clasifica cada plan en UNO de estos grupos canónicos según su nivel de cobertura real, aunque el nombre comercial sea un código (ej. "A", "C", "CL MAX"):
   - "RC": solo Responsabilidad Civil / Terceros básico.
   - "Garage": cobertura de robo/incendio en garaje.
   - "Todo/Total": Terceros con Robo/Incendio/Destrucción Total (sin llegar a Todo Riesgo).
   - "Terceros Completo": Terceros Completo y sus variantes.
   - "Todo Riesgo": Todo Riesgo con o sin franquicia.
   Para otras ramas (Hogar, etc.) usa "grupo": null.
5. COBERTURAS (EXHAUSTIVIDAD OBLIGATORIA): Lista ABSOLUTAMENTE TODAS las coberturas, beneficios, servicios, cláusulas adicionales y adicionales opcionales que el manual mencione para cada plan. NO te limites a las 8 estándar. Cada cobertura adicional se agrega con su PROPIA clave en snake_case usando su nombre real (NO uses una clave genérica como "cobertura_adicional"). Es preferible incluir de más que omitir algo.
   Ejemplos de coberturas adicionales frecuentes en Autos/Motos que debés buscar e incluir si aparecen: asistencia_al_vehiculo, auxilio_mecanico, remolque_grua, auto_sustituto, taller_oficial, gestoria_tramites, responsabilidad_civil_paises_limitrofes, granizo, inundacion, terremoto, accidentes_personales_conductor, muerte_invalidez, gastos_medicos, gastos_sepelio, cobertura_neumaticos, llanta_robada, robo_ruedas, accesorios_gnc, equipaje, cristales, cerraduras, danio_por_intento_robo, asistencia_juridica, asistencia_al_hogar, asistencia_al_viajero, ambulancia, cobertura_granizo_full, cobertura_0km, valor_a_nuevo. (La lista es orientativa: incluí cualquier otra que figure.)
6. CLAVES ESTÁNDAR: Para estas 8 coberturas usá EXACTAMENTE estas claves: responsabilidad_civil, robo_hurto, incendio, destruccion_total, danos_parciales, granizo, cristales_cerraduras, auxilio_mecanico. El resto, con su nombre real en snake_case.
7. FIDELIDAD: Extrae solo información explícita. No asumas ni inventes coberturas.
8. VALORES AUSENTES: Si una cobertura no se menciona para un plan, el valor debe ser "No incluye" o "No especificado".
9. LÍMITES Y FRANQUICIAS: Inclúyelos en la descripción (ej: "Sí - Franquicia 10%", "Sí - Límite 3 eventos anuales").

Tu respuesta debe ser ESTRICTAMENTE un objeto JSON válido con esta estructura:
{
  "compania": "Nombre",
  "fecha_actualizacion_manual": "DD/MM/AAAA o null",
  "ramas": [
    {
      "rama": "Autos",
      "planes": [
        {
          "nombre_plan": "Todo Riesgo",
          "variante": "A",
          "grupo": "Todo Riesgo",
          "coberturas": {
            "responsabilidad_civil": "Sí - hasta $X",
            "robo_hurto": "Sí - total y parcial",
            "incendio": "Sí",
            "destruccion_total": "Sí",
            "danos_parciales": "Sí - franquicia 4%",
            "granizo": "Sí",
            "cristales_cerraduras": "Sí",
            "auxilio_mecanico": "Sí - 4 servicios/año",
            "auto_sustituto": "Sí - hasta 15 días",
            "taller_oficial": "Sí",
            "accidentes_personales_conductor": "Sí - $2.000.000",
            "responsabilidad_civil_paises_limitrofes": "Sí"
          },
          "particularidades": "Máximo 2 oraciones."
        }
      ]
    }
  ]
}

No incluyas texto fuera del JSON. Si no hay variantes, omite el campo "variante" o pon null."""


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

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip().rstrip("```").strip()


def extract_coverages_from_text(text: str, company_name: str) -> dict:
    truncated = text[:MAX_INPUT_CHARS] if len(text) > MAX_INPUT_CHARS else text

    prompt = f"{SYSTEM_PROMPT}\n\nCompañía: {company_name}\n\nContenido del manual:\n\n{truncated}"

    raw = _generate_with_retry(prompt)
    return json.loads(clean_json_response(raw))


def extract_from_pdf(pdf_path: Path, company_name: str) -> dict:
    text = extract_text_from_pdf(pdf_path)
    return extract_coverages_from_text(text, company_name)


def extract_from_text(text: str, company_name: str) -> dict:
    return extract_coverages_from_text(text, company_name)
