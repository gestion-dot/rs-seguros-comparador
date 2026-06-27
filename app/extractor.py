import json
import pdfplumber
from pathlib import Path
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_PROMPT = """Eres un analista experto en seguros de Argentina. Tu tarea es analizar manuales de pólizas de seguros y extraer la información clave de las coberturas para alimentar una base de datos comparativa.

REGLAS DE EXTRACCIÓN:
1. RAMAS: Divide la información por ramo (Autos, Motos, Hogar, Comercio, Vida, etc.) según lo que ofrezca cada compañía.
2. PLANES: Dentro de cada rama, identifica todos los planes comerciales (Terceros, Terceros Completo, Todo Riesgo, etc.).
3. VARIANTES: Si un plan tiene variantes (A, B, C, D, Garage u otras), crea una entrada por variante.
4. COBERTURAS: Incluye TODAS las coberturas mencionadas para cada plan. Los 8 campos estándar van con claves canónicas; cualquier cobertura adicional se agrega con su nombre original.
5. FIDELIDAD: Extrae solo información explícita. No asumas ni inventes coberturas.
6. VALORES AUSENTES: Si una cobertura no se menciona para un plan, el valor debe ser "No incluye" o "No especificado".
7. LÍMITES Y FRANQUICIAS: Inclúyelos en la descripción (ej: "Sí - Franquicia 10%", "Sí - Límite 3 eventos anuales").

CLAVES ESTÁNDAR (usar exactamente estas):
- responsabilidad_civil
- robo_hurto
- incendio
- destruccion_total
- danos_parciales
- granizo
- cristales_cerraduras
- auxilio_mecanico

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
          "coberturas": {
            "responsabilidad_civil": "...",
            "robo_hurto": "...",
            "incendio": "...",
            "destruccion_total": "...",
            "danos_parciales": "...",
            "granizo": "...",
            "cristales_cerraduras": "...",
            "auxilio_mecanico": "...",
            "cobertura_adicional": "..."
          },
          "particularidades": "Máximo 2 oraciones."
        }
      ]
    }
  ]
}

No incluyas texto fuera del JSON. Si no hay variantes, omite el campo "variante" o pon null."""


def extract_text_from_pdf(pdf_path: Path) -> str:
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
    truncated = text[:60000] if len(text) > 60000 else text

    prompt = f"{SYSTEM_PROMPT}\n\nCompañía: {company_name}\n\nContenido del manual:\n\n{truncated}"

    response = model.generate_content(prompt)
    raw = response.text
    return json.loads(clean_json_response(raw))


def extract_from_pdf(pdf_path: Path, company_name: str) -> dict:
    text = extract_text_from_pdf(pdf_path)
    return extract_coverages_from_text(text, company_name)


def extract_from_text(text: str, company_name: str) -> dict:
    return extract_coverages_from_text(text, company_name)
