import os
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from .auth import create_token, verify_token, check_credentials
from .database import get_db, init_db, Company, Branch, Plan, Coverage, SyncLog
from .drive import list_subfolders, get_file_content_as_pdf_path
from .web_source import extract_text_from_url
from .extractor import extract_from_pdf, extract_from_text, QuotaExhaustedError

app = FastAPI(title="RS Seguros Comparador")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
assets_dir = FRONTEND_DIR / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# Sync state
sync_state = {
    "running": False,
    "progress": [],   # list of {"msg": str, "type": "info|ok|error|processing"}
    "total": 0,
    "done": 0,
    "current": "",
    "error": None,
    "started_at": None,
    "finished_at": None,
}

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1yCVSrdeMQkn7HI529RkZLaNVGNd3RgEB")


# ─── Auth ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(req: LoginRequest):
    if not check_credentials(req.username, req.password):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    return {"token": create_token(req.username)}


# ─── Companies ───────────────────────────────────────────────────────────────

@app.get("/companies")
def get_companies(db: Session = Depends(get_db), _=Depends(verify_token)):
    companies = db.query(Company).filter(Company.activa == True).all()
    return [
        {
            "id": c.id,
            "nombre": c.nombre,
            "fuente": c.fuente,
            "logo_url": c.logo_url,
            "ultima_sync": c.ultima_sync.isoformat() if c.ultima_sync else None,
            "ramas": list({b.rama for b in c.branches}),
        }
        for c in companies
    ]


class UrlCompanyRequest(BaseModel):
    nombre: str
    url_manual: str
    logo_url: Optional[str] = None


@app.post("/companies/url")
def add_url_company(req: UrlCompanyRequest, db: Session = Depends(get_db), _=Depends(verify_token)):
    existing = db.query(Company).filter(Company.nombre == req.nombre).first()
    if existing:
        existing.url_manual = req.url_manual
        existing.logo_url = req.logo_url
        existing.fuente = "url"
        db.commit()
        return {"id": existing.id, "message": "Actualizada"}
    company = Company(nombre=req.nombre, fuente="url", url_manual=req.url_manual, logo_url=req.logo_url)
    db.add(company)
    db.commit()
    db.refresh(company)
    return {"id": company.id, "message": "Creada"}


@app.patch("/companies/{company_id}/logo")
def update_logo(company_id: int, body: dict, db: Session = Depends(get_db), _=Depends(verify_token)):
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404)
    company.logo_url = body.get("logo_url")
    db.commit()
    return {"ok": True}


# ─── Comparison data ─────────────────────────────────────────────────────────

@app.get("/ramas")
def get_ramas(db: Session = Depends(get_db), _=Depends(verify_token)):
    """Get all distinct branches across all companies."""
    branches = db.query(Branch.rama).distinct().all()
    return sorted({b.rama for b in branches})


@app.get("/planes")
def get_planes(rama: str, db: Session = Depends(get_db), _=Depends(verify_token)):
    """Get all distinct plan names for a given branch."""
    plans = (
        db.query(Plan.nombre_plan, Plan.variante)
        .join(Branch)
        .filter(Branch.rama == rama)
        .distinct()
        .all()
    )
    result = {}
    for nombre, variante in plans:
        if nombre not in result:
            result[nombre] = []
        if variante and variante not in result[nombre]:
            result[nombre].append(variante)
    return result


@app.get("/compare")
def compare(
    rama: str,
    plan: str,
    companies: str,
    variante: Optional[str] = None,
    db: Session = Depends(get_db),
    _=Depends(verify_token),
):
    """Compare a specific plan across selected companies."""
    company_ids = [int(x) for x in companies.split(",") if x.strip()]

    result = {}
    all_coverage_keys = {}  # key -> label

    for cid in company_ids:
        company = db.query(Company).filter(Company.id == cid).first()
        if not company:
            continue

        branch = db.query(Branch).filter(Branch.company_id == cid, Branch.rama == rama).first()
        if not branch:
            result[str(cid)] = {"nombre": company.nombre, "logo_url": company.logo_url, "coberturas": {}}
            continue

        query = db.query(Plan).filter(Plan.branch_id == branch.id, Plan.nombre_plan == plan)
        if variante:
            query = query.filter(Plan.variante == variante)
        plan_obj = query.first()

        if not plan_obj:
            result[str(cid)] = {"nombre": company.nombre, "logo_url": company.logo_url, "coberturas": {}}
            continue

        coberturas = {}
        for cov in plan_obj.coverages:
            coberturas[cov.campo_clave] = cov.valor
            all_coverage_keys[cov.campo_clave] = cov.campo_label

        result[str(cid)] = {
            "nombre": company.nombre,
            "logo_url": company.logo_url,
            "particularidades": plan_obj.particularidades,
            "coberturas": coberturas,
        }

    # Normalize: fill missing coverages with "No incluye"
    for cid_data in result.values():
        for key in all_coverage_keys:
            if key not in cid_data["coberturas"]:
                cid_data["coberturas"][key] = "No incluye"

    return {
        "coverage_keys": all_coverage_keys,
        "companies": result,
    }


# ─── Sync ────────────────────────────────────────────────────────────────────

def save_extracted_data(data: dict, db: Session):
    company_name = data.get("compania", "Desconocida")
    fecha = data.get("fecha_actualizacion_manual")

    company = db.query(Company).filter(Company.nombre == company_name).first()
    if company:
        # Clear existing branch/plan/coverage data
        for branch in company.branches:
            db.delete(branch)
        db.flush()
    else:
        company = db.query(Company).filter(Company.nombre == company_name).first()

    if company:
        company.fecha_manual = fecha
        company.ultima_sync = datetime.utcnow()
    else:
        # Should not happen but safety net
        company = Company(nombre=company_name, fecha_manual=fecha, ultima_sync=datetime.utcnow())
        db.add(company)
        db.flush()

    LABEL_MAP = {
        "responsabilidad_civil": "Responsabilidad Civil",
        "robo_hurto": "Robo y/o Hurto",
        "incendio": "Incendio",
        "destruccion_total": "Destrucción Total",
        "danos_parciales": "Daños Parciales",
        "granizo": "Granizo",
        "cristales_cerraduras": "Cristales y Cerraduras",
        "auxilio_mecanico": "Auxilio Mecánico / Remolque",
    }

    for rama_data in data.get("ramas", []):
        rama_name = rama_data.get("rama", "General")
        branch = Branch(company_id=company.id, rama=rama_name)
        db.add(branch)
        db.flush()

        for plan_data in rama_data.get("planes", []):
            plan = Plan(
                branch_id=branch.id,
                nombre_plan=plan_data.get("nombre_plan", ""),
                variante=plan_data.get("variante"),
                particularidades=plan_data.get("particularidades"),
            )
            db.add(plan)
            db.flush()

            for key, valor in plan_data.get("coberturas", {}).items():
                label = LABEL_MAP.get(key, key.replace("_", " ").title())
                cov = Coverage(plan_id=plan.id, campo_clave=key, campo_label=label, valor=str(valor))
                db.add(cov)

    db.commit()


def log_sync(msg: str, kind: str = "info"):
    sync_state["progress"].append({"msg": msg, "type": kind})


def _company_has_data(db, company_name: str) -> bool:
    """True if the company already has at least one extracted plan (so we can skip it)."""
    company = db.query(Company).filter(Company.nombre == company_name).first()
    if not company:
        return False
    for branch in company.branches:
        if branch.plans:
            return True
    return False


def run_sync(db_session_factory, force: bool = False):
    from .database import SessionLocal
    db = SessionLocal()
    try:
        sync_state["running"] = True
        sync_state["progress"] = []
        sync_state["error"] = None
        sync_state["done"] = 0
        sync_state["total"] = 0
        sync_state["current"] = ""
        sync_state["started_at"] = datetime.utcnow().isoformat()
        sync_state["finished_at"] = None

        # 1. Sync Drive folders
        log_sync("🔌 Conectando con Google Drive...", "info")
        try:
            subfolders = list_subfolders(DRIVE_FOLDER_ID)
            log_sync(f"✓ Drive: {len(subfolders)} carpetas encontradas", "ok")
        except Exception as e:
            log_sync(f"✗ Error Drive: {e}", "error")
            subfolders = []

        # 2. URL-based companies
        url_companies = db.query(Company).filter(Company.fuente == "url", Company.activa == True).all()
        if url_companies:
            log_sync(f"🌐 {len(url_companies)} compañía(s) por URL", "info")

        # URL sources first (e.g. LES / La Equidad) so they're guaranteed to load
        # before the larger Drive batch can exhaust the daily AI quota.
        all_tasks = [("url", c) for c in url_companies] + [("drive", f) for f in subfolders]
        sync_state["total"] = len(all_tasks)
        drive_names_in_drive = {f["name"].upper() for f in subfolders}

        for fuente, item in all_tasks:
            try:
                if fuente == "drive":
                    company_name = item["name"]
                    folder_id = item["id"]
                    sync_state["current"] = company_name

                    company = db.query(Company).filter(Company.nombre == company_name).first()
                    if not company:
                        company = Company(nombre=company_name, fuente="drive", drive_folder_id=folder_id)
                        db.add(company)
                        db.commit()
                        db.refresh(company)
                    else:
                        company.drive_folder_id = folder_id
                        company.fuente = "drive"
                        db.commit()

                    if not force and _company_has_data(db, company_name):
                        log_sync(f"⏭ {company_name}: ya cargada, se omite", "info")
                        db.query(Company).filter(Company.nombre == company_name).update({"activa": True})
                        db.commit()
                        sync_state["done"] += 1
                        continue

                    log_sync(f"⏳ {company_name}: descargando manual...", "processing")
                    pdf_path = get_file_content_as_pdf_path(folder_id, company_name)
                    if not pdf_path:
                        log_sync(f"⚠ {company_name}: sin archivo en la carpeta", "error")
                        sync_state["done"] += 1
                        continue

                    log_sync(f"🤖 {company_name}: analizando con IA...", "processing")
                    data = extract_from_pdf(pdf_path, company_name)
                    save_extracted_data(data, db)
                    db.query(Company).filter(Company.nombre == company_name).update({"activa": True})
                    db.commit()

                    ramas = [r.get("rama","?") for r in data.get("ramas",[])]
                    log_sync(f"✓ {company_name}: OK ({', '.join(ramas)})", "ok")
                    log = SyncLog(company_nombre=company_name, accion="updated", detalle="Drive sync")
                    db.add(log)
                    db.commit()

                else:
                    company = item
                    sync_state["current"] = company.nombre
                    if not force and _company_has_data(db, company.nombre):
                        log_sync(f"⏭ {company.nombre}: ya cargada, se omite", "info")
                        sync_state["done"] += 1
                        continue
                    log_sync(f"⏳ {company.nombre}: leyendo URL...", "processing")
                    text = extract_text_from_url(company.url_manual)
                    log_sync(f"🤖 {company.nombre}: analizando con IA...", "processing")
                    data = extract_from_text(text, company.nombre)
                    save_extracted_data(data, db)
                    db.query(Company).filter(Company.id == company.id).update({
                        "ultima_sync": datetime.utcnow(), "activa": True,
                    })
                    db.commit()
                    ramas = [r.get("rama","?") for r in data.get("ramas",[])]
                    log_sync(f"✓ {company.nombre}: OK ({', '.join(ramas)})", "ok")
                    log = SyncLog(company_nombre=company.nombre, accion="updated", detalle="URL sync")
                    db.add(log)
                    db.commit()

            except QuotaExhaustedError:
                log_sync(
                    "🛑 Cuota diaria de IA (Gemini) agotada. Lo cargado quedó guardado. "
                    "Volvé a tocar Sincronizar después del reset diario (medianoche hora del Pacífico).",
                    "error",
                )
                break

            except Exception as e:
                name = item["name"] if fuente == "drive" else item.nombre
                log_sync(f"✗ {name}: ERROR — {str(e)[:120]}", "error")
                log = SyncLog(company_nombre=name, accion="error", detalle=str(e))
                db.add(log)
                db.commit()

            sync_state["done"] += 1

        # Deactivate Drive companies no longer in Drive
        all_drive_companies = db.query(Company).filter(Company.fuente == "drive").all()
        for c in all_drive_companies:
            if c.nombre.upper() not in drive_names_in_drive:
                c.activa = False
                db.commit()
                log = SyncLog(company_nombre=c.nombre, accion="deactivated", detalle="No encontrada en Drive")
                db.add(log)
                db.commit()

        ok = sum(1 for p in sync_state["progress"] if p["type"] == "ok")
        err = sum(1 for p in sync_state["progress"] if p["type"] == "error")
        log_sync(f"✅ Sincronización completada — {ok} OK / {err} con errores", "ok")
        sync_state["current"] = ""
        sync_state["running"] = False
        sync_state["finished_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        sync_state["error"] = str(e)
        sync_state["running"] = False
        sync_state["finished_at"] = datetime.utcnow().isoformat()
        log_sync(f"✗ Error fatal: {e}", "error")
    finally:
        db.close()


@app.post("/sync")
def trigger_sync(force: bool = False, _=Depends(verify_token)):
    if sync_state["running"]:
        raise HTTPException(status_code=409, detail="Sincronización en curso")
    thread = threading.Thread(target=run_sync, args=(None,), kwargs={"force": force}, daemon=True)
    thread.start()
    modo = "completa" if force else "incremental"
    return {"message": f"Sincronización {modo} iniciada"}


@app.get("/sync/status")
def sync_status(_=Depends(verify_token)):
    return {
        "running": sync_state["running"],
        "total": sync_state["total"],
        "done": sync_state["done"],
        "current": sync_state["current"],
        "progress": sync_state["progress"],   # full log, frontend paginates
        "error": sync_state["error"],
        "started_at": sync_state["started_at"],
        "finished_at": sync_state["finished_at"],
    }


@app.get("/sync/logs")
def sync_logs(db: Session = Depends(get_db), _=Depends(verify_token)):
    logs = db.query(SyncLog).order_by(SyncLog.timestamp.desc()).limit(50).all()
    return [{"timestamp": l.timestamp, "company": l.company_nombre, "accion": l.accion, "detalle": l.detalle} for l in logs]


# ─── Pages ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(str(FRONTEND_DIR / "dashboard.html"))


@app.on_event("startup")
def startup():
    init_db()
    # Seed LES company if not exists
    from .database import SessionLocal
    db = SessionLocal()
    try:
        les = db.query(Company).filter(Company.nombre == "LES").first()
        if not les:
            les = Company(nombre="LES", fuente="url", url_manual="https://manual-les.netlify.app/")
            db.add(les)
            db.commit()
    finally:
        db.close()
