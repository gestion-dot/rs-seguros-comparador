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
from .extractor import extract_from_pdf, extract_from_text, QuotaExhaustedError, COBERTURAS_VEHICULO

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
            "nombre": c.nombre_oficial or c.nombre,
            "fuente": c.fuente,
            "logo_url": c.logo_url,
            "inspeccion": c.inspeccion,
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


def _rama_es_vehiculo(rama: str) -> bool:
    n = _normalizar(rama)
    return "auto" in n or "moto" in n or "vehiculo" in n


def _normalizar(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


# Grupos canónicos de planes para Autos/Motos (orden = prioridad de clasificación)
GRUPOS_VEHICULO = ["Todo Riesgo", "Garage", "Terceros Completo", "Todo/Total", "RC"]


def clasificar_grupo(nombre_plan: str) -> Optional[str]:
    """Mapea el nombre comercial de un plan a uno de los grupos canónicos."""
    n = _normalizar(nombre_plan)
    if not n:
        return None
    if "todo riesgo" in n or "todoriesgo" in n or "todo-riesgo" in n:
        return "Todo Riesgo"
    if "garage" in n or "garaje" in n:
        return "Garage"
    if "terceros completo" in n or "tercero completo" in n or "terc. completo" in n or "completo" in n:
        return "Terceros Completo"
    if "total" in n:
        return "Todo/Total"
    if (
        "rc" in n.split()
        or "responsabilidad civil" in n
        or "terceros" in n
        or "tercero" in n
        or "basic" in n
    ):
        return "RC"
    return None


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


@app.get("/grupos")
def get_grupos(rama: str, db: Session = Depends(get_db), _=Depends(verify_token)):
    """Grupos canónicos para una rama de vehículos.

    Por premisa de negocio, TODAS las compañías ofrecen estos grupos en Autos/Motos;
    si en la base falta alguno para una compañía, es porque falta el manual (se muestra
    "FALTA MANUAL PRODUCTO" en la comparativa), no porque no exista el producto.
    """
    if not _rama_es_vehiculo(rama):
        return []
    return ["RC", "Garage", "Todo/Total", "Terceros Completo", "Todo Riesgo"]


FALTA_MANUAL = "FALTA MANUAL PRODUCTO"

# Coberturas estándar (para mostrar filas aunque ninguna compañía tenga datos)
COBERTURAS_ESTANDAR = {
    "responsabilidad_civil": "Responsabilidad Civil",
    "robo_hurto": "Robo y/o Hurto",
    "incendio": "Incendio",
    "destruccion_total": "Destrucción Total",
    "danos_parciales": "Daños Parciales",
    "granizo": "Granizo",
    "cristales_cerraduras": "Cristales y Cerraduras",
    "auxilio_mecanico": "Auxilio Mecánico / Remolque",
}


@app.get("/compare")
def compare(
    rama: str,
    companies: str,
    plan: Optional[str] = None,
    grupo: Optional[str] = None,
    grupos: Optional[str] = None,
    variante: Optional[str] = None,
    db: Session = Depends(get_db),
    _=Depends(verify_token),
):
    """Compara compañías por grupo(s) (una columna por plan que encuadre) o por plan exacto.

    En Autos/Motos, una compañía seleccionada que no tenga plan en los grupos pedidos
    igual aparece como columna con "FALTA MANUAL PRODUCTO" (premisa: el producto existe,
    falta cargar el manual).
    """
    company_ids = [int(x) for x in companies.split(",") if x.strip()]
    grupos_sel = [g.strip() for g in (grupos or grupo or "").split(",") if g.strip()]
    es_veh = _rama_es_vehiculo(rama)

    columns = []            # columnas con datos
    sin_datos = []          # (cid, company) sin plan en los grupos pedidos
    all_coverage_keys = {}  # key -> label
    inspeccion_by_cid = {}  # cid -> modo de inspección (manual)

    for cid in company_ids:
        company = db.query(Company).filter(Company.id == cid).first()
        if not company:
            continue
        inspeccion_by_cid[cid] = company.inspeccion

        branch = db.query(Branch).filter(Branch.company_id == cid, Branch.rama == rama).first()
        seleccion = []
        if branch:
            candidatos = db.query(Plan).filter(Plan.branch_id == branch.id).all()
            if grupos_sel:
                seleccion = [p for p in candidatos if (p.grupo or clasificar_grupo(p.nombre_plan)) in grupos_sel]
            elif plan:
                seleccion = [p for p in candidatos if p.nombre_plan == plan]
            else:
                seleccion = candidatos
            if variante:
                seleccion = [p for p in seleccion if p.variante == variante] or seleccion

        if seleccion:
            for p in seleccion:
                coberturas = {}
                for cov in p.coverages:
                    coberturas[cov.campo_clave] = cov.valor
                    all_coverage_keys[cov.campo_clave] = cov.campo_label
                etiqueta_plan = p.nombre_plan + (f" ({p.variante})" if p.variante else "")
                columns.append({
                    "id": f"{cid}-{p.id}",
                    "company_id": cid,
                    "nombre": company.nombre_oficial or company.nombre,
                    "logo_url": company.logo_url,
                    "plan_real": etiqueta_plan,
                    "particularidades": p.particularidades,
                    "falta_manual": False,
                    "coberturas": coberturas,
                })
        else:
            sin_datos.append((cid, company))

    if es_veh:
        # Autos/Motos: filas FIJAS en orden + fila de Inspección (manual)
        coverage_keys = {k: lbl for k, lbl in COBERTURAS_VEHICULO}
        coverage_keys["inspeccion"] = "Inspección"
        for col in columns:
            insp = inspeccion_by_cid.get(col["company_id"]) or "No especificado"
            nueva = {k: col["coberturas"].get(k, "No especificado") for k, _ in COBERTURAS_VEHICULO}
            nueva["inspeccion"] = insp
            col["coberturas"] = nueva
        # Compañías sin plan en los grupos pedidos → FALTA MANUAL (pero la inspección sí se conoce)
        if grupos_sel:
            for cid, company in sin_datos:
                cob = {k: FALTA_MANUAL for k, _ in COBERTURAS_VEHICULO}
                cob["inspeccion"] = (company.inspeccion or "No especificado")
                columns.append({
                    "id": f"{cid}-nomanual",
                    "company_id": cid,
                    "nombre": company.nombre_oficial or company.nombre,
                    "logo_url": company.logo_url,
                    "plan_real": "",
                    "particularidades": None,
                    "falta_manual": True,
                    "coberturas": cob,
                })
        return {"coverage_keys": coverage_keys, "columns": columns}

    # Otras ramas: claves dinámicas según lo extraído
    for col in columns:
        for key in all_coverage_keys:
            col["coberturas"].setdefault(key, "No incluye")
    return {
        "coverage_keys": all_coverage_keys,
        "columns": columns,
    }


# ─── Sync ────────────────────────────────────────────────────────────────────

def save_extracted_data(data: dict, db: Session, company: Optional[Company] = None):
    fecha = data.get("fecha_actualizacion_manual")

    # Always write into the canonical company record (created from the Drive folder
    # name or the URL company). The AI's returned "compania" name often differs
    # (e.g. folder "ALLIANZ" vs AI "Allianz Argentina"), which would otherwise
    # create a duplicate record and leave the original empty / inactive.
    if company is None:
        company_name = data.get("compania", "Desconocida")
        company = db.query(Company).filter(Company.nombre == company_name).first()
        if company is None:
            company = Company(nombre=company_name)
            db.add(company)
            db.flush()

    # Clear existing branch/plan/coverage data before re-inserting
    for branch in list(company.branches):
        db.delete(branch)
    db.flush()

    company.fecha_manual = fecha
    company.ultima_sync = datetime.utcnow()
    company.activa = True

    LABEL_MAP = dict(COBERTURAS_VEHICULO)

    for rama_data in data.get("ramas", []):
        rama_name = rama_data.get("rama", "General")
        branch = Branch(company_id=company.id, rama=rama_name)
        db.add(branch)
        db.flush()

        for plan_data in rama_data.get("planes", []):
            nombre_plan = plan_data.get("nombre_plan", "")
            grupo = plan_data.get("grupo") or clasificar_grupo(nombre_plan)
            plan = Plan(
                branch_id=branch.id,
                nombre_plan=nombre_plan,
                variante=plan_data.get("variante"),
                grupo=grupo,
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


def run_sync(db_session_factory, force: bool = False, only_company_id: Optional[int] = None):
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

        instructions = None  # el prompt es fijo (lista de coberturas en el extractor)
        drive_names_in_drive = None  # solo se usa en sync completo, para desactivar bajas

        if only_company_id is not None:
            # Sincronización INDIVIDUAL: solo esta compañía (siempre re-extrae)
            force = True
            company = db.query(Company).filter(Company.id == only_company_id).first()
            if not company:
                log_sync("✗ Compañía no encontrada", "error")
                all_tasks = []
            else:
                log_sync(f"🔄 Sincronizando {company.nombre_oficial or company.nombre}...", "info")
                if company.fuente == "url":
                    all_tasks = [("url", company)]
                else:
                    all_tasks = [("drive", {"name": company.nombre, "id": company.drive_folder_id})]
        else:
            # Sincronización COMPLETA
            log_sync("🔌 Conectando con Google Drive...", "info")
            try:
                subfolders = list_subfolders(DRIVE_FOLDER_ID)
                log_sync(f"✓ Drive: {len(subfolders)} carpetas encontradas", "ok")
            except Exception as e:
                log_sync(f"✗ Error Drive: {e}", "error")
                subfolders = []

            url_companies = db.query(Company).filter(Company.fuente == "url", Company.activa == True).all()
            if url_companies:
                log_sync(f"🌐 {len(url_companies)} compañía(s) por URL", "info")

            # URL primero, luego Drive
            all_tasks = [("url", c) for c in url_companies] + [("drive", f) for f in subfolders]
            drive_names_in_drive = {f["name"].upper() for f in subfolders}

        sync_state["total"] = len(all_tasks)

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
                    data = extract_from_pdf(pdf_path, company_name, instructions)
                    save_extracted_data(data, db, company=company)
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
                    data = extract_from_text(text, company.nombre, instructions)
                    save_extracted_data(data, db, company=company)
                    db.commit()
                    ramas = [r.get("rama","?") for r in data.get("ramas",[])]
                    log_sync(f"✓ {company.nombre}: OK ({', '.join(ramas)})", "ok")
                    log = SyncLog(company_nombre=company.nombre, accion="updated", detalle="URL sync")
                    db.add(log)
                    db.commit()

            except QuotaExhaustedError:
                log_sync(
                    "🛑 Cuota diaria de IA agotada. Lo cargado quedó guardado. "
                    "Volvé a sincronizar más tarde.",
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

        # Deactivate Drive companies no longer in Drive (solo en sync completo)
        if drive_names_in_drive is not None:
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


@app.post("/sync/company/{company_id}")
def trigger_sync_one(company_id: int, _=Depends(verify_token)):
    """Sincroniza (re-analiza) UNA sola compañía."""
    if sync_state["running"]:
        raise HTTPException(status_code=409, detail="Sincronización en curso")
    thread = threading.Thread(target=run_sync, args=(None,), kwargs={"only_company_id": company_id}, daemon=True)
    thread.start()
    return {"message": "Sincronización individual iniciada"}


class InspeccionBody(BaseModel):
    inspeccion: Optional[str] = None


@app.patch("/companies/{company_id}/inspeccion")
def set_inspeccion(company_id: int, body: InspeccionBody, db: Session = Depends(get_db), _=Depends(verify_token)):
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404)
    company.inspeccion = body.inspeccion or None
    db.commit()
    return {"ok": True, "inspeccion": company.inspeccion}


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
