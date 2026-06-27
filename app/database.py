from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

import os

# Prefer a durable external database (Postgres) if DATABASE_URL is set;
# otherwise fall back to local SQLite (ephemeral on Render free tier).
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # SQLAlchemy needs the "postgresql://" scheme (Neon/Supabase often give "postgres://")
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    _db_path = os.getenv("DATABASE_PATH", "./seguros.db")
    DATABASE_URL = f"sqlite:///{_db_path}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, unique=True, index=True)  # clave interna (= carpeta de Drive)
    nombre_oficial = Column(String, nullable=True)     # nombre comercial para mostrar
    fuente = Column(String, default="drive")  # "drive" o "url"
    drive_folder_id = Column(String, nullable=True)
    url_manual = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)
    fecha_manual = Column(String, nullable=True)
    activa = Column(Boolean, default=True)
    ultima_sync = Column(DateTime, nullable=True)
    branches = relationship("Branch", back_populates="company", cascade="all, delete-orphan")


class Branch(Base):
    __tablename__ = "branches"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    rama = Column(String)
    company = relationship("Company", back_populates="branches")
    plans = relationship("Plan", back_populates="branch", cascade="all, delete-orphan")


class Plan(Base):
    __tablename__ = "plans"
    id = Column(Integer, primary_key=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    nombre_plan = Column(String)
    variante = Column(String, nullable=True)
    grupo = Column(String, nullable=True)  # grupo canónico (RC, Garage, Todo Riesgo, etc.)
    particularidades = Column(Text, nullable=True)
    branch = relationship("Branch", back_populates="plans")
    coverages = relationship("Coverage", back_populates="plan", cascade="all, delete-orphan")


class Coverage(Base):
    __tablename__ = "coverages"
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("plans.id"))
    campo_clave = Column(String)
    campo_label = Column(String)
    valor = Column(Text)
    plan = relationship("Plan", back_populates="coverages")


class SyncLog(Base):
    __tablename__ = "sync_log"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    company_nombre = Column(String)
    accion = Column(String)
    detalle = Column(Text, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    # Migración ligera: agregar columna 'grupo' a tablas ya existentes (Postgres)
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE plans ADD COLUMN IF NOT EXISTS grupo VARCHAR"))
            conn.execute(text("ALTER TABLE companies ADD COLUMN IF NOT EXISTS nombre_oficial VARCHAR"))
            conn.commit()
    except Exception:
        pass  # SQLite u otra DB: la columna ya viene de create_all
