"""
AlterScore FastAPI Application
================================
Entrypoint: starts the server, initialises DB, warms up the model,
and serves the React frontend as static files.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.database import init_db
from backend.routes import router
from ml.model import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + warm up model."""
    logger.info("=== AlterScore starting up ===")
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    init_db()
    logger.info("Database initialised.")
    try:
        model = get_model()
        logger.info(f"Model warm-up complete. Loaded: {model._loaded}")
    except Exception as e:
        logger.error(f"Model warm-up failed: {e}")
    yield
    logger.info("=== AlterScore shutting down ===")


app = FastAPI(
    title="AlterScore API",
    description="AI-driven alternate credit scoring for thin-file and no-file borrowers",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api/v1")

# Serve React frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))

    @app.get("/{path_name:path}", include_in_schema=False)
    async def serve_spa(path_name: str):
        index = frontend_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"error": "Frontend not found"}
