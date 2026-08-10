"""
Viewer backend.

Two endpoints are all the front end needs:

    GET  /api/problems        every registered problem + its parameter schema
    POST /api/solve           {slug, values} -> solved trajectory

The UI is generated from the schema, so this file never changes when a new
problem module is added.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import registry

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load_all()
    yield


app = FastAPI(title="Starship Landing GNC Viewer", version="0.1.0",
              lifespan=lifespan)


class SolveRequest(BaseModel):
    slug: str
    values: dict[str, Any] = {}


@app.get("/api/problems")
def list_problems() -> dict[str, Any]:
    return {"problems": [p.describe() for p in registry.all_problems()]}


@app.post("/api/solve")
def solve(req: SolveRequest) -> dict[str, Any]:
    try:
        problem = registry.get(req.slug)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown problem: {req.slug}")

    traj = problem.solve(req.values)
    return traj.to_dict()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Static front end last, so it does not shadow the API routes.
if WEB_DIR.is_dir():
    app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")
    app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/style.css")
    def style() -> FileResponse:
        return FileResponse(WEB_DIR / "style.css")
