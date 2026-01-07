import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


app = FastAPI(title="EVE Trader Data Export")


def _export_path() -> Path:
    return Path(os.getenv("EXPORT_FILE", "/app/data/training_data_cleaned.parquet"))


@app.get("/health")
def health():
    p = _export_path()
    return {"ok": True, "file": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0}


@app.get("/training_data_cleaned.parquet")
def download_training_data():
    p = _export_path()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Missing export file: {p}")
    return FileResponse(path=str(p), media_type="application/octet-stream", filename=p.name)
