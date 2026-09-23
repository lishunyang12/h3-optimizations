"""Serve a direct, full-video A/B page for the MiniMax-H3 moon scene."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/lustre/raplab/client/sylarl/minimax-h3-native")
CASE = (
    "vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-"
    "qkv-e4m3-all-main-fp8-o-bundle-taeh3-fp16-persistent-rdma-cross-rank-audio"
)
DEFAULT_STANDARD = (
    WORKSPACE
    / "results/h3-moon-teahouse-standard-vsa4-bf16-fullprompt-retry2-20260908/"
    "vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct/out_t2va1.mp4"
)
DEFAULT_OPTIMIZED = (
    WORKSPACE
    / "results/h3-moon-teahouse-fullprompt-optimized-d6-single-20260908"
    / CASE
    / "out_t2va1.mp4"
)


def env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).resolve()


STANDARD_VIDEO = env_path("H3_COMPARE_STANDARD_VIDEO", DEFAULT_STANDARD)
OPTIMIZED_VIDEO = env_path("H3_COMPARE_OPTIMIZED_VIDEO", DEFAULT_OPTIMIZED)
STANDARD_E2E = float(os.getenv("H3_COMPARE_STANDARD_E2E", "27.995"))
OPTIMIZED_E2E = float(os.getenv("H3_COMPARE_OPTIMIZED_E2E", "0"))
MEDIA_SECONDS = float(os.getenv("H3_COMPARE_MEDIA_SECONDS", "15.083333"))

app = FastAPI(title="MiniMax-H3 video A/B", docs_url=None, redoc_url=None)


def require_video(path: Path) -> Path:
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"comparison video is not ready: {path.name}",
        )
    return path


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "static/index.html")


@app.get("/health")
async def health() -> dict[str, object]:
    ready = STANDARD_VIDEO.is_file() and OPTIMIZED_VIDEO.is_file()
    return {
        "status": "ok" if ready else "preparing",
        "standard_ready": STANDARD_VIDEO.is_file(),
        "optimized_ready": OPTIMIZED_VIDEO.is_file(),
    }


@app.get("/api/comparison")
async def comparison() -> dict[str, object]:
    standard = require_video(STANDARD_VIDEO)
    optimized = require_video(OPTIMIZED_VIDEO)
    return {
        "scene": "Moon Teahouse Signal",
        "media_seconds": MEDIA_SECONDS,
        "standard": {
            "label": "Standard VSA + FastH3 4-step",
            "detail": "BF16 DiT/QKV · standard H3 VAE",
            "e2e_seconds": STANDARD_E2E,
            "rtf": round(STANDARD_E2E / MEDIA_SECONDS, 4),
            "size_bytes": standard.stat().st_size,
            "url": "/video/standard.mp4",
        },
        "optimized": {
            "label": "Optimized <15 s stack",
            "detail": "FP8 DiT · E4M3 QKV · O-bundle · TAEH3 FP16",
            "e2e_seconds": OPTIMIZED_E2E,
            "rtf": (
                round(OPTIMIZED_E2E / MEDIA_SECONDS, 4)
                if OPTIMIZED_E2E > 0
                else None
            ),
            "size_bytes": optimized.stat().st_size,
            "url": "/video/optimized.mp4",
        },
    }


@app.get("/video/standard.mp4")
@app.head("/video/standard.mp4")
async def standard_video() -> FileResponse:
    return FileResponse(require_video(STANDARD_VIDEO), media_type="video/mp4")


@app.get("/video/optimized.mp4")
@app.head("/video/optimized.mp4")
async def optimized_video() -> FileResponse:
    return FileResponse(require_video(OPTIMIZED_VIDEO), media_type="video/mp4")
