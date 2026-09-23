"""Single-session continuous MiniMax-H3 demonstration gateway.

The diffusion server deliberately stays bound to localhost.  This process is
the small public surface: it maintains a generation queue, serves completed
MP4 clips, and exposes only the calibrated preset by default.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = Path("/lustre/raplab/client/sylarl/minimax-h3-native")
DEFAULT_PROMPT = HERE.parent / "prompts" / "laundry_balcony_glowfish_15s.txt"
DEFAULT_BOOTSTRAP = (
    DEFAULT_WORKSPACE / "results/h3-prompt-laundry-d6-repeat3-kmax280-20260908/"
    "vllm-omni-tp1-sp8-fasth3-vsa-flashinfer-topk162-rdma-qk-direct-qkv-e4m3-"
    "all-main-fp8-o-bundle-taeh3-fp16-persistent-rdma-cross-rank-audio/"
    "out_t2va1-repeat-3.mp4"
)
CALIBRATED_SEED = 1101
MAX_PROMPT_CHARS = 4096


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    backend_url: str
    runtime_dir: Path
    prompt_path: Path
    bootstrap_clip: Path
    bootstrap_generation_s: float
    queue_depth: int
    bootstrap_copies: int
    idle_timeout_s: float
    request_timeout_s: float
    generation_enabled: bool
    allow_custom: bool
    max_files: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            backend_url=os.getenv("H3_DEMO_BACKEND_URL", "http://127.0.0.1:8091"),
            runtime_dir=Path(
                os.getenv(
                    "H3_DEMO_RUNTIME_DIR",
                    str(DEFAULT_WORKSPACE / "services/minimax-h3-live-demo"),
                )
            ),
            prompt_path=Path(os.getenv("H3_DEMO_PROMPT_PATH", str(DEFAULT_PROMPT))),
            bootstrap_clip=Path(os.getenv("H3_DEMO_BOOTSTRAP_CLIP", str(DEFAULT_BOOTSTRAP))),
            bootstrap_generation_s=max(0.0, float(os.getenv("H3_DEMO_BOOTSTRAP_GENERATION_S", "14.680"))),
            queue_depth=max(2, int(os.getenv("H3_DEMO_QUEUE_DEPTH", "3"))),
            bootstrap_copies=max(0, int(os.getenv("H3_DEMO_BOOTSTRAP_COPIES", "2"))),
            idle_timeout_s=max(15.0, float(os.getenv("H3_DEMO_IDLE_TIMEOUT_S", "90"))),
            request_timeout_s=max(30.0, float(os.getenv("H3_DEMO_REQUEST_TIMEOUT_S", "900"))),
            generation_enabled=env_flag("H3_DEMO_GENERATION_ENABLED", True),
            allow_custom=env_flag("H3_DEMO_ALLOW_CUSTOM", False),
            max_files=max(4, int(os.getenv("H3_DEMO_MAX_FILES", "12"))),
        )


@dataclass
class Clip:
    id: str
    sequence: int
    prompt: str
    prompt_sha256: str
    media_seconds: float
    generation_seconds: float | None
    rtf: float | None
    created_at: float
    source: str
    qualified: bool
    path: Path
    seed: int = CALIBRATED_SEED
    content_sha256: str | None = None

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("path")
        value["url"] = f"/media/{self.id}.mp4"
        return value


class EnqueueRequest(BaseModel):
    prompt: str = Field(default="", max_length=MAX_PROMPT_CHARS)


class DemoState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.preset_prompt = settings.prompt_path.read_text(encoding="utf-8").rstrip("\n")
        self.prompt_sha256 = hashlib.sha256(self.preset_prompt.encode()).hexdigest()
        self.lock = asyncio.Lock()
        self.ready: deque[Clip] = deque()
        self.files: dict[str, Clip] = {}
        self.consumed_generated: deque[str] = deque()
        self.custom_prompts: deque[str] = deque()
        self.sequence = 0
        self.manual_backlog = 0
        self.generating_prompt: str | None = None
        self.generation_started_monotonic: float | None = None
        self.generation_started_at: float | None = None
        self.last_error: str | None = None
        self.last_warmup: dict[str, Any] | None = None
        self.last_completed_clip: Clip | None = None
        self.recent_live_timings: deque[tuple[float, float]] = deque(maxlen=12)
        self.live_completed_segments = 0
        self.delivered_live_segments = 0
        self.failed_generation_attempts = 0
        self.generated_content_digests: set[str] = set()
        self.first_live_seen = False
        self.last_heartbeat = 0.0
        self.worker: asyncio.Task[None] | None = None
        self.started_at = time.time()

    async def initialize(self) -> None:
        self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings.bootstrap_clip.is_file():
            return
        bootstrap_digest = hashlib.sha256(self.settings.bootstrap_clip.read_bytes()).hexdigest()
        for _ in range(self.settings.bootstrap_copies):
            self.sequence += 1
            clip = Clip(
                id=f"bootstrap-{self.sequence}",
                sequence=self.sequence,
                prompt=self.preset_prompt,
                prompt_sha256=self.prompt_sha256,
                media_seconds=15.083333,
                generation_seconds=self.settings.bootstrap_generation_s,
                rtf=round(self.settings.bootstrap_generation_s / 15.083333, 4),
                created_at=time.time(),
                source="qualified-prebuilt-buffer",
                qualified=self.settings.bootstrap_generation_s < 15.083333,
                path=self.settings.bootstrap_clip,
                content_sha256=bootstrap_digest,
            )
            self.ready.append(clip)
            self.files[clip.id] = clip
            self.last_completed_clip = clip

    async def heartbeat(self) -> None:
        async with self.lock:
            self.last_heartbeat = time.monotonic()

    async def enqueue(self, prompt: str) -> dict[str, Any]:
        prompt = prompt.strip() or self.preset_prompt
        if prompt != self.preset_prompt and not self.settings.allow_custom:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The <15 s lane is bound to the calibrated preset. "
                    "Custom prompts require the separately served BF16-QKV compatibility lane."
                ),
            )
        async with self.lock:
            if self.manual_backlog + len(self.custom_prompts) >= 8:
                raise HTTPException(status_code=429, detail="generation queue is full")
            if prompt == self.preset_prompt:
                self.manual_backlog += 1
            else:
                self.custom_prompts.append(prompt)
            self.last_heartbeat = time.monotonic()
            return {"accepted": True, "position": self.manual_backlog + len(self.custom_prompts)}

    async def consume(self, clip_id: str) -> None:
        async with self.lock:
            match = next((clip for clip in self.ready if clip.id == clip_id), None)
            if match is None:
                return
            self.ready.remove(match)
            if match.source == "live-generated":
                self.consumed_generated.append(match.id)
            self.last_heartbeat = time.monotonic()
            self._collect_files_locked()

    def _collect_files_locked(self) -> None:
        while len(self.consumed_generated) > self.settings.max_files:
            clip_id = self.consumed_generated.popleft()
            clip = self.files.pop(clip_id, None)
            if clip is not None:
                clip.path.unlink(missing_ok=True)

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            ready = [clip.public() for clip in self.ready]
            latest = self.last_completed_clip.public() if self.last_completed_clip else None
            now_monotonic = time.monotonic()
            active = now_monotonic - self.last_heartbeat <= self.settings.idle_timeout_s
            generation_elapsed = (
                round(now_monotonic - self.generation_started_monotonic, 3)
                if self.generation_started_monotonic is not None
                else None
            )
            buffer_seconds = round(sum(clip.media_seconds for clip in self.ready), 3)
            recent_generation_seconds = sum(item[0] for item in self.recent_live_timings)
            recent_media_seconds = sum(item[1] for item in self.recent_live_timings)
            production_rate = (
                round(recent_media_seconds / recent_generation_seconds, 4)
                if recent_generation_seconds > 0
                else round(15.083333 / self.settings.bootstrap_generation_s, 4)
            )
            last_generation_seconds = latest["generation_seconds"] if latest else None
            last_media_seconds = latest["media_seconds"] if latest else None
            last_headroom_seconds = (
                round(last_media_seconds - last_generation_seconds, 3)
                if last_generation_seconds is not None and last_media_seconds is not None
                else None
            )
            return {
                "service": "H3 Live",
                "mode": "calibrated-preset",
                "backend_url": self.settings.backend_url,
                "generation_enabled": self.settings.generation_enabled,
                "allow_custom": self.settings.allow_custom,
                "viewer_active": active,
                "preset": {
                    "id": "laundry-balcony-glowfish",
                    "title": "Glowfish After Midnight",
                    "prompt": self.preset_prompt,
                    "prompt_sha256": self.prompt_sha256,
                    "duration_seconds": 15.083333,
                    "target_rtf": 1.0,
                    "seed": CALIBRATED_SEED,
                    "deterministic": True,
                },
                "generation_queue": {
                    "active": self.generating_prompt is not None,
                    "pending": self.manual_backlog + len(self.custom_prompts),
                    "started_at": self.generation_started_at,
                    "elapsed_seconds": generation_elapsed,
                },
                "playout_queue": ready,
                "latest": latest,
                "metrics": {
                    "last_generation_seconds": last_generation_seconds,
                    "last_media_seconds": last_media_seconds,
                    "last_rtf": latest["rtf"] if latest else None,
                    "last_headroom_seconds": last_headroom_seconds,
                    "production_rate_x": production_rate,
                    "playout_buffer_seconds": buffer_seconds,
                    "ready_segments": len(ready),
                    "live_completed_segments": self.live_completed_segments,
                    "delivered_live_segments": self.delivered_live_segments,
                    "failed_generation_attempts": self.failed_generation_attempts,
                    "unique_generated_outputs": len(self.generated_content_digests),
                    "window_segments": len(self.recent_live_timings),
                },
                "last_warmup": self.last_warmup,
                "last_error": self.last_error,
                "uptime_seconds": round(time.time() - self.started_at, 1),
            }

    async def worker_loop(self) -> None:
        while True:
            try:
                await self._worker_iteration()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the demo alive while the backend restarts
                async with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(3)

    async def _worker_iteration(self) -> None:
        restore_manual = False
        restore_custom = False
        async with self.lock:
            active = time.monotonic() - self.last_heartbeat <= self.settings.idle_timeout_s
            desired = self.settings.queue_depth + self.manual_backlog + len(self.custom_prompts)
            if (
                not self.settings.generation_enabled
                or not active
                or self.generating_prompt is not None
                or len(self.ready) >= desired
            ):
                prompt = None
            elif self.custom_prompts:
                prompt = self.custom_prompts.popleft()
                restore_custom = True
                self.generating_prompt = prompt
            else:
                prompt = self.preset_prompt
                self.generating_prompt = prompt
                if self.manual_backlog:
                    self.manual_backlog -= 1
                    restore_manual = True
            if prompt is not None:
                self.generation_started_monotonic = time.monotonic()
                self.generation_started_at = time.time()
        if prompt is None:
            await asyncio.sleep(0.25)
            return
        try:
            clip = await self._generate(prompt)
        except Exception:
            async with self.lock:
                if restore_custom:
                    self.custom_prompts.appendleft(prompt)
                elif restore_manual:
                    self.manual_backlog += 1
                self.generating_prompt = None
                self.generation_started_monotonic = None
                self.generation_started_at = None
                self.failed_generation_attempts += 1
            raise
        async with self.lock:
            is_preset = prompt == self.preset_prompt
            discard_cold_warmup = is_preset and not self.first_live_seen and clip.rtf is not None and clip.rtf >= 1.0
            self.live_completed_segments += 1
            self.last_completed_clip = clip
            if clip.content_sha256 is not None:
                self.generated_content_digests.add(clip.content_sha256)
            if is_preset:
                self.first_live_seen = True
            if discard_cold_warmup:
                self.last_warmup = {
                    "generation_seconds": clip.generation_seconds,
                    "media_seconds": clip.media_seconds,
                    "rtf": clip.rtf,
                    "reason": "one-time compile/JIT warmup; excluded from playout",
                }
                clip.path.unlink(missing_ok=True)
            else:
                self.ready.append(clip)
                self.files[clip.id] = clip
                self.delivered_live_segments += 1
                if clip.generation_seconds is not None:
                    self.recent_live_timings.append((clip.generation_seconds, clip.media_seconds))
            self.generating_prompt = None
            self.generation_started_monotonic = None
            self.generation_started_at = None
            self.last_error = None

    async def _generate(self, prompt: str) -> Clip:
        form = {
            "prompt": prompt,
            "width": "1280",
            "height": "720",
            "fps": "24",
            "num_inference_steps": "4",
            "flow_shift": "12",
            "seed": str(CALIBRATED_SEED),
            "aspect_ratio": "16:9",
            "quality": "lossless",
            "extra_params": json.dumps(
                {"task": "t2va", "duration": 15, "audio_flow_shift": 3.0},
                separators=(",", ":"),
            ),
        }
        multipart = {key: (None, value) for key, value in form.items()}
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_s) as client:
            response = await client.post(f"{self.settings.backend_url.rstrip('/')}/v1/videos/sync", files=multipart)
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        if len(response.content) < 1024:
            raise RuntimeError("backend returned an implausibly small video payload")

        self.sequence += 1
        digest = hashlib.sha256(response.content).hexdigest()
        clip_id = f"live-{self.sequence}-{digest[:10]}"
        partial = self.settings.runtime_dir / f".{clip_id}.partial"
        output = self.settings.runtime_dir / f"{clip_id}.mp4"
        partial.write_bytes(response.content)
        partial.replace(output)
        media_seconds = await probe_duration(output)
        return Clip(
            id=clip_id,
            sequence=self.sequence,
            prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            media_seconds=media_seconds,
            generation_seconds=round(elapsed, 3),
            rtf=round(elapsed / media_seconds, 4),
            created_at=time.time(),
            source="live-generated",
            qualified=prompt == self.preset_prompt and elapsed < media_seconds,
            path=output,
            content_sha256=digest,
        )


async def probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 15.083333
    process = await asyncio.create_subprocess_exec(
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"ffprobe failed: {stderr.decode(errors='replace').strip()}")
    duration = float(stdout.decode().strip())
    if duration <= 0:
        raise RuntimeError("ffprobe returned a non-positive duration")
    return duration


SETTINGS = Settings.from_env()
STATE = DemoState(SETTINGS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await STATE.initialize()
    STATE.worker = asyncio.create_task(STATE.worker_loop(), name="h3-demo-producer")
    try:
        yield
    finally:
        if STATE.worker is not None:
            STATE.worker.cancel()
            await asyncio.gather(STATE.worker, return_exceptions=True)


app = FastAPI(title="H3 Live", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "static/index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
async def state() -> dict[str, Any]:
    return await STATE.snapshot()


@app.post("/api/heartbeat")
async def heartbeat() -> dict[str, bool]:
    await STATE.heartbeat()
    return {"ok": True}


@app.post("/api/enqueue")
async def enqueue(request: EnqueueRequest) -> dict[str, Any]:
    return await STATE.enqueue(request.prompt)


@app.post("/api/clips/{clip_id}/consumed")
async def consumed(clip_id: str) -> dict[str, bool]:
    await STATE.consume(clip_id)
    return {"ok": True}


@app.get("/media/{clip_id}.mp4")
@app.head("/media/{clip_id}.mp4")
async def media(clip_id: str) -> FileResponse:
    async with STATE.lock:
        clip = STATE.files.get(clip_id)
    if clip is None or not clip.path.is_file():
        raise HTTPException(status_code=404, detail="clip not found")
    return FileResponse(clip.path, media_type="video/mp4")
