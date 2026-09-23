"""Encode unchanged AAC packets while the full video VAE is running."""
import json
import os
import threading
import time

import torch
from vllm.logger import init_logger
from vllm_omni.diffusion.h3_profile_nvtx import profile_range

from .chunked_cpu_output import encode_minimax_h3_aac


class AsyncAudioEncode:
    def __init__(self, audio, producer_done, sample_rate, request):
        if not isinstance(audio, torch.Tensor) or not audio.is_cuda:
            raise ValueError("Early AAC requires a CUDA audio waveform")
        self.request = request
        self.gain_db = float(os.environ.get('H3_OUTPUT_AUDIO_GAIN_DB', '0'))
        self.bitrate = int(os.environ['H3_OUTPUT_AUDIO_BITRATE']) if os.environ.get('H3_OUTPUT_AUDIO_BITRATE') else None
        self.sample_rate = sample_rate
        self.result = None
        self.error = None
        self.encode_seconds = 0.0
        self.stream = torch.cuda.Stream(device=audio.device)
        self.ready = torch.cuda.Event()
        self.host = torch.empty_like(audio, device="cpu", dtype=torch.float32, pin_memory=True)
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(producer_done)
                audio.record_stream(self.stream)
                source = audio.detach().float()
                source.record_stream(self.stream)
                self.host.copy_(source, non_blocking=True)
                self.ready.record(self.stream)
        except BaseException:
            self.stream.synchronize()
            raise
        self.thread = threading.Thread(target=self._run, name="h3-early-aac", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.ready.synchronize()
            raise

    def _run(self):
        try:
            self.ready.synchronize()
            start = time.perf_counter()
            with profile_range("output.early_aac_encode"):
                self.result = encode_minimax_h3_aac(self.host, self.sample_rate,
                                                   gain_db=self.gain_db, bitrate=self.bitrate)
            self.encode_seconds = time.perf_counter() - start
        except BaseException as error:
            self.error = error

    def finish(self):
        start = time.perf_counter()
        self.thread.join()
        join_seconds = time.perf_counter() - start
        if self.error is not None:
            raise RuntimeError("Early AAC worker failed after video VAE drained") from self.error
        if self.result is None:
            raise RuntimeError("Early AAC worker produced no packets")
        init_logger(__name__).info("H3_FULL_VAE_EARLY_AAC %s", json.dumps(dict(
            request=self.request, rank=0, packets=len(self.result.packets),
            packet_bytes=sum(len(p.data) for p in self.result.packets),
            encode_seconds=self.encode_seconds, join_seconds=join_seconds,
            sample_rate=self.sample_rate, original_codec=True, gain_db=self.gain_db,
            bitrate=self.bitrate, status="PASS"), sort_keys=True))
        return self.result

    def drain(self):
        self.thread.join()
