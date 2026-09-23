# H3 Live demo

This is a deliberately small, single-session demonstration client for the
qualified MiniMax-H3 D6 service.  It follows the same product contract as a
real-time clip service: generation and playout are separate queues, autoplay
consumes completed clips, and a prepared prompt keeps the stream filled while
the viewer is connected.

The public gateway binds to `0.0.0.0:7860`; the vLLM diffusion endpoint remains
private on `127.0.0.1:8091`.  The default path accepts only the prompt whose QKV
E4M3 scales and packing geometry were calibrated.  Set
`H3_DEMO_ALLOW_CUSTOM=1` only when `H3_DEMO_BACKEND_URL` points to a separately
launched BF16-QKV compatibility service.

```bash
tools/minimax_h3/serve_h3_live_demo.sh start
tools/minimax_h3/serve_h3_live_demo.sh status
tools/minimax_h3/serve_h3_live_demo.sh logs
tools/minimax_h3/serve_h3_live_demo.sh stop
```

To start the qualified model and gateway together, use
`tools/minimax_h3/serve_h3_reactor_demo.sh start`; `start-public` also opens
the temporary HTTPS tunnel.

For a temporary public HTTPS URL, place the checksum-verified Cloudflare
`cloudflared` 2026.8.3 binary at
`/lustre/raplab/client/sylarl/minimax-h3-native/tools/cloudflared-2026.8.3`,
then run `serve_h3_live_demo.sh tunnel-start`.  The generated URL is saved in
the runtime directory as `public-url.txt`; `tunnel-stop` closes it.

The initial two entries reference formal measured sample 3 (14.680 s generation
for 15.083333 s of media), which lets muted browser autoplay begin immediately.
Live requests start on the first browser heartbeat and keep a three-clip
playout buffer.  Each live clip reports HTTP end-to-end generation time, probed
media duration, and their ratio (RTF).

If the first live request pays a one-time compile/JIT cost and has RTF greater
than or equal to one, it is recorded as `last_warmup` but never enters the
playout queue.  The producer immediately builds the next segment, so the two
qualified startup clips cover compilation without presenting it as steady
state throughput.
