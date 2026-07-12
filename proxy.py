#!/usr/bin/env python3
"""
proxy.py — Priority Queue Proxy for edge-llm-server
=====================================================
Sits between clients and llama-server on port 8082.
Forwards to llama-server on port 8081.

What it does
  - Accepts requests tagged with X-Priority: high / medium / low
    (or inferred from X-Client-Id: chat / moderate / batch)
  - Holds requests in a priority queue: high > medium > low
  - Only forwards when the server has a free slot (semaphore-based,
    no polling gap)
  - Records queue_wait, ttft_at_proxy, token_generation separately
    (ttft_at_proxy = wall-clock from proxy-forward to first token
     received at proxy; NOT equivalent to server-side prompt_eval)

Usage
  # Terminal 1 — llama-server
./build/bin/llama-server \
    -m ~/Qwen2.5-1.5B-Q4_K_M.gguf \
    --port 8081 --parallel 2 --ctx-size 2048 --kv-unified \
    2>&1 | tee reserved_slot__priority_log.txt

  # Terminal 2 — proxy (defaults)
  python3 proxy.py

  # Terminal 2 — proxy (tuned for stress tests)
  python3 proxy.py --arrival-window 0.010 --dispatch-gap 0.005

  # Terminal 2 — proxy (reserve 1 of 2 slots for Chat)
  python3 proxy.py --reserved-chat-slots 1

  # Terminal 3 — evaluation
  python3 eval.py --port 8082 --server-port 8081 \\
      --label proxy_qos --output results_proxy.json

Design notes
  - SLOTS (= --parallel on server) set via --slots CLI arg (default 2).
  - ARRIVAL_WINDOW: dispatcher waits this many seconds after acquiring
    the semaphore before pulling from the queue, so simultaneous
    arrivals in the same burst are all enqueued before the priority
    decision is made.  Under heavy load you may want to increase this
    (e.g. 0.010–0.020s) to capture slower-arriving clients; under
    low load keep it small to avoid adding unnecessary latency.
    Tune with --arrival-window.
  - RESERVED_CHAT_SLOTS: number of physical slots the dispatcher will
    refuse to hand to Moderate/Batch requests, keeping them free for
    Chat even when Chat has nothing queued right now.  This addresses
    the no-preemption gap seen under staggered arrivals, where Chat
    could land behind an already-running long Batch/Moderate job with
    no way to interrupt it.  Configurable via --reserved-chat-slots;
    0 disables reservation and reproduces pre-reservation dispatch
    behaviour exactly.  There is currently no timeout that releases an
    unused reserved slot back to Batch — that policy question is still
    open.
  - Streaming is passed through chunk-by-chunk so TTFT measurement
    in the client remains accurate.
"""

import argparse
import json
import logging
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── CLI arguments ─────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Priority-queue proxy for edge-llm-server"
    )
    parser.add_argument(
        "--proxy-host", default="localhost",
        help="Interface to bind proxy on (default: localhost)",
    )
    parser.add_argument(
        "--proxy-port", default=8082, type=int,
        help="Port proxy listens on (default: 8082)",
    )
    parser.add_argument(
        "--server-host", default="localhost",
        help="llama-server host (default: localhost)",
    )
    parser.add_argument(
        "--server-port", default=8081, type=int,
        help="llama-server port (default: 8081)",
    )
    parser.add_argument(
        "--slots", default=2, type=int,
        help="Number of inference slots — must match --parallel on server (default: 2)",
    )
    parser.add_argument(
        "--arrival-window", default=0.005, type=float,
        metavar="SECONDS",
        help=(
            "Seconds the dispatcher waits after acquiring a slot before "
            "pulling from the priority queue.  This allows simultaneous "
            "arrivals to all enqueue before the dispatch decision is made. "
            "Increase for stress tests with many concurrent clients. "
            "(default: 0.005)"
        ),
    )
    parser.add_argument(
        "--dispatch-gap", default=0.005, type=float,
        metavar="SECONDS",
        help=(
            "Seconds the dispatcher sleeps after each pop before looping back "
            "to acquire the next slot.  This staggers successive dispatch "
            "decisions so that when multiple slots are free simultaneously, "
            "each slot's arrival window starts AFTER the previous slot's pop, "
            "ensuring all burst arrivals are queued before every priority "
            "decision.  Without this, two concurrent slot acquisitions sleep "
            "their arrival windows in parallel and both pop from the same "
            "queue snapshot, which can let a lower-priority client (that "
            "arrived first) claim a slot ahead of a higher-priority one that "
            "arrived 1-2ms later.  Set to 0 to disable (reverts to v5 "
            "behaviour).  (default: 0.005)"
        ),
    )
    parser.add_argument(
        "--reserved-chat-slots", default=1, type=int,
        metavar="N",
        help=(
            "Number of physical slots to keep reserved for Chat (priority "
            "0) requests. The dispatcher will not forward a Moderate or "
            "Batch request if doing so would leave fewer than N slots free "
            "for Chat -- even if Chat has nothing queued at that moment. "
            "This removes the wait Chat otherwise experiences behind an "
            "already-running Moderate/Batch request under staggered "
            "arrivals (see July 2 staggered-arrival results). There is no "
            "timeout yet that releases an unused reserved slot back to "
            "Batch; it stays reserved indefinitely. Must satisfy "
            "0 <= N <= --slots. Set to 0 to disable reservation and "
            "reproduce pre-reservation dispatch behaviour exactly. "
            "(default: 1)"
        ),
    )
    args = parser.parse_args()

    if args.reserved_chat_slots < 0:
        parser.error("--reserved-chat-slots must be >= 0")
    if args.reserved_chat_slots > args.slots:
        parser.error(
            f"--reserved-chat-slots ({args.reserved_chat_slots}) cannot "
            f"exceed --slots ({args.slots})"
        )
    return args


# ── Configuration (populated at startup from CLI args) ───────────────────────

_cfg: argparse.Namespace  # set in main()

PRIORITY_MAP = {
    "high":   0,
    "medium": 1,
    "low":    2,
    "chat":     0,
    "moderate": 1,
    "batch":    2,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [proxy] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("proxy")

# ── Shared state ─────────────────────────────────────────────────────────────

_pq: queue.PriorityQueue = queue.PriorityQueue()
_seq_lock    = threading.Lock()
_seq_counter = 0

# Semaphore starts at SLOTS (all free).
# Dispatcher acquire()s before forwarding; _forward release()s when done.
_slot_sem: threading.Semaphore  # initialised in main()

# Tracks how many *physical* slots are currently occupied, broken down by
# priority class (0=chat, 1=moderate, 2=batch).  Used only for the
# --reserved-chat-slots policy; the semaphore above remains the sole source
# of truth for "is a physical slot free at all".
_active_by_priority = {0: 0, 1: 0, 2: 0}
_active_lock = threading.Lock()

_results: list[dict] = []
_results_lock = threading.Lock()

# ── Sequence number generator ───────────────────────────────────────────────
def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


# ── Pending request  ───────────────────────────────────────────────────────────

class PendingRequest:
    def __init__(self, method: str, path: str, headers: dict,
                 body: bytes, priority: int, client_id: str, req_id: str):
        self.method     = method
        self.path       = path
        self.headers    = headers
        self.body       = body
        self.priority   = priority
        self.client_id  = client_id
        self.req_id     = req_id
        self.t_arrived  = time.perf_counter()
        self.ready_event: threading.Event = threading.Event()
        self.response_status: Optional[int] = None
        self.response_headers: Optional[list] = None
        self.response_body_iter = None

        self.timing = {}

        self.t_queue_enter = None
        self.t_dispatch = None
        self.t_first_token = None
        self.t_complete = None


# ── Slot dispatcher ───────────────────────────────────────────────────────────

def _reservation_blocks(priority: int) -> bool:
    """
    True if dispatching a request of this priority right now would violate
    --reserved-chat-slots, i.e. it is not Chat and forwarding it would leave
    fewer than reserved_chat_slots physical slots free for Chat to use
    immediately if a Chat request were to arrive next.

    Chat (priority 0) is never blocked by its own reservation. With
    --reserved-chat-slots 0 this always returns False, reproducing
    pre-reservation dispatch behaviour exactly.
    """
    if priority == 0 or _cfg.reserved_chat_slots <= 0:
        return False
    with _active_lock:
        non_chat_active = _active_by_priority[1] + _active_by_priority[2]
    return non_chat_active >= (_cfg.slots - _cfg.reserved_chat_slots)


def _dispatcher():
    while True:
        # WAIT for something in the queue that is actually dispatchable
        # under the reservation policy (no slot held yet). If the
        # highest-priority item is non-chat and reservation forbids it,
        # put it back and keep polling -- a chat arrival will naturally
        # become queue-head next (priority queue ordering), or the
        # reservation will clear as an already-running non-chat request
        # completes and decrements _active_by_priority.
        req = None
        while req is None:
            try:
                priority, seq, candidate = _pq.get(timeout=0.05)
            except queue.Empty:
                time.sleep(0.01)
                continue

            if _reservation_blocks(candidate.priority):
                _pq.put((priority, seq, candidate))
                time.sleep(0.01)
                continue

            req = candidate

        # Put it back; now acquire the semaphore and re-collect with arrival window
        _pq.put((priority, seq, req))

        _slot_sem.acquire()
        time.sleep(_cfg.arrival_window)   # now all burst clients can arrive

        # Pop the highest-priority request
        priority, seq, req = _pq.get()

        # Re-check: a higher-priority (still non-chat) request could have
        # taken queue-head during the arrival window, or another dispatch
        # could -- in principle -- have tightened the reservation in the
        # meantime. If this pop is no longer dispatchable, release the
        # slot we're not using and retry rather than forwarding it.
        if _reservation_blocks(req.priority):
            _pq.put((priority, seq, req))
            _slot_sem.release()
            continue

        req.timing["queue_wait"] = time.perf_counter() - req.t_arrived

        with _active_lock:
            _active_by_priority[req.priority] += 1

        t = threading.Thread(target=_forward, args=(req,), daemon=True)
        t.start()

        if _cfg.dispatch_gap > 0:
            time.sleep(_cfg.dispatch_gap)


def _forward(req: PendingRequest):
    """
    Forward one request to llama-server, stream the response back,
    record split timings, then release the semaphore.
    """
    url = f"http://{_cfg.server_host}:{_cfg.server_port}{req.path}"
    http_req = Request(url, data=req.body, method=req.method)
    for k, v in req.headers.items():
        if k.lower() in ("host", "content-length", "transfer-encoding"):
            continue
        http_req.add_header(k, v)
    http_req.add_header("Content-Length", str(len(req.body)))

    t_forward = time.perf_counter()
    req.timing["t_forward"] = t_forward

    try:
        resp = urlopen(http_req, timeout=120)
    except URLError as e:
        req.response_status = 502
        req.response_headers = []
        req.response_body_iter = iter([f"Proxy error: {e}".encode()])
        req.timing["prompt_eval"]      = 0.0
        req.timing["token_generation"] = 0.0
        req.ready_event.set()
        with _active_lock:
            _active_by_priority[req.priority] -= 1
        _slot_sem.release()
        return

    req.response_status  = resp.status
    req.response_headers = list(resp.headers.items())

    first_token_seen = False
    t_first_token    = None

    def _stream():
        nonlocal first_token_seen, t_first_token
        try:
            for raw_line in resp:
                if not first_token_seen:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                chunk = json.loads(data_str)
                                delta = (chunk.get("choices", [{}])[0]
                                             .get("delta", {})
                                             .get("content", ""))
                                if delta:
                                    t_first_token = time.perf_counter()
                                    req.t_first_token = t_first_token
                                    first_token_seen = True
                            except (json.JSONDecodeError, IndexError):
                                pass
                yield raw_line
        finally:
            t_done = time.perf_counter()
            req.t_complete = t_done
            req.timing["prompt_eval"] = (
                (t_first_token - t_forward) if t_first_token else 0.0
            )
            req.timing["token_generation"] = (
                (t_done - t_first_token) if t_first_token else 0.0
            )
            req.timing["total_service_time"] = t_done - t_forward

            with _active_lock:
                _active_by_priority[req.priority] -= 1
            _slot_sem.release()

            record = {
                "dispatch_to_first_token": round(
                    (req.t_first_token - req.t_dispatch)
                    if req.t_first_token and req.t_dispatch
                    else 0,
                    4,
                ),
                "req_id":           req.req_id,
                "client_id":        req.client_id,
                "priority":         req.priority,
                "queue_wait":       round(req.timing.get("queue_wait", 0), 4),
                "ttft_at_proxy":    round(req.timing.get("prompt_eval", 0), 4),
                "token_generation": round(req.timing.get("token_generation", 0), 4),
                "total_service":    round(req.timing.get("total_service_time", 0), 4),
                "total_e2e":        round(
                    req.timing.get("queue_wait", 0) +
                    req.timing.get("total_service_time", 0), 4
                ),
            }
            with _results_lock:
                _results.append(record)

            log.info(
                f"[{req.client_id:8s}] "
                f"qw={record['queue_wait']:.3f}s "
                f"bw={record['dispatch_to_first_token']:.3f}s "
                f"ttft={record['ttft_at_proxy']:.3f}s "
                f"tg={record['token_generation']:.3f}s "
                f"e2e={record['total_e2e']:.3f}s"
            )

    req.response_body_iter = _stream()
    req.ready_event.set()


# ── HTTP handler ──────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _infer_priority(self, headers: dict) -> tuple[int, str]:
        p_header   = headers.get("x-priority", "").lower().strip()
        cid_header = headers.get("x-client-id", "").lower().strip()
        client_id  = cid_header or "unknown"

        if p_header in PRIORITY_MAP:
            return PRIORITY_MAP[p_header], client_id
        if cid_header in PRIORITY_MAP:
            return PRIORITY_MAP[cid_header], client_id
        return 1, client_id

    def do_POST(self):
        self._handle()

    def do_GET(self):
        if self.path == "/health":
            self._passthrough_get()
            return
        self._handle()

    def _passthrough_get(self):
        url = f"http://{_cfg.server_host}:{_cfg.server_port}{self.path}"
        try:
            resp = urlopen(url, timeout=5)
            body = resp.read()
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() in ("transfer-encoding",):
                    continue
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def _handle(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length) if length else b""
        hdrs   = {k.lower(): v for k, v in self.headers.items()}

        priority, client_id = self._infer_priority(hdrs)
        req_id = str(uuid.uuid4())[:8]

        req = PendingRequest(
            method    = self.command,
            path      = self.path,
            headers   = dict(self.headers),
            body      = body,
            priority  = priority,
            client_id = client_id,
            req_id    = req_id,
        )

        seq = _next_seq()
        log.info(f"[{client_id:8s}] arrived  priority={priority}  seq={seq}")
        req.t_queue_enter = time.perf_counter()
        _pq.put((priority, seq, req))

        req.ready_event.wait(timeout=130)

        if req.response_status is None:
            self.send_response(504)
            self.end_headers()
            self.wfile.write(b"Proxy timeout")
            return

        self.send_response(req.response_status)
        sent_headers = set()
        for k, v in (req.response_headers or []):
            if k.lower() in ("transfer-encoding",):
                continue
            if k.lower() not in sent_headers:
                self.send_header(k, v)
                sent_headers.add(k.lower())
        self.end_headers()

        try:
            for chunk in req.response_body_iter:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class ProxyHandlerWithResults(ProxyHandler):
    def do_GET(self):
        if self.path == "/proxy/results":
            with _results_lock:
                body = json.dumps(_results, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _cfg, _slot_sem
    _cfg = _parse_args()
    _slot_sem = threading.Semaphore(_cfg.slots)

    dt = threading.Thread(target=_dispatcher, daemon=True, name="dispatcher")
    dt.start()

    server = ThreadingHTTPServer(
        (_cfg.proxy_host, _cfg.proxy_port), ProxyHandlerWithResults
    )
    log.info(f"Priority proxy listening on {_cfg.proxy_host}:{_cfg.proxy_port}")
    log.info(f"Forwarding to llama-server at {_cfg.server_host}:{_cfg.server_port}")
    log.info(f"Slot capacity  : {_cfg.slots}")
    log.info(f"Arrival window : {_cfg.arrival_window * 1000:.1f} ms")
    log.info(f"Dispatch gap   : {_cfg.dispatch_gap * 1000:.1f} ms  (set 0 to disable)")
    if _cfg.reserved_chat_slots > 0:
        log.info(
            f"Reserved slots : {_cfg.reserved_chat_slots} of {_cfg.slots} "
            f"held for Chat (Moderate/Batch capped at "
            f"{_cfg.slots - _cfg.reserved_chat_slots} concurrent)"
        )
    else:
        log.info("Reserved slots : 0 (reservation disabled)")
    log.info("Columns: queue_wait | ttft_at_proxy | token_gen | e2e")
    log.info("─" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down proxy.")
        if _results:
            out = "proxy_results.json"
            with open(out, "w") as f:
                json.dump(_results, f, indent=2)
            log.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()