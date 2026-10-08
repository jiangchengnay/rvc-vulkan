"""T1.3 MVP: default-off capture/replay of a BatchRunner op tape.

Design constraints (from _diag/replay_t13_poc.md):
  - Risk 5 (aliasing): in-place ops reuse their input buffer id as the output
    (add_inplace, mul_inplace, leaky_relu put the result in the `a` slot;
    copy puts dst in the `a` slot too, NOT `c`). The bid remap MUST therefore
    be a single-valued function, otherwise replay silently corrupts chains.
  - Lifecycle: the tape MUST be exported BEFORE BatchRunner.release(), because
    release() clears _records and invalidates BatchTensors.
  - Gate: env RVC_TRAIN_REPLAY_DEC must be "1" to enable. Unset -> zero
    behaviour change.

Replay needs TWO phases, not one. The ids in _records are ids of
ALREADY-ALLOCATED engine buffers; renumbering them is not an allocation, so a
fresh runner rejects them with InvalidBuffer. Hence:
    phase 1 ALLOCATE: recreate a real buffer per distinct bid, at its shape
    phase 2 RE-ISSUE:  call rvc_batch_add with the new ids and the same ps
That is why we must record per-bid SHAPES at capture time: shape is a
per-buffer property (one bid can feed many ops and be the output of another),
so it cannot be re-derived per op.
"""
from __future__ import annotations

import hashlib
import json
import os

__all__ = [
    "CaptureSession",
    "Tape",
    "enabled",
    "digest",
    "replay",
    "remap_of",
    "allocate",
]


def enabled() -> bool:
    """True when capture/replay is switched on via RVC_TRAIN_REPLAY_DEC=1."""
    return os.environ.get("RVC_TRAIN_REPLAY_DEC", "0") == "1"


def digest(arr) -> str:
    """Deterministic sha256 over a numpy array's dtype, shape and raw bytes."""
    import numpy as np

    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


class Tape:
    """An ordered list of recorded ops, plus per-bid shapes.

    ``shapes`` maps buffer id -> shape list. It is what makes replay possible:
    without it the replayed runner cannot allocate buffers with the right
    sizes and the engine answers InvalidBuffer.
    """

    def __init__(self) -> None:
        self.entries = []
        self.shapes = {}
        self.values = {}
        self.errors = []

    def add(self, entry) -> None:
        self.entries.append(entry)

    def note_value(self, bid, arr) -> None:
        """Record the initial VALUE of a buffer that no op in the tape produces.

        Intermediates get their contents from the replayed ops, but the
        chain's true inputs (x, w, b, ...) were handed in by the caller as
        numpy arrays. Without their values replay would run on zeros and give
        a meaningless - though perfectly well-formed - answer.
        """
        if bid is None or arr is None:
            return
        k = str(bid)
        if k not in self.values:
            import numpy as np

            self.values[k] = np.ascontiguousarray(arr)

    def value_of(self, bid):
        return self.values.get(str(bid))

    def note_shape(self, bid, shape) -> None:
        """Record the shape a buffer id was created with.

        First writer wins for a given bid; a later conflicting shape is
        recorded as an error rather than silently overwriting, because that
        would mean one buffer id was reused at two different sizes (which
        would make faithful replay impossible and is worth surfacing loudly).
        """
        if bid is None or shape is None:
            return
        s = list(shape)
        prev = self.shapes.get(str(bid))
        if prev is None:
            self.shapes[str(bid)] = s
        elif prev != s:
            self.errors.append("bid %s shape conflict: %s vs %s" % (bid, prev, s))

    def shape_of(self, bid):
        s = self.shapes.get(str(bid))
        return None if s is None else list(s)

    def to_json(self) -> str:
        # `values` holds numpy arrays, so serialise them as base64 payloads
        # carrying dtype+shape. Dropping them silently degrades replay to
        # zero-input and produces a well-formed but meaningless answer.
        import base64

        import numpy as np

        vals = {}
        for k, a in self.values.items():
            aa = np.ascontiguousarray(a)
            vals[k] = {
                "dtype": str(aa.dtype),
                "shape": list(aa.shape),
                "b64": base64.b64encode(aa.tobytes()).decode("ascii"),
            }
        return json.dumps(
            {
                "entries": self.entries,
                "shapes": self.shapes,
                "values": vals,
                "errors": self.errors,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(s):
        import base64

        import numpy as np

        t = Tape()
        d = json.loads(s)
        t.entries = d.get("entries", [])
        t.shapes = d.get("shapes", {})
        t.errors = d.get("errors", [])
        for k, v in d.get("values", {}).items():
            arr = np.frombuffer(
                base64.b64decode(v["b64"]), dtype=v["dtype"]
            ).reshape(v["shape"])
            t.values[k] = np.ascontiguousarray(arr)
        return t

    def __len__(self) -> int:
        return len(self.entries)


def input_bids_of(tape) -> set:
    """The bids that must be materialised before replay: TRUE INPUTS only.

    This is the crux of the replay design, and getting it wrong is silent.
    A bid is a true input iff some op READS it and NO op in the tape PRODUCES
    it. Output bids must be excluded, because the replayed operators allocate
    their own output buffers through _alloc_output; if we pre-allocated a
    buffer for an output bid, the op would write into its own buffer rather
    than ours, leaving our zero-filled buffer untouched for any later op that
    reads that bid - and zeros would propagate to the final result.
    """
    produced = set()
    for e in tape.entries:
        tb = e["tensor_buf"]
        if tb is not None:
            produced.add(int(tb))
        if e["out"] is not None:
            c = int(e["c"])
            # In-place ops (leaky_relu, add_inplace, mul_inplace) pass their
            # input id in the c slot as the write-back target. `c == a` is the
            # reliable signal: it holds for every in-place op and cannot fire
            # for a producer that writes to a fresh buffer.
            #
            # Deliberately NOT hardcoding an op-number list here. An earlier
            # version used `e["op"] in (3, 4, 6)`, a guess that happens to
            # exclude copy (op 7) - and copy DOES write an output (dst in the
            # `a` slot, c=0). That guess was only harmless because copy also
            # passes tensor_buf; any future op that allocates an output and
            # forgets tensor_buf would be silently classified as an input,
            # zero-filled by allocate(), and corrupt the replay.
            if c == int(e["a"]):
                produced.add(c)
    read = set()
    for e in tape.entries:
        for k in ("a", "b", "c"):
            read.add(int(e[k]))
    return {b for b in read if b not in produced and b != 0}


def remap_of(tape) -> dict:
    """Single-valued old-bid -> new-bid map, allocated in first-seen order.

    Risk 5: because in-place ops reuse their input id as output, the same
    source bid must always map to the same target bid.
    """
    m = {}

    def get(bid):
        if bid not in m:
            m[bid] = len(m)
        return m[bid]

    for e in tape.entries:
        for k in ("a", "b", "c"):
            get(e[k])
        if e["tensor_buf"] is not None:
            get(e["tensor_buf"])
    for k in tape.shapes:
        get(int(k))
    return m


class CaptureSession:
    """Records every BatchRunner op, and the shape of every buffer id.

    Usage::

        with CaptureSession(br) as cap:
            ...  # normal ops, forwarded to the real BatchRunner
        tape = cap.tape

    Nothing is recorded unless enabled(); when disabled this is a no-op.
    """

    def __init__(self, br) -> None:
        self.br = br
        self.tape = Tape()
        self._orig_record = None
        self._orig_resolve = None
        self._orig_alloc = None

    def __enter__(self):
        if not enabled():
            return self
        tape = self.tape

        # --- op recording (the single choke point for all operators) ---
        self._orig_record = self.br._record
        orig = self._orig_record

        def _recording_record(op, a_id, b_id, c_id, ps, out_shape, tensor_buf=None):
            ps_l = [int(p) for p in ps] + [0] * (11 - len(ps))
            # Guard against SILENT replay degradation: replay() learns an
            # output buffer's size only from this out_shape. If an op writes a
            # result but reports no shape, replay cannot size the buffer and
            # would quietly produce a wrong answer. Record it as a tape error
            # rather than letting it pass unnoticed.
            if out_shape is None and tensor_buf is not None:
                tape.errors.append(
                    "op %s wrote buffer %s but reported no out_shape; "
                    "replay cannot size it" % (int(op), int(tensor_buf))
                )
            # The mirror hazard: an op that declares an output shape but gives
            # no way to identify its output buffer. `input_bids_of` would then
            # mistake that output for an external input, allocate() would
            # zero-fill it, and the replay would silently diverge.
            #
            # An output is locatable by exactly two signals: an explicit
            # tensor_buf, or a `c` slot naming a real destination. The latter
            # holds when c aliases the in-place target (c == a) or c is a
            # fresh id distinct from both operands. It does NOT hold when c is
            # 0 (unused slot, as in leaky_relu/copy) or when c aliases b (an
            # operand, not a destination).
            c_i, a_i, b_i = int(c_id), int(a_id), int(b_id)
            if (
                out_shape is not None
                and tensor_buf is None
                and not (c_i == a_i or c_i not in (0, a_i, b_i))
            ):
                tape.errors.append(
                    "op %s declares out_shape %s but no tensor_buf and c=%s "
                    "names no destination; replay cannot locate its output"
                    % (int(op), list(out_shape), c_i)
                )
            tape.add(
                {
                    "op": int(op),
                    "a": int(a_id),
                    "b": int(b_id),
                    "c": int(c_id),
                    "ps": ps_l,
                    "out": None if out_shape is None else list(out_shape),
                    "tensor_buf": None if tensor_buf is None else int(tensor_buf),
                }
            )
            return orig(op, a_id, b_id, c_id, ps, out_shape, tensor_buf)

        self.br._record = _recording_record

        # --- input shape capture ---
        # _resolve_input returns (bid, shape, owned); wrapping it is how we
        # learn the shape of every buffer that enters an op as an operand.
        self._orig_resolve = self.br._resolve_input
        orig_resolve = self._orig_resolve

        def _recording_resolve(x, name, buf=None):
            bid, shape, owned = orig_resolve(x, name, buf)
            tape.note_shape(bid, shape)
            # `owned` is True only for the numpy-upload path, i.e. a buffer
            # whose contents came from the caller and are NOT produced by any
            # op in the tape. Those are exactly the values replay must restore.
            if owned:
                try:
                    import numpy as _np

                    tape.note_value(bid, _np.asarray(x, dtype=_np.float32))
                except Exception as exc:  # never let diagnostics break capture
                    tape.errors.append("note_value(%s) failed: %r" % (name, exc))
            return bid, shape, owned

        self.br._resolve_input = _recording_resolve

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._orig_record is not None:
            self.br._record = self._orig_record
            self._orig_record = None
        if self._orig_resolve is not None:
            self.br._resolve_input = self._orig_resolve
            self._orig_resolve = None
        return False

    def export(self, path: str) -> dict:
        """Write the tape to disk as JSON; returns a summary dict."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.tape.to_json())
        ops = {}
        for e in self.tape.entries:
            ops[e["op"]] = ops.get(e["op"], 0) + 1
        return {
            "path": path,
            "n_ops": len(self.tape),
            "ops": dict(sorted(ops.items())),
            "n_shapes": len(self.tape.shapes),
            "errors": list(self.tape.errors),
        }

    def remap(self) -> dict:
        return remap_of(self.tape)


def allocate(br, tape, remap, only_inputs=True):
    """Phase 1 of replay: materialise a real buffer for each TRUE INPUT bid.

    Buffers come from the same pooled-upload path the honest forward pass uses.
    Where we recorded the input's initial value we upload that value; otherwise
    the buffer starts as zeros, which is correct only for buffers an op will
    fully overwrite.

    Why only INPUTS (the crux): replayed operators allocate their OWN output
    buffers via _alloc_output. If we pre-allocated a buffer for an output bid,
    the op would write into its own buffer rather than ours, leaving our
    zero-filled buffer untouched - and any later op reading that bid would read
    zeros, silently corrupting the final result. This exact bug produced a
    maxdiff of 4.95 with an all-zero output before it was fixed.

    A bid whose shape we never learned is reported in 'missing' rather than
    guessed, because guessing a size would produce a silently wrong replay.
    """
    import numpy as np

    want = input_bids_of(tape) if only_inputs else set(remap.keys())

    made = {}
    missing = []
    restored = []
    for old_bid in sorted(remap.keys()):
        if int(old_bid) not in want:
            continue
        shape = tape.shape_of(old_bid)
        if shape is None:
            missing.append(int(old_bid))
            continue
        # Use the recorded initial value when this bid is a true input;
        # otherwise zeros, which the replayed ops will overwrite.
        val = tape.value_of(old_bid)
        if val is None:
            arr = np.zeros(tuple(shape), dtype=np.float32)
        else:
            arr = np.ascontiguousarray(val, dtype=np.float32)
            restored.append(int(old_bid))
        b, need_copy = br._ctx._pooled_reserve(arr)
        if need_copy:
            br._up_q.append((b, arr))
        br._owned_inputs.add(b)
        made[int(remap[old_bid])] = b
    return {"made": made, "missing": missing, "restored": restored}


def replay(br, tape, remap=None):
    """Re-issue a captured tape onto a fresh BatchRunner.

    Two phases (see module docstring): allocate real buffers per bid, then
    call rvc_batch_add with the new ids and the captured ps.

    Honours both traps:
      * aliasing (risk 5): a single-valued remap keeps an in-place op's
        write-back to its input id consistent.
      * conv1d carries the BIAS in the ``c`` slot and the OUTPUT elsewhere,
        which is why each entry stores ``tensor_buf`` separately. Never
        assume out == c.

    Returns {new_bid: BatchTensor} so the caller can pull final values out.
    """
    if remap is None:
        remap = remap_of(tape)

    from runtime import vulkan_ops as _V

    alloc = allocate(br, tape, remap)
    made = alloc["made"]

    # Output bids must get a REAL buffer from the engine's own allocator.
    # replay() drives rvc_batch_add directly, bypassing the op wrappers, so
    # _alloc_output is never called for us. If we handed the engine a bare
    # remapped integer for an output slot, nothing would own that buffer's
    # storage and the result would vanish (the `is_zero=True` bug on bids 4/7).
    out_alloc = {}
    for e in tape.entries:
        if e["out"] is None:
            continue
        ob = e["tensor_buf"]
        ob = int(ob) if ob is not None else int(e["c"])
        if ob in out_alloc:
            continue
        shape = e["out"]
        n = 1
        for d in shape:
            n *= int(d)
        if n <= 0:
            # A zero/negative element count is never a legitimate output and
            # means the recorded shape is unusable. Fail loudly: silently
            # allocating here would produce a wrong replay that still "runs".
            raise RuntimeError(
                "replay: op %s declares output buffer %s with unusable "
                "shape %s" % (e["op"], ob, shape)
            )
        out_alloc[ob] = br._alloc_output(n)

    def m(bid):
        """old bid -> the concrete engine buffer id to use on the new runner.

        Resolution order matters, and it is the whole ballgame:
          1. An OUTPUT bid resolves to out_alloc, the buffer the ENGINE gave
             us via _alloc_output. Never to `made`: those are the input
             buffers we uploaded, and an output bid has no entry there.
          2. An INPUT bid resolves through remap to the uploaded buffer.
        `made` is keyed by NEW bid (see allocate), so the remap hop is required.
        """
        b = int(bid)
        if b in out_alloc:
            return out_alloc[b]
        nb = remap.get(b)
        if nb is None:
            return made.get(b, 0)
        return made.get(nb, nb)

    tensors = {}
    for e in tape.entries:
        a, b, c = m(e["a"]), m(e["b"]), m(e["c"])
        ps = list(e["ps"])
        with br._lock:
            br._ensure_begin()
            _V._vulkan._check(
                _V._vulkan.dll.rvc_batch_add(
                    br._ctx._handle, int(e["op"]), a, b, c, *ps
                ),
                "replay rvc_batch_add(op=%s)" % e["op"],
            )
            br._records.append((int(e["op"]), a, b, c, ps))
        if e["out"] is not None:
            tb = e["tensor_buf"]
            ob = int(tb) if tb is not None else int(e["c"])
            out_buf = out_alloc.get(ob, m(ob))
            t = _V.BatchTensor(br, out_buf, list(e["out"]))
            with br._lock:
                br._tensors.append(t)
            tensors[ob] = t
    return {"tensors": tensors, "alloc": alloc, "out_alloc": out_alloc}
