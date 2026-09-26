"""Minimal GGUF writer -- build a tiny, valid, randomly-initialised GGUF.

Momentum deliberately ships no model: the coach downloads one on first use.
That makes the *real* llama.cpp load path awkward to exercise -- CI would need
a several-hundred-megabyte download plus network access, and any test that
does so is slow and flaky.

This module writes a structurally valid GGUF (format v3) containing a
nano-sized ``llama`` with random weights. llama.cpp loads it and runs genuine
inference, so tests can drive ``LlmEngine`` against the real native stack
entirely offline.

The weights are random, so generated text is meaningless. This fixture
exercises *plumbing* -- GGUF parsing, tensor mapping, the sampling loop, the
chat template -- never model quality.

Only the subset of GGUF that a llama checkpoint needs is implemented; there is
no quantisation, no multimodal and no GGUFWriter re-export.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types (gguf_type).
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FORMAT = {
    _T_UINT8: "<B",
    _T_INT8: "<b",
    _T_UINT16: "<H",
    _T_INT16: "<h",
    _T_UINT32: "<I",
    _T_INT32: "<i",
    _T_FLOAT32: "<f",
    _T_BOOL: "<B",
    _T_UINT64: "<Q",
    _T_INT64: "<q",
    _T_FLOAT64: "<d",
}

# ggml_token_type values used by the SentencePiece tokenizer.
_TT_NORMAL, _TT_CONTROL = 1, 3


# Explicitly-typed metadata values. GGUF distinguishes u32/i32/u64/i64/f32, and
# llama.cpp rejects a key whose declared width does not match what it expects
# (``general.alignment`` must be u32, for example), so callers need a way to say
# which width to emit instead of relying on inference from the Python type.
class _Typed:
    """A metadata value with an explicit GGUF type tag."""

    __slots__ = ("gguf_type", "value")

    def __init__(self, gguf_type: int, value: Any) -> None:
        self.gguf_type = gguf_type
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Typed({self.gguf_type}, {self.value!r})"


def u32(value: int) -> _Typed:
    """Tag *value* as a 32-bit unsigned GGUF integer."""
    return _Typed(_T_UINT32, int(value))


def i32(value: int) -> _Typed:
    """Tag *value* as a 32-bit signed GGUF integer."""
    return _Typed(_T_INT32, int(value))


def f32(value: float) -> _Typed:
    """Tag *value* as a 32-bit float (required by some RMS-epsilon keys)."""
    return _Typed(_T_FLOAT32, float(value))


class _TypedArray:
    """A metadata array with an explicit element type."""

    __slots__ = ("element_type", "values")

    def __init__(self, element_type: int, values: Sequence[Any]) -> None:
        self.element_type = element_type
        self.values = list(values)

    def __len__(self) -> int:
        return len(self.values)


def f32_array(values: Sequence[float]) -> _TypedArray:
    """Tag *values* as an ``arr[f32]`` -- required by tokenizer.ggml.scores."""
    return _TypedArray(_T_FLOAT32, [float(v) for v in values])


def i32_array(values: Sequence[int]) -> _TypedArray:
    """Tag *values* as an ``arr[i32]`` -- required by tokenizer token types."""
    return _TypedArray(_T_INT32, [int(v) for v in values])


def str_array(values: Sequence[str]) -> _TypedArray:
    """Tag *values* as an ``arr[str]`` -- required by tokenizer.ggml.tokens."""
    return _TypedArray(_T_STRING, list(values))


def _align(offset: int, alignment: int = DEFAULT_ALIGNMENT) -> int:
    """Round *offset* up to the next *alignment* boundary."""
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)


class _Writer:
    """Incremental little-endian GGUF serializer."""

    def __init__(self, alignment: int = DEFAULT_ALIGNMENT) -> None:
        self._buf = bytearray()
        self._alignment = alignment

    # -- primitives -------------------------------------------------
    def raw(self, fmt: str, value: Any) -> None:
        self._buf += struct.pack(fmt, value)

    def gguf_string(self, value: str) -> None:
        """Write a length-prefixed UTF-8 string (no NUL terminator)."""
        encoded = value.encode("utf-8")
        self.raw("<Q", len(encoded))
        self._buf += encoded

    def value(self, value: Any) -> None:
        """Write a metadata value given its already-declared type."""
        if isinstance(value, _Typed):
            self.raw(_SCALAR_FORMAT[value.gguf_type], value.value)
        elif isinstance(value, bool):
            self.raw(_SCALAR_FORMAT[_T_BOOL], value)
        elif isinstance(value, _TypedArray):
            self.raw("<I", value.element_type)
            self.raw("<Q", len(value.values))
            for item in value.values:
                if isinstance(item, str):
                    self.gguf_string(item)
                else:
                    self.raw(_SCALAR_FORMAT[value.element_type], item)
        elif isinstance(value, str):
            self.gguf_string(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            self.array(list(value))
        elif isinstance(value, (float, np.floating)):
            self.raw(_SCALAR_FORMAT[_T_FLOAT64], float(value))
        elif isinstance(value, (int, np.integer)):
            self.raw(_SCALAR_FORMAT[_T_INT64], int(value))
        else:
            raise TypeError(f"unsupported GGUF value: {type(value)!r}")

    def scalar_type(self, value: Any) -> int:
        if isinstance(value, _Typed):
            return value.gguf_type
        if isinstance(value, _TypedArray):
            return _T_ARRAY
        # bool must be checked before int: it is a subclass, and GGUF has a
        # distinct BOOL type that llama.cpp validates keys against.
        if isinstance(value, bool):
            return _T_BOOL
        if isinstance(value, str):
            return _T_STRING
        if isinstance(value, (list, tuple, np.ndarray)):
            # Arrays declare their own element type; this is the array tag.
            return _T_ARRAY
        if isinstance(value, (float, np.floating)):
            return _T_FLOAT64
        if isinstance(value, (int, np.integer)):
            return _T_INT64
        raise TypeError(f"cannot infer GGUF type for {type(value)!r}")

    def array(self, values: Sequence[Any]) -> None:
        """Write an array: element type, then uint64 count, then elements."""
        if not len(values):
            self.raw("<I", _T_INT64)
            self.raw("<Q", 0)
            return
        self.raw("<I", self.scalar_type(values[0]))
        self.raw("<Q", len(values))
        for item in values:
            if isinstance(item, str):
                self.gguf_string(item)
            elif isinstance(item, (float, np.floating)):
                self.raw(_SCALAR_FORMAT[_T_FLOAT64], float(item))
            else:
                self.raw(_SCALAR_FORMAT[_T_INT64], int(item))


def write_gguf(
    path: Path,
    metadata: Sequence[tuple[str, Any]],
    tensors: Sequence[tuple[str, np.ndarray]],
    alignment: int = DEFAULT_ALIGNMENT,
) -> Path:
    """Write a GGUF v3 file containing F32 *tensors* and scalar *metadata*.

    ``metadata`` may embed arrays as numpy arrays or plain lists of str/int/
    float. Tensor data is aligned to *alignment* bytes, as the format requires.
    """
    kv = _Writer(alignment)
    for key, value in metadata:
        kv.gguf_string(key)
        kv.raw("<I", kv.scalar_type(value))
        kv.value(value)

    directory = _Writer(alignment)
    data = bytearray()
    for name, array in tensors:
        block = np.ascontiguousarray(array, dtype=np.float32)
        # Pad the running offset so each tensor starts on an aligned boundary.
        pad = (-len(data)) % alignment
        data += b"\x00" * pad
        offset = len(data)
        directory.gguf_string(name)
        directory.raw("<I", block.ndim)
        # GGUF records dimensions in ggml order, where ne[0] varies fastest.
        # NumPy is row-major with the *last* axis varying fastest, so the shape
        # is written reversed. Storing this the other way round makes every
        # tensor transposed, which llama.cpp rejects via check_tensor_dims.
        for dim in reversed(block.shape):
            directory.raw("<Q", int(dim))
        directory.raw("<I", GGML_TYPE_F32)
        directory.raw("<Q", offset)
        data += block.tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = GGUF_MAGIC + struct.pack("<I", GGUF_VERSION)
    prefix += struct.pack("<Q", len(tensors))
    prefix += struct.pack("<Q", len(metadata))
    prefix += bytes(kv._buf) + bytes(directory._buf)

    # llama.cpp locates the tensor-data section by rounding the end of the
    # tensor directory up to ``general.alignment``. The data section must be
    # padded to that same boundary, otherwise every tensor is read shifted and
    # the last ones fall past the end of the file.
    prefix += b"\x00" * ((-len(prefix)) % alignment)

    with path.open("wb") as handle:
        handle.write(prefix)
        handle.write(bytes(data))
    return path


def _nano_vocab() -> tuple[list[str], list[float], list[int]]:
    """Vocabulary for the fixture: specials, byte fallbacks, and word pieces.

    llama.cpp builds its byte->token table by looking up literal ``<0xNN>``
    pieces in the vocabulary, then calls ``unordered_map::at`` for every input
    byte while tokenizing. Any byte without such a piece therefore aborts the
    process with ``std::out_of_range`` rather than raising in Python, so all
    256 byte-fallback tokens must be present for *any* text to tokenize.
    """
    tokens = ["<unk>", "<s>", "</s>"]
    scores = [0.0, 0.0, 0.0]
    types = [_TT_CONTROL, _TT_CONTROL, _TT_CONTROL]

    for byte in range(256):
        tokens.append(f"<0x{byte:02X}>")
        # Descending scores keep the greedy sampler well-defined.
        scores.append(-float(byte))
        types.append(_TT_NORMAL)

    # Printable word pieces, so a prompt tokenizes into more than one token and
    # the chat template has something real to work with.
    for code in range(0x20, 0x7F):
        char = chr(code)
        tokens.append(char if code == 0x20 else "▁" + char)
        scores.append(-float(code))
        types.append(_TT_NORMAL)
    return tokens, scores, types


def build_nano_llama(
    path: Path,
    *,
    n_vocab: int = 354,
    n_embd: int = 32,
    n_layer: int = 2,
    n_head: int = 4,
    n_ff: int = 64,
    n_ctx: int = 128,
    seed: int = 1234,
) -> Path:
    """Write a randomly-initialised nano ``llama`` GGUF to *path*.

    The result loads in ``llama_cpp.Llama`` and performs genuine forward
    passes, which is enough to test loading, the chat template and token
    streaming. Weights are random, so completions are gibberish by design.
    """
    rng = np.random.default_rng(seed)
    head_dim = n_embd // n_head

    def normal(*shape: int) -> np.ndarray:
        # Small scale keeps activations from exploding to inf/NaN at 2 layers.
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    tokens, scores, token_types = _nano_vocab()
    n_vocab = min(n_vocab, len(tokens))
    tokens, scores, token_types = (
        tokens[:n_vocab],
        scores[:n_vocab],
        token_types[:n_vocab],
    )

    metadata: list[tuple[str, Any]] = [
        ("general.architecture", "llama"),
        ("general.name", "momentum-nano-fixture"),
        ("general.file_type", u32(0)),
        ("general.alignment", u32(DEFAULT_ALIGNMENT)),
        ("llama.context_length", u32(n_ctx)),
        ("llama.embedding_length", u32(n_embd)),
        ("llama.block_count", u32(n_layer)),
        ("llama.feed_forward_length", u32(n_ff)),
        ("llama.attention.head_count", u32(n_head)),
        ("llama.attention.head_count_kv", u32(n_head)),
        ("llama.attention.layer_norm_rms_epsilon", f32(1e-5)),
        ("llama.rope.dimension_count", u32(head_dim)),
        ("llama.vocab_size", u32(n_vocab)),
        ("tokenizer.ggml.model", "llama"),
        ("tokenizer.ggml.tokens", str_array(tokens)),
        ("tokenizer.ggml.scores", f32_array(scores)),
        ("tokenizer.ggml.token_type", i32_array(token_types)),
        ("tokenizer.ggml.bos_token_id", u32(1)),
        ("tokenizer.ggml.eos_token_id", u32(2)),
        ("tokenizer.ggml.unknown_token_id", u32(0)),
        ("tokenizer.ggml.add_bos_token", True),
        ("tokenizer.ggml.add_eos_token", False),
    ]

    # Tensor arrays below are given in numpy (row-major) order; write_gguf
    # reverses the dimensions when recording them, so each shape here is the
    # transpose of what llama.cpp expects in ggml order.
    tensors: list[tuple[str, np.ndarray]] = [
        ("token_embd.weight", normal(n_vocab, n_embd))
    ]
    for layer in range(n_layer):
        prefix = f"blk.{layer}"
        tensors += [
            (f"{prefix}.attn_norm.weight", np.ones(n_embd, dtype=np.float32)),
            (f"{prefix}.attn_q.weight", normal(n_embd, n_embd)),
            (f"{prefix}.attn_k.weight", normal(n_embd, n_embd)),
            (f"{prefix}.attn_v.weight", normal(n_embd, n_embd)),
            (f"{prefix}.attn_output.weight", normal(n_embd, n_embd)),
            (f"{prefix}.ffn_norm.weight", np.ones(n_embd, dtype=np.float32)),
            (f"{prefix}.ffn_gate.weight", normal(n_ff, n_embd)),
            # ffn_down is the transpose of gate/up in llama's layout.
            (f"{prefix}.ffn_down.weight", normal(n_embd, n_ff)),
            (f"{prefix}.ffn_up.weight", normal(n_ff, n_embd)),
        ]
    tensors.append(("output_norm.weight", np.ones(n_embd, dtype=np.float32)))
    # Untied embeddings: the output projection is its own matrix.
    tensors.append(("output.weight", normal(n_vocab, n_embd)))

    return write_gguf(path, metadata, tensors)
