"""Tests for the minimal GGUF writer and the synthetic nano-llama fixture.

The writer emits a real GGUF v3 container. These tests verify the bytes it
produces (header, metadata, tensor directory) without needing llama.cpp, and —
when the native backend is importable — confirm llama.cpp actually loads the
fixture and runs inference through it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from momentum.llm.gguf_writer import (
    DEFAULT_ALIGNMENT,
    GGML_TYPE_F32,
    GGUF_MAGIC,
    GGUF_VERSION,
    build_nano_llama,
    u32,
    write_gguf,
)

_HEADER = struct.calcsize("<4sIQQ")


def _read_header(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    assert raw[:4] == GGUF_MAGIC
    version, n_tensors, n_kv = struct.unpack("<IQQ", raw[4:_HEADER])
    assert version == GGUF_VERSION
    return n_tensors, n_kv


class TestWriteGguf:
    def test_header_records_tensor_and_kv_counts(self, tmp_path: Path) -> None:
        out = write_gguf(
            tmp_path / "m.gguf",
            [("a", u32(1)), ("b", "two")],
            [("t", np.zeros((2, 3), dtype=np.float32))],
        )
        assert _read_header(out) == (1, 2)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        assert write_gguf(tmp_path / "deep" / "nested" / "m.gguf", [], []).exists()

    def test_tensor_data_section_starts_aligned(self, tmp_path: Path) -> None:
        """The data section must begin on an alignment boundary.

        llama.cpp computes the data offset by rounding the end of the tensor
        directory up; an unpadded section shifts every tensor and the last ones
        fall past EOF.
        """
        out = write_gguf(
            tmp_path / "m.gguf",
            [("k" * 7, "v"), ("n", u32(3))],
            [("t", np.ones((4, 4), dtype=np.float32))],
        )
        raw = out.read_bytes()
        start = raw.index(np.ones((4, 4), dtype=np.float32).tobytes())
        assert start % DEFAULT_ALIGNMENT == 0, f"data at {start} is not aligned"

    def test_odd_sized_metadata_still_aligns(self, tmp_path: Path) -> None:
        out = write_gguf(
            tmp_path / "m.gguf",
            [("odd", "x" * 13)],
            [("t", np.full((3,), 2.0, dtype=np.float32))],
        )
        raw = out.read_bytes()
        start = raw.index(np.full((3,), 2.0, dtype=np.float32).tobytes())
        assert start % DEFAULT_ALIGNMENT == 0

    def test_rejects_unsupported_value_type(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            write_gguf(tmp_path / "m.gguf", [("k", object())], [])

    def test_writes_f32_tensor_type_tag(self, tmp_path: Path) -> None:
        out = write_gguf(
            tmp_path / "m.gguf", [], [("t", np.zeros(2, dtype=np.float32))]
        )
        assert struct.pack("<I", GGML_TYPE_F32) in out.read_bytes()


class TestNanoLlama:
    def test_is_small_and_well_formed(self, tmp_path: Path) -> None:
        out = build_nano_llama(tmp_path / "nano.gguf")
        # ~180 KB: kilobytes, not the ~720 MB a real checkpoint needs.
        assert 1024 < out.stat().st_size < 5 * 1024 * 1024
        n_tensors, n_kv = _read_header(out)
        assert n_tensors == 1 + 2 * 9 + 2  # embedding + 2 layers + 2 output
        assert n_kv > 10

    def test_is_deterministic_for_a_fixed_seed(self, tmp_path: Path) -> None:
        a = build_nano_llama(tmp_path / "a.gguf", seed=99)
        b = build_nano_llama(tmp_path / "b.gguf", seed=99)
        c = build_nano_llama(tmp_path / "c.gguf", seed=100)
        assert a.read_bytes() == b.read_bytes()
        assert a.read_bytes() != c.read_bytes()

    def test_honours_layer_and_width_overrides(self, tmp_path: Path) -> None:
        out = build_nano_llama(tmp_path / "nano.gguf", n_layer=1, n_embd=16, n_head=2)
        assert _read_header(out)[0] == 1 + 1 * 9 + 2

    def test_trailing_weights_are_finite(self, tmp_path: Path) -> None:
        """NaN/inf weights would poison every forward pass."""
        out = build_nano_llama(tmp_path / "nano.gguf")
        tail = np.frombuffer(out.read_bytes()[-4096:], dtype=np.float32)
        assert np.isfinite(tail).all()


def _llama_cpp_or_skip() -> object:
    return pytest.importorskip("llama_cpp", reason="native backend absent")


class TestRealLlamaCpp:
    """The point of the fixture: exercise the genuine native load path."""

    def test_loads_and_generates(self, tmp_path: Path) -> None:
        llama_cpp = _llama_cpp_or_skip()
        out = build_nano_llama(tmp_path / "nano.gguf")
        llm = llama_cpp.Llama(
            model_path=str(out), n_ctx=128, n_threads=2, verbose=False, seed=7
        )
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
            temperature=0.0,
        )
        assert isinstance(result["choices"][0]["message"]["content"], str)
        del llm

    def test_streams_chunks(self, tmp_path: Path) -> None:
        llama_cpp = _llama_cpp_or_skip()
        out = build_nano_llama(tmp_path / "nano.gguf")
        llm = llama_cpp.Llama(
            model_path=str(out), n_ctx=128, n_threads=2, verbose=False, seed=7
        )
        chunks = list(
            llm.create_chat_completion(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=8,
                temperature=0.0,
                stream=True,
            )
        )
        assert chunks
        # The first chunk carries a role but no content — the exact shape that
        # used to crash the engine's response parsing.
        first_delta = chunks[0]["choices"][0].get("delta", {})
        assert "role" in first_delta
        for chunk in chunks:
            assert "choices" in chunk
        del llm

    def test_engine_generate_uses_the_fixture(self, tmp_path: Path) -> None:
        """LlmEngine drives the real backend against a real (tiny) model."""
        from momentum.llm.engine import LlmEngine

        out = build_nano_llama(tmp_path / "nano.gguf")
        engine = LlmEngine(model_path=out, n_ctx=128, n_threads=2)
        engine.load()
        assert engine.is_loaded
        text = engine.generate(
            [{"role": "user", "content": "hello"}], max_tokens=4, temperature=0.0
        )
        assert isinstance(text, str)
        engine.unload()
        assert not engine.is_loaded
