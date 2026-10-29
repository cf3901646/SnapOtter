"""Contract tests for session_cache: the byte budget, LRU eviction, the
providers-in-key rule, per-call-option passthrough, and idle eviction that
never races an in-flight request. No GPU or ONNX needed; sessions are plain
objects and the clock is passed in."""
import os
import sys
import weakref

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import session_cache  # noqa: E402

CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU = ["CPUExecutionProvider"]
MB = 1024 * 1024


class Session:
    pass


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    session_cache._entries.clear()
    session_cache._used_this_request.clear()
    session_cache._busy = False
    monkeypatch.setenv("AI_SESSION_CACHE_BYTES", str(100 * MB))
    monkeypatch.delenv("AI_SESSION_IDLE_TIMEOUT", raising=False)
    yield
    session_cache._entries.clear()
    session_cache._busy = False


def counting_factory():
    made = []

    def create():
        made.append(Session())
        return made[-1]

    return create, made


def test_second_call_reuses_the_session():
    create, made = counting_factory()
    a = session_cache.get_session("m", CUDA, create, 10 * MB)
    b = session_cache.get_session("m", CUDA, create, 10 * MB)
    assert a is b
    assert len(made) == 1


def test_providers_are_part_of_the_key():
    create, made = counting_factory()
    cuda = session_cache.get_session("m", CUDA, create, 10 * MB)
    cpu = session_cache.get_session("m", CPU, create, 10 * MB)
    assert cuda is not cpu
    assert len(made) == 2


def test_call_options_bypass_the_cache():
    create, _ = counting_factory()
    a = session_cache.get_session("m", CUDA, create, 10 * MB, sess_options="x")
    b = session_cache.get_session("m", CUDA, create, 10 * MB, sess_options="x")
    assert a is not b
    assert not session_cache._entries


def test_unknown_cost_is_not_cached():
    create, _ = counting_factory()
    session_cache.get_session("m", CUDA, create, None)
    assert not session_cache._entries


def test_session_larger_than_budget_is_built_but_not_cached():
    create, made = counting_factory()
    session_cache.get_session("big", CUDA, create, 101 * MB)
    assert len(made) == 1
    assert not session_cache._entries


def test_budget_of_one_byte_disables_caching(monkeypatch):
    monkeypatch.setenv("AI_SESSION_CACHE_BYTES", "1")
    create, made = counting_factory()
    session_cache.get_session("m", CUDA, create, 10 * MB)
    session_cache.get_session("m", CUDA, create, 10 * MB)
    assert len(made) == 2


def test_no_gpu_and_no_env_budget_means_no_caching(monkeypatch):
    monkeypatch.delenv("AI_SESSION_CACHE_BYTES")
    monkeypatch.setattr(session_cache, "_auto_budget", lambda: 0)
    create, made = counting_factory()
    session_cache.get_session("m", CPU, create, 1 * MB)
    session_cache.get_session("m", CPU, create, 1 * MB)
    assert len(made) == 2


def test_forced_cpu_override_zeroes_the_auto_budget(monkeypatch):
    monkeypatch.setenv("SNAPOTTER_GPU", "0")
    assert session_cache._auto_budget() == 0


def test_two_tools_exceeding_the_budget_evict_the_least_recently_used():
    create, _ = counting_factory()
    session_cache.get_session("a", CUDA, create, 60 * MB)
    session_cache.get_session("b", CUDA, create, 60 * MB)
    assert [k[0] for k in session_cache._entries] == ["b"]


def test_a_hit_refreshes_recency():
    create, _ = counting_factory()
    session_cache.get_session("a", CUDA, create, 40 * MB)
    session_cache.get_session("b", CUDA, create, 40 * MB)
    session_cache.get_session("a", CUDA, create, 40 * MB)
    session_cache.get_session("c", CUDA, create, 40 * MB)
    assert sorted(k[0] for k in session_cache._entries) == ["a", "c"]


def test_eviction_actually_releases_the_session():
    create, made = counting_factory()
    session_cache.get_session("a", CUDA, create, 60 * MB)
    ref = weakref.ref(made.pop())
    session_cache.get_session("b", CUDA, create, 60 * MB)
    assert ref() is None


def test_evict_all_clears_everything():
    create, _ = counting_factory()
    session_cache.get_session("a", CUDA, create, 10 * MB)
    session_cache.get_session("b", CUDA, create, 10 * MB)
    session_cache.evict_all("test")
    assert not session_cache._entries


def test_idle_eviction_drops_only_stale_entries(monkeypatch):
    monkeypatch.setenv("AI_SESSION_IDLE_TIMEOUT", "60")
    create, _ = counting_factory()
    session_cache.get_session("old", CUDA, create, 10 * MB)
    session_cache.get_session("new", CUDA, create, 10 * MB)
    now = session_cache._entries[("new", tuple(CUDA))].last_used
    session_cache._entries[("old", tuple(CUDA))].last_used = now - 61
    assert session_cache.evict_idle(now) == 1
    assert [k[0] for k in session_cache._entries] == ["new"]


def test_idle_eviction_never_runs_during_a_request(monkeypatch):
    monkeypatch.setenv("AI_SESSION_IDLE_TIMEOUT", "60")
    create, _ = counting_factory()
    session_cache.request_started()
    session_cache.get_session("m", CUDA, create, 10 * MB)
    entry = session_cache._entries[("m", tuple(CUDA))]
    assert session_cache.evict_idle(entry.last_used + 10_000) == 0
    session_cache.request_finished()
    # A long job must not count as idle time: finishing refreshes last_used.
    assert session_cache.evict_idle(entry.last_used) == 0
    assert session_cache._entries


def test_failed_create_leaves_no_entry():
    def boom():
        raise RuntimeError("load failed")

    with pytest.raises(RuntimeError):
        session_cache.get_session("m", CUDA, boom, 10 * MB)
    assert not session_cache._entries


def test_session_cost_scales_with_file_size_and_provider(tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x" * 1000)
    assert session_cache.session_cost(str(onnx), CUDA) == 1000 * 8
    assert session_cache.session_cost(str(onnx), CPU) == 1000 * 4
    assert session_cache.session_cost(str(tmp_path / "missing.onnx"), CUDA) is None


def test_env_parsing_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("AI_SESSION_IDLE_TIMEOUT", "soon")
    assert session_cache.idle_timeout_s() == session_cache.AUTO_IDLE_TIMEOUT_S
    monkeypatch.setenv("AI_SESSION_IDLE_TIMEOUT", "-5")
    assert session_cache.idle_timeout_s() == session_cache.AUTO_IDLE_TIMEOUT_S


def test_reaper_loop_never_evicts_a_session_mid_request(monkeypatch):
    import threading
    import time

    monkeypatch.setenv("AI_SESSION_IDLE_TIMEOUT", "1")
    create, made = counting_factory()
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            # Far-future clock: everything looks idle, so only the busy flag protects it.
            session_cache.evict_idle(time.monotonic() + 10_000)

    t = threading.Thread(target=hammer)
    t.start()
    try:
        for _ in range(200):
            session_cache.request_started()
            session = session_cache.get_session("m", CUDA, create, 10 * MB)
            time.sleep(0.001)
            assert session_cache.get_session("m", CUDA, create, 10 * MB) is session
            session_cache.request_finished()
    finally:
        stop.set()
        t.join()
