import asyncio
import multiprocessing as mp
import socket
from dataclasses import replace

import pytest

from deepinfer_server.backend import Backend, select_master_port
from deepinfer_server.protocol import ServiceError

from .fakes import EngineFactory


async def drain(handle):
    tokens = []
    async with asyncio.timeout(5):
        while True:
            updates = await handle.queue.get()
            if isinstance(updates, Exception):
                raise updates
            for update in updates:
                tokens.append(update["token_id"])
                if update["finished"]:
                    return tokens


async def test_coordinator_drives_idle_ranks(settings, protocol, body, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    settings = replace(settings, expert_parallel_size=2, data_parallel_size=2)
    context = mp.get_context("spawn")
    factory = EngineFactory(barrier=context.Barrier(2))
    backend = Backend(settings, engine_factory=factory)
    await backend.start()
    try:
        first = backend.submit(protocol.prepare(body))
        assert await drain(first) == [65] * 4
        backend.cancel(first)
        # New arrivals must resume every DP rank after an idle interval.
        await asyncio.sleep(0.2)
        handles = [backend.submit(protocol.prepare(body)) for _ in range(3)]
        assert await asyncio.gather(*(drain(h) for h in handles)) == [[65] * 4] * 3
        for handle in handles:
            backend.cancel(handle)
    finally:
        await backend.close()
    assert not any(p.is_alive() for p in backend.processes)
    assert not backend.thread.is_alive()


async def test_cancel_and_capacity(settings, protocol, body, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    context = mp.get_context("spawn")
    events = context.Queue()
    backend = Backend(
        replace(settings, max_num_seqs=1), engine_factory=EngineFactory(events=events)
    )
    await backend.start()
    try:
        handle = backend.submit(protocol.prepare(body))
        with pytest.raises(ServiceError) as exc:
            backend.submit(protocol.prepare(body))
        assert exc.value.status == 429
        await asyncio.wait_for(handle.queue.get(), 5)
        backend.cancel(handle)
        messages = [await asyncio.to_thread(events.get, True, 5) for _ in range(2)]
        assert [m[0] for m in messages] == ["add", "cancel"]
        next_handle = backend.submit(protocol.prepare(body))
        assert await drain(next_handle) == [65] * 4
        backend.cancel(next_handle)
    finally:
        await backend.close()
        events.close()


async def test_crashed_worker_fails_pending_requests(settings, protocol, body, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = Backend(settings, engine_factory=EngineFactory(crash=True))
    await backend.start()
    try:
        handle = backend.submit(protocol.prepare(body))
        with pytest.raises(ServiceError):
            await drain(handle)
        assert not backend.healthy
    finally:
        await backend.close()


async def test_slow_consumer_is_cancelled(settings):
    backend = Backend(settings)
    backend.healthy = True
    from types import SimpleNamespace

    handle = backend.submit(SimpleNamespace(token_ids=[0], sampling={}))
    for _ in range(257):
        backend._deliver(handle.request_id, [{"token_id": 1}])
    result = handle.queue.get_nowait()
    assert isinstance(result, ServiceError)
    assert handle.request_id not in backend.pending


def test_master_port_is_available(monkeypatch):
    monkeypatch.delenv("MASTER_PORT", raising=False)
    with socket.socket() as occupied:
        occupied.bind(("", 0))
        occupied.listen()
        port = int(select_master_port(occupied.getsockname()[1])) - 50
        assert port != occupied.getsockname()[1]
        with socket.socket() as probe:
            probe.bind(("", port))


@pytest.mark.parametrize("value", ["invalid", "50", "65536", "8050"])
def test_invalid_master_port(monkeypatch, value):
    monkeypatch.setenv("MASTER_PORT", value)
    with pytest.raises(ValueError, match="MASTER_PORT"):
        select_master_port(8000)


@pytest.mark.parametrize("explicit", [None, "31050"])
async def test_dp_workers_share_master_port(settings, monkeypatch, explicit):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if explicit is None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
    else:
        monkeypatch.setenv("MASTER_PORT", explicit)
    context = mp.get_context("spawn")
    events = context.Queue()
    backend = Backend(
        replace(settings, expert_parallel_size=2, data_parallel_size=2),
        engine_factory=EngineFactory(events=events, report_master_port=True),
    )
    await backend.start()
    try:
        reports = [await asyncio.to_thread(events.get, True, 5) for _ in range(2)]
        ports = dict(reports)
        assert ports[0] == ports[1]
        if explicit is not None:
            assert ports[0] == explicit
        else:
            assert 51 <= int(ports[0]) <= 65535
    finally:
        await backend.close()
        events.close()
