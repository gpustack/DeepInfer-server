"""Serialize engine operations and keep all DP ranks in the same execution round."""

import asyncio
import logging
import multiprocessing as mp
import os
import queue
import signal
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from uuid import uuid4

from .config import Settings
from .protocol import Prepared, ServiceError
from .worker import Engine, run_worker

logger = logging.getLogger(__name__)


def select_master_port(serving_port: int) -> str:
    explicit = os.environ.get("MASTER_PORT")
    if explicit is not None:
        try:
            port = int(explicit)
        except ValueError as exc:
            raise ValueError("MASTER_PORT must be an integer between 51 and 65535") from exc
        if not 51 <= port <= 65535 or port - 50 == serving_port:
            raise ValueError(
                "MASTER_PORT must be between 51 and 65535, with MASTER_PORT - 50 "
                "different from the HTTP port"
            )
        return str(port)
    # DeepInfer uses MASTER_PORT - 50 for the shared DP/CP rendezvous.
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        if port != serving_port and port <= 65535 - 50:
            return str(port + 50)
    raise RuntimeError("Could not select a distributed initialization port")


@dataclass
class Handle:
    request_id: str
    queue: asyncio.Queue


class Backend:
    def __init__(self, settings: Settings, *, engine_factory=Engine):
        self.settings = settings
        self.factory = engine_factory
        self.commands = queue.Queue()
        self.pending: dict[str, Handle] = {}
        self.processes = []
        self.connections: list[Connection] = []
        self.stopping = threading.Event()
        self.healthy = False
        self.thread = None

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.ready = self.loop.create_future()
        self.thread = threading.Thread(
            target=self._coordinate, name="engine-coordinator", daemon=True
        )
        self.thread.start()
        try:
            await asyncio.wait_for(asyncio.shield(self.ready), self.settings.startup_timeout + 5)
        except BaseException:
            await self.close()
            raise

    def submit(self, prepared: Prepared) -> Handle:
        if not self.healthy:
            raise ServiceError("Inference engine is unavailable", 503)
        capacity = self.settings.max_num_seqs * self.settings.data_parallel_size
        if len(self.pending) >= capacity:
            raise ServiceError("Too many active requests; retry later", 429)
        handle = Handle("chatcmpl-" + uuid4().hex, asyncio.Queue(maxsize=256))
        self.pending[handle.request_id] = handle
        self.commands.put(("add", handle.request_id, (prepared.token_ids, prepared.sampling)))
        return handle

    def cancel(self, handle: Handle) -> None:
        if self.pending.pop(handle.request_id, None) is not None:
            self.commands.put(("cancel", handle.request_id, None))

    def _deliver(self, request_id: str, updates: list[dict]) -> None:
        handle = self.pending.get(request_id)
        if handle is None:
            return
        try:
            handle.queue.put_nowait(updates)
        except asyncio.QueueFull:
            self.cancel(handle)
            while not handle.queue.empty():
                handle.queue.get_nowait()
            handle.queue.put_nowait(ServiceError("Client is consuming output too slowly", 503))

    def _set_ready(self) -> None:
        if not self.ready.done() and not self.stopping.is_set():
            self.healthy = True
            self.ready.set_result(None)

    def _fail(self, message: str) -> None:
        self.healthy = False
        error = ServiceError(message, 503)
        if not self.ready.done():
            self.ready.set_exception(error)
        for handle in self.pending.values():
            while not handle.queue.empty():
                handle.queue.get_nowait()
            handle.queue.put_nowait(error)

    def _receive(self, connection: Connection, deadline: float):
        while time.monotonic() < deadline:
            if self.stopping.is_set():
                raise InterruptedError("Service is stopping")
            if connection.poll(min(0.1, max(0, deadline - time.monotonic()))):
                try:
                    kind, data = connection.recv()
                except EOFError as exc:
                    raise RuntimeError("DeepInfer worker exited; check server logs") from exc
                if kind == "fatal":
                    raise RuntimeError(data)
                return kind, data
        raise TimeoutError("DeepInfer worker timed out; check server logs")

    def _coordinate(self) -> None:
        try:
            context = mp.get_context("spawn")
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            devices = (
                visible.split(",") if visible else [str(i) for i in range(self.settings.world_size)]
            )
            if len(devices) < self.settings.world_size or any(
                not device.isdigit() for device in devices
            ):
                raise ValueError("CUDA_VISIBLE_DEVICES must contain enough numeric GPU indices")
            if len(set(devices)) != len(devices):
                raise ValueError("CUDA_VISIBLE_DEVICES must not contain duplicate indices")
            master_port = select_master_port(self.settings.port)
            logger.info("DeepInfer distributed initialization port: %s", int(master_port) - 50)
            for rank in range(self.settings.data_parallel_size):
                parent, child = context.Pipe()
                cp = self.settings.context_parallel_size
                process = context.Process(
                    target=run_worker,
                    args=(
                        child,
                        self.settings,
                        rank,
                        devices[rank * cp : (rank + 1) * cp],
                        self.factory,
                        master_port,
                    ),
                    name=f"deepinfer-dp-{rank}",
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            deadline = time.monotonic() + self.settings.startup_timeout
            for connection in self.connections:
                kind, _ = self._receive(connection, deadline)
                if kind != "ready":
                    raise RuntimeError("Worker did not signal readiness")
            self.loop.call_soon_threadsafe(self._set_ready)
            routes: dict[str, int] = {}
            active = False
            while not self.stopping.is_set():
                commands = []
                try:
                    commands.append(self.commands.get(timeout=0 if active else 0.1))
                except queue.Empty:
                    if not active:
                        if any(not process.is_alive() for process in self.processes):
                            raise RuntimeError("DeepInfer worker exited") from None
                        continue
                while True:
                    try:
                        commands.append(self.commands.get_nowait())
                    except queue.Empty:
                        break
                per_rank = [[] for _ in self.connections]
                for operation, request_id, payload in commands:
                    if operation == "add":
                        loads = [list(routes.values()).count(i) for i in range(len(per_rank))]
                        rank = min(range(len(loads)), key=loads.__getitem__)
                        routes[request_id] = rank
                        per_rank[rank].append((operation, request_id, payload))
                    elif request_id in routes:
                        rank = routes.pop(request_id)
                        per_rank[rank].append((operation, request_id, None))
                # Send to every rank before receiving: each step can enter DP collectives.
                for connection, operations in zip(self.connections, per_rank, strict=True):
                    connection.send(("tick", operations))
                deadline = time.monotonic() + self.settings.step_timeout
                active = False
                grouped: dict[str, list[dict]] = {}
                for connection in self.connections:
                    kind, result = self._receive(connection, deadline)
                    if kind != "result":
                        raise RuntimeError("Unexpected worker response")
                    still_active, outputs = result
                    active = active or still_active
                    for output in outputs:
                        request_id = output["request_id"]
                        grouped.setdefault(request_id, []).append(output)
                        if output.get("finished") or "error" in output:
                            routes.pop(request_id, None)
                for request_id, outputs in grouped.items():
                    self.loop.call_soon_threadsafe(self._deliver, request_id, outputs)
        except Exception as exc:
            if not self.stopping.is_set():
                logger.exception("Inference coordinator failed")
            self.loop.call_soon_threadsafe(self._fail, str(exc))
        finally:
            self._shutdown_workers()

    def _signal_workers(self, sig: int) -> None:
        for process in self.processes:
            if process.pid is None:
                continue
            try:
                # Each driver creates a session shared by its CP and preprocessing children.
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                if process.is_alive():
                    with suppress(ProcessLookupError):
                        os.kill(process.pid, sig)

    def _shutdown_workers(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("stop", None))
            except (OSError, BrokenPipeError):
                pass
        deadline = time.monotonic() + 2
        for process in self.processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        self._signal_workers(signal.SIGTERM)
        for process in self.processes:
            process.join(timeout=0.2)
        self._signal_workers(signal.SIGKILL)
        for process in self.processes:
            process.join(timeout=1)
        for connection in self.connections:
            connection.close()

    async def close(self) -> None:
        self.healthy = False
        self.stopping.set()
        if self.thread is not None:
            await asyncio.to_thread(self.thread.join, 5)
            if self.thread.is_alive():
                self._signal_workers(signal.SIGKILL)
                await asyncio.to_thread(self.thread.join, 3)
