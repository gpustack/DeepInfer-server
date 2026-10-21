import asyncio
import os
import time
from dataclasses import dataclass
from uuid import uuid4

from deepinfer_server.backend import Handle


class FakeBackend:
    def __init__(
        self, tokenizer, text="Hello, world! 🌍", *, reason="stop", stall=False, error=None
    ):
        self.tokenizer = tokenizer
        self.text = text
        self.reason = reason
        self.stall = stall
        self.error = error
        self.healthy = False
        self.cancelled = []
        self.submitted = []

    async def start(self):
        self.healthy = True

    async def close(self):
        self.healthy = False

    def submit(self, prepared):
        handle = Handle("chatcmpl-" + uuid4().hex, asyncio.Queue())
        self.submitted.append(prepared)
        if self.error:
            handle.queue.put_nowait(self.error)
        elif not self.stall:
            tokens = self.tokenizer.encode(self.text)
            for i, token in enumerate(tokens):
                handle.queue.put_nowait(
                    [
                        {
                            "request_id": handle.request_id,
                            "token_id": token,
                            "finished": i == len(tokens) - 1,
                            "finish_reason": self.reason if i == len(tokens) - 1 else "",
                        }
                    ]
                )
        return handle

    def cancel(self, handle):
        self.cancelled.append(handle.request_id)


@dataclass
class EngineFactory:
    barrier: object = None
    events: object = None
    crash: bool = False
    report_master_port: bool = False

    def __call__(self, settings, rank, devices):
        if self.report_master_port:
            self.events.put((rank, os.environ["MASTER_PORT"]))
        return FakeEngine(self, rank)


class FakeEngine:
    def __init__(self, factory, rank):
        self.factory = factory
        self.rank = rank
        self.requests = {}

    def add(self, request_id, token_ids, sampling):
        self.requests[request_id] = 0
        if self.factory.events is not None:
            self.factory.events.put(("add", self.rank, request_id))

    def abort(self, request_id):
        self.requests.pop(request_id, None)
        if self.factory.events is not None:
            self.factory.events.put(("cancel", self.rank, request_id))

    def tick(self):
        if self.factory.crash:
            os._exit(3)
        if self.factory.barrier is not None:
            self.factory.barrier.wait(timeout=3)
        time.sleep(0.02)
        outputs = []
        active = bool(self.requests)
        for request_id in list(self.requests):
            self.requests[request_id] += 1
            finished = self.requests[request_id] >= 4
            outputs.append(
                {
                    "request_id": request_id,
                    "token_id": 65,
                    "finished": finished,
                    "finish_reason": "stop" if finished else "",
                }
            )
            if finished:
                self.requests.pop(request_id)
        return active, outputs

    def close(self):
        pass
