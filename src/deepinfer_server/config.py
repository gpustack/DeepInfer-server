from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Settings:
    model: str
    served_model_name: str = "deepseek-flash"
    host: str = "0.0.0.0"
    port: int = 8000
    expert_parallel_size: int = 8
    data_parallel_size: int = 1
    context_parallel_size: int = 8
    max_model_len: int = 32768
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 32768
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = False
    startup_timeout: float = 1800
    request_timeout: float = 600
    step_timeout: float = 300

    def validate(self) -> None:
        for name in (
            "expert_parallel_size",
            "data_parallel_size",
            "context_parallel_size",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "startup_timeout",
            "request_timeout",
            "step_timeout",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.model or not self.served_model_name:
            raise ValueError("model and served_model_name must not be empty")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.world_size != self.expert_parallel_size:
            raise ValueError("This service requires DP × CP = EP on a single node")

    @property
    def world_size(self) -> int:
        return self.data_parallel_size * self.context_parallel_size
