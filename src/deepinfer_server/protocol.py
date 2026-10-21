from dataclasses import dataclass
from typing import Any

from deepseek_recipe import (
    ChatCompletionRequest,
    ConversationRequest,
    ConversionOptions,
    DeepseekV41Encoding,
    ModelVersion,
    SpecialToken,
    Tokenizer,
)

from .config import Settings


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 400, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.param = param

    def body(self) -> dict:
        return {
            "error": {
                "message": str(self),
                "type": "invalid_request_error" if self.status < 500 else "server_error",
                "param": self.param,
                "code": None,
            }
        }


@dataclass
class Prepared:
    converted: ConversationRequest
    token_ids: list[int]
    sampling: dict[str, Any]
    model: str
    stream: bool
    include_usage: bool
    continuous_usage_stats: bool


class Protocol:
    def __init__(self, settings: Settings, tokenizer_path: str):
        self.settings = settings
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        eos_ids = self.tokenizer.encode(SpecialToken.Eos.default_str())
        if len(eos_ids) != 1:
            raise ValueError("Tokenizer must encode EOS as a single token")
        self.eos_token_id = eos_ids[0]
        self.encoding = DeepseekV41Encoding().with_tokenizer(self.tokenizer)
        self.options = ConversionOptions().with_model_versions_by_name(
            {
                settings.served_model_name: ModelVersion.DeepseekV41Flash,
            }
        )

    def prepare(self, body: Any) -> Prepared:
        if not isinstance(body, dict):
            raise ServiceError("Request body must be a JSON object")
        if body.get("model") != self.settings.served_model_name:
            raise ServiceError(f"Model must be {self.settings.served_model_name!r}", 404, "model")
        allowed = {
            "model",
            "messages",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "seed",
            "ignore_eos",
            "thinking",
            "reasoning_effort",
            "tools",
            "tool_choice",
            "n",
            "user",
            "response_format",
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "top_logprobs",
        }
        unknown = set(body) - allowed
        if unknown:
            raise ServiceError(f"Unsupported parameter: {sorted(unknown)[0]}")
        if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
            raise ServiceError("Only n=1 is supported", param="n")
        for key in ("frequency_penalty", "presence_penalty", "top_logprobs"):
            if body.get(key, 0) not in (None, 0):
                raise ServiceError(f"{key} is not supported", param=key)
        if body.get("logprobs") not in (None, False):
            raise ServiceError("logprobs is not supported", param="logprobs")
        fmt = body.get("response_format")
        if fmt is not None and fmt != {"type": "text"}:
            raise ServiceError(
                "Only response_format.type=text is supported", param="response_format"
            )
        if body.get("tool_choice", "auto") not in ("auto", "none"):
            raise ServiceError("Only tool_choice=auto or none is supported", param="tool_choice")
        tools = body.get("tools") or []
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                    if tool["function"].get("strict"):
                        raise ServiceError("Strict tool schemas are not supported", param="tools")
        if type(body.get("stream", False)) is not bool:
            raise ServiceError("stream must be a boolean", param="stream")
        stream_options = body.get("stream_options")
        if stream_options is None:
            stream_options = {}
        if not isinstance(stream_options, dict):
            raise ServiceError("stream_options must be an object", param="stream_options")
        unknown_stream_options = set(stream_options) - {"include_usage", "continuous_usage_stats"}
        if unknown_stream_options:
            key = sorted(unknown_stream_options)[0]
            raise ServiceError(f"Unsupported stream option: {key}", param=f"stream_options.{key}")
        for key in ("include_usage", "continuous_usage_stats"):
            if type(stream_options.get(key, False)) is not bool:
                raise ServiceError(f"{key} must be a boolean", param=f"stream_options.{key}")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ServiceError("messages must be a nonempty array", param="messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ServiceError("Each message must be an object", param="messages")
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if (
                        not isinstance(part, dict)
                        or part.get("type") != "text"
                        or not isinstance(part.get("text"), str)
                    ):
                        raise ServiceError("Only text content is supported", param="messages")
        body = dict(body)
        body.pop("stream_options", None)
        ignore_eos = body.pop("ignore_eos", False)
        if type(ignore_eos) is not bool:
            raise ServiceError("ignore_eos must be a boolean", param="ignore_eos")
        if "max_completion_tokens" in body:
            if "max_tokens" in body:
                raise ServiceError("Specify only one of max_tokens and max_completion_tokens")
            body["max_tokens"] = body.pop("max_completion_tokens")
        converted = ChatCompletionRequest(body).convert(self.options)
        token_ids = self.encoding.encode(converted.conversation)
        max_tokens = converted.resolve_max_tokens(len(token_ids), self.settings.max_model_len)
        if len(token_ids) + max_tokens > self.settings.max_model_len:
            raise ServiceError("The prompt leaves no room for output within max_model_len")
        options = converted.inference_options
        sampling = {
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": max_tokens,
            "seq_seed": options.seed,
            "ignore_eos": ignore_eos,
        }
        return Prepared(
            converted,
            token_ids,
            sampling,
            body["model"],
            converted.stream,
            stream_options.get("include_usage", False),
            stream_options.get("continuous_usage_stats", False),
        )
