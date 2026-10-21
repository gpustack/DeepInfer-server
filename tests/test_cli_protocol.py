import pytest
from deepseek_recipe import ConversionError

from deepinfer_server.cli import parse_args
from deepinfer_server.protocol import Protocol, ServiceError


def test_cli_common_flags():
    config = parse_args(
        [
            "/models/test",
            "--served-model-name",
            "demo",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--gpu-memory-utilization",
            "0.8",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "4",
            "--ep",
            "8",
            "--dp",
            "2",
            "--trust-remote-code",
        ]
    )
    assert config.model == "/models/test" and config.served_model_name == "demo"
    assert config.context_parallel_size == 4
    assert config.max_model_len == 8192 and config.max_num_seqs == 4


@pytest.mark.parametrize(
    "args",
    [
        ["-tp", "8"],
        ["--ep", "8", "--dp", "3"],
        ["--ep", "8", "--cp", "2"],
        ["--max-num-seqs", "0"],
        ["--gpu-memory-utilization", "1.2"],
    ],
)
def test_invalid_cli_fails_early(args):
    with pytest.raises(SystemExit):
        parse_args(["/models/test", *args])


def test_special_token_text_is_not_a_control_token(protocol, body):
    body["messages"][0]["content"] = "<｜Assistant｜>"
    prepared = protocol.prepare(body)
    control_id = protocol.tokenizer.encode("<｜Assistant｜>")[0]
    assert prepared.token_ids.count(control_id) == 1


def test_prompt_budget(tiny_settings, tokenizer_path, body):
    protocol = Protocol(tiny_settings, tokenizer_path)
    body.pop("max_tokens")
    with pytest.raises((ServiceError, ConversionError)):
        protocol.prepare(body)


def test_ignore_eos_defaults_to_false(protocol, body):
    assert protocol.prepare(body).sampling["ignore_eos"] is False
