from dataclasses import replace

import pytest
from deepseek_recipe import SpecialToken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from deepinfer_server.config import Settings
from deepinfer_server.protocol import Protocol


@pytest.fixture(scope="session")
def tokenizer_path(tmp_path_factory):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [
            getattr(SpecialToken, name).default_str()
            for name in dir(SpecialToken)
            if name[0].isupper()
        ]
    )
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))
    return str(path)


@pytest.fixture
def settings():
    return Settings(
        model="/unused",
        expert_parallel_size=1,
        context_parallel_size=1,
        max_model_len=4096,
        startup_timeout=10,
        step_timeout=5,
    )


@pytest.fixture
def protocol(settings, tokenizer_path):
    return Protocol(settings, tokenizer_path)


@pytest.fixture
def body():
    return {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hello"}],
        "thinking": {"type": "disabled"},
        "max_tokens": 128,
    }


@pytest.fixture
def tiny_settings(settings):
    return replace(settings, max_model_len=8)
