"""Exercise a running server with non-streaming and streaming requests."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    for path in ("/health", "/v1/models"):
        request = urllib.request.Request(base + path, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            print(path, json.load(response))
    for stream in (False, True):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "thinking": {"type": "disabled"},
            "max_tokens": 64,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            if not stream:
                result = json.load(response)
                assert result["object"] == "chat.completion", result
                assert result["choices"][0]["message"]["content"], result
                assert "<｜end▁of▁sentence｜>" not in result["choices"][0]["message"]["content"]
                print(json.dumps(result, ensure_ascii=False))
                continue
            done = False
            content = ""
            usage = None
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                chunk = json.loads(line[6:])
                assert "error" not in chunk, chunk
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    content += choice["delta"].get("content") or ""
            assert done and content and usage, (done, content, usage)
            assert "<｜end▁of▁sentence｜>" not in content
            print("stream:", content, usage)
    print("Smoke test passed")


if __name__ == "__main__":
    main()
