"""Send token IDs through vLLM Router; raw text is not a chat template."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:13010/v1/completions")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--model-path")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--tokens-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    if args.tokens_file:
        tokens = json.loads(args.tokens_file.read_text())
    else:
        if not args.model_path:
            parser.error("--model-path is required for --text")
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(args.model_path) / "tokenizer.json"))
        tokens = tokenizer.encode(args.text, add_special_tokens=False).ids
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        parser.error("prompt must be a non-empty array of non-negative integer token IDs")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    body = {"model": args.model, "prompt": tokens, "temperature": 0,
            "max_tokens": args.max_tokens, "stream": args.stream}
    request = urllib.request.Request(args.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        if args.stream:
            for line in response:
                print(line.decode(), end="", flush=True)
        else:
            print(json.dumps(json.load(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
