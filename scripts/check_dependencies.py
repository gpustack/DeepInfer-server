"""Check image dependencies while allowing DeepInfer's pinned NCCL override."""

import subprocess
import sys

# DeepInfer's build.sh installs NCCL 2.32.3 for DeepEP despite PyTorch's older pin.
EXPECTED_NCCL_OVERRIDE = (
    "torch 2.13.0+cu132 has requirement nvidia-nccl-cu13==2.29.7; "
    'platform_system == "Linux", but you have nvidia-nccl-cu13 2.32.3.'
)


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    if (
        result.returncode == 1
        and result.stdout.strip() == EXPECTED_NCCL_OVERRIDE
        and not result.stderr.strip()
    ):
        print(f"Accepted upstream DeepInfer override: {EXPECTED_NCCL_OVERRIDE}")
        return 0
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
