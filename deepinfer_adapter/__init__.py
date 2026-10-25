# SPDX-License-Identifier: Apache-2.0
"""DeepInfer adapter: expose a DeepInfer engine as a vLLM-style gRPC worker.

The adapter speaks the subset of the vLLM worker contract that a router needs on
a ``grpc://`` worker, and delegates generation to a pluggable backend.
"""

from __future__ import annotations

__version__ = "0.2.0"
