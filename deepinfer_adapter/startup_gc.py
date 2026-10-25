# SPDX-License-Identifier: Apache-2.0
"""Exclude process-lifetime startup objects from periodic cyclic GC scans."""
import gc
import time


def freeze_startup_objects() -> dict:
    """Call once after engine construction, before accepting any requests.

    Frozen cycles live until process exit. Request objects created afterward
    remain eligible for the engine's existing periodic collections.
    """
    started = time.perf_counter()
    collected = gc.collect()
    gc.freeze()
    return {"collected_before_freeze": collected, "frozen_objects": gc.get_freeze_count(),
            "seconds": time.perf_counter() - started, "automatic_gc_enabled": gc.isenabled()}
