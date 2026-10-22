"""Skip NUMA binding on hosts whose GPUs report no NUMA node.

Enable with ``ENABLE_NUMA_SHIM=1`` when starting ``scripts/run_adapter.sh``,
which prepends this directory to ``PYTHONPATH`` so that this ``sitecustomize``
runs at interpreter startup.

On such hosts every ``/sys/class/drm/card*/device/numa_node`` reads ``-1``, so
``deep_cluster.topology.which_numa()`` returns ``-1`` and libnuma segfaults
inside ``numa_node_to_cpus(-1, ...)`` before any exception can be raised. The
``try/except`` in ``deepinfer.worker.base._bind_numa`` therefore cannot help:
the process dies inside the ctypes call.

When the host exposes a single NUMA node covering every CPU, the bind that would
have run is a no-op anyway, so skipping it only forgoes a first-touch
optimization.
"""

try:
    from deep_cluster import topology
    from deep_cluster.topology import bind as _bind
except Exception:  # pragma: no cover - the shim must never break startup
    pass
else:

    def bind_numa(node: int) -> int:  # noqa: ARG001
        return -1

    topology.bind_numa = bind_numa
    _bind.bind_numa = bind_numa
