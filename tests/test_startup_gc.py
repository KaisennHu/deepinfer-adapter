"""Run the real collector in a subprocess to avoid freezing the test runner."""
import os
from pathlib import Path
import subprocess
import sys


def test_runtime_cycles_are_reclaimed_after_startup_freeze():
    source = Path(__file__).resolve().parents[1]
    code = '''
import gc, json, weakref
from deepinfer_adapter.startup_gc import freeze_startup_objects
class Object:
    pass
gc.disable()
engine = Object()
engine.cycle = engine
engine.pending = {}
engine_ref = weakref.ref(engine)
report = freeze_startup_objects()
assert report['frozen_objects'] > 0 and not report['automatic_gc_enabled']
for i in range(100):
    request = Object()
    request.cycle = request
    engine.pending[i] = request
    ref = weakref.ref(request)
    del request
    del engine.pending[i]
    assert ref() is not None
    gc.collect()
    assert ref() is None
assert gc.get_freeze_count() == report['frozen_objects']
# Noncyclic runtime resources retain ordinary reference-counted cleanup.
resource = Object()
ref = weakref.ref(resource)
del resource
assert ref() is None
# Startup cycles intentionally survive, but are reclaimable when unfrozen.
del engine
gc.collect()
assert engine_ref() is not None
gc.unfreeze()
gc.collect()
assert engine_ref() is None
print(json.dumps(dict(report, reclaimed_runtime_cycles=100, reference_count_cleanup=True, startup_lifetime_verified=True)))
'''
    result = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=str(source)),
                            text=True, capture_output=True, check=True)
    print(result.stdout)
