# stage timer. splits every answer's time into "our code" vs "waiting on the model api"
import time
from contextlib import contextmanager


class Timing:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.stages = []   # (name, seconds, is_model)

    @contextmanager
    def stage(self, name):
        """times a block. anything timed inside it (a model call, a smaller stage) is taken out, so stages never double count"""
        s, n0 = time.perf_counter(), len(self.stages)
        try:
            yield
        finally:
            inner = sum(x for _, x, _m in self.stages[n0:])
            self.stages.append((name, max(0.0, time.perf_counter() - s - inner), False))

    def model(self, name, seconds):
        """a model call. its time came out of an enclosing program stage, so take it back out of that"""
        self.stages.append((name, seconds, True))

    def report(self):
        total = time.perf_counter() - self.t0
        model = sum(s for _, s, m in self.stages if m)
        return {"total": round(total, 3), "model": round(model, 3), "program": round(max(0.0, total - model), 3),
                "stages": [{"name": n, "ms": round(s * 1000, 1), "model": m} for n, s, m in self.stages]}
