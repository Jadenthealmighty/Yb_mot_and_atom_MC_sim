"""Terminal progress.

Set RUNLOG_ROOT to put the run folders somewhere other than next to this
file, or RUNLOG_LOG_EVERY_S to change the metric print throttle.
"""

import contextlib
import datetime
import json
import os
import time
import traceback

__all__ = ["start", "new_run_dir", "Run"]

RUNS_ROOT = os.environ.get(
    "RUNLOG_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
LOG_EVERY_S = float(os.environ.get("RUNLOG_LOG_EVERY_S", 10.0))


def _jsonable(obj):
    """Best-effort conversion of numpy scalars/arrays into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "size", 1) == 1:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


class Run:
    """Handle to one run's output folder. Made by `start()`, not directly."""

    def __init__(self, name, directory, params=None, note=None):
        self.name = name
        self.dir = directory
        self.t_start = time.time()
        self._finished = False
        self._last_log_print = 0.0
        self._files = []
        os.makedirs(self.dir, exist_ok=True)
        self._transcript = open(os.path.join(self.dir, "run.log"), "a",
                                buffering=1)
        self._metrics = open(os.path.join(self.dir, "metrics.jsonl"), "a",
                             buffering=1)
        self._emit("=" * 72)
        self._emit(f"{name}  ->  {self.dir}")
        if note:
            self._emit(note)
        self._emit("=" * 72)
        if params:
            with open(os.path.join(self.dir, "params.json"), "w") as fh:
                json.dump(_jsonable(params), fh, indent=2, sort_keys=True)

    def out(self, filename):
        """Absolute path to `filename` inside this run's folder."""
        path = os.path.join(self.dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def say(self, message):
        """Print a line of progress, timestamped by elapsed seconds."""
        self._emit(str(message), stamp=True)

    def log(self, step=None, **metrics):
        """Record named scalars. Written every call, printed on a throttle."""
        row = {"t_s": round(time.time() - self.t_start, 3)}
        if step is not None:
            row["step"] = _jsonable(step)
        row.update(_jsonable(metrics))
        self._metrics.write(json.dumps(row) + "\n")
        now = time.time()
        if now - self._last_log_print >= LOG_EVERY_S:
            self._last_log_print = now
            body = "  ".join(f"{k}={_fmt(v)}" for k, v in row.items()
                             if k != "t_s")
            self._emit("    " + body, stamp=True)

    def figure(self, fig, name, close=True, dpi=150):
        """Save a matplotlib figure as <name>.png in the run folder."""
        path = self.out(f"{name}.png")
        fig.savefig(path, dpi=dpi)
        if close:
            import matplotlib.pyplot as plt
            plt.close(fig)
        self._files.append(os.path.basename(path))
        self._emit(f"  [figure] {os.path.basename(path)}", stamp=True)
        return path

    def file(self, filename, label=""):
        """Note a data file already written into the run folder."""
        path = filename if os.path.isabs(filename) else self.out(filename)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        self._files.append(os.path.basename(path))
        self._emit(f"  [file] {os.path.basename(path)} "
                   f"({size / 1e6:.1f} MB){'  ' + label if label else ''}",
                   stamp=True)
        return path

    def finish(self, summary=None):
        """Write summary.json, print it, and close the transcript."""
        if self._finished:
            return
        self._finished = True
        elapsed = time.time() - self.t_start
        if summary:
            summary = _jsonable(summary)
            with open(os.path.join(self.dir, "summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
            self._emit("")
            self._emit("-" * 72)
            self._emit("SUMMARY")
            width = max(len(k) for k in summary)
            for key in sorted(summary):
                self._emit(f"  {key:<{width}}  {_fmt(summary[key])}")
        self._emit("-" * 72)
        self._emit(f"{len(self._files)} files written to {self.dir}")
        self._emit(f"finished in {elapsed / 60:.1f} min")
        self._transcript.close()
        self._metrics.close()

    def _emit(self, text, stamp=False):
        if stamp:
            text = f"[{time.time() - self.t_start:7.1f}s] {text}"
        print(text, flush=True)
        if not self._transcript.closed:
            self._transcript.write(text + "\n")


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)) and len(value) > 6:
        return f"[{len(value)} values]"
    return str(value)


def new_run_dir(name):
    """Create timestamp folder."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(RUNS_ROOT, f"{name}_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


@contextlib.contextmanager
def start(name, params=None, note=None):
    run = Run(name, new_run_dir(name), params, note)
    try:
        yield run
    except BaseException:
        run._emit("")
        run._emit("RUN FAILED:")
        run._emit(traceback.format_exc())
        run.finish()
        raise
    else:
        run.finish()
