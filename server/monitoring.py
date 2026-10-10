"""Thread-safe state of the active demo, separate from preload/upload previews."""
from copy import deepcopy
from datetime import datetime, timezone
import math
import threading
import time
from uuid import uuid4

from .knowledge import PHASES, SENSORS


class MonitorState:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._state = {"session_id": None, "status": "idle", "source": None,
                           "filename": None, "timestamp": None, "condition": "unknown",
                           "metrics": {}, "received_at": None}
            self._received = None

    def start(self, source="simulator", filename=None):
        with self._lock:
            if self._state['status'] in ('starting', 'running'):
                raise ValueError('已有监测会话运行，请先停止后再开始')
            self._state = dict(session_id=uuid4().hex, status="starting", source=source,
                               filename=filename, timestamp=None, condition="unknown",
                               metrics={}, received_at=None)
            self._received = None
            return self._state["session_id"]

    def record(self, row, session_id):
        metrics = {}
        for sensor in SENSORS:
            value = row.get(sensor['key'])
            try:
                value = float(value) if value is not None and not isinstance(value, bool) else None
            except (ValueError, TypeError, OverflowError):
                value = None
            metrics[sensor['key']] = value if value is not None and math.isfinite(value) else None
        condition = str(row.get("operating_condition", "unknown"))
        with self._lock:
            if (session_id != self._state['session_id'] or
                    self._state['status'] not in ('starting', 'running')):
                return
            self._state.update(status="running", timestamp=str(row.get("timestamp", "")),
                               condition=condition if condition in PHASES else "unknown", metrics=metrics,
                               received_at=datetime.now(timezone.utc).isoformat())
            self._received = time.monotonic()

    def finish(self, session_id, status="stopped"):
        with self._lock:
            if session_id and session_id == self._state['session_id']:
                self._state['status'] = status

    def snapshot(self):
        with self._lock:
            state = deepcopy(self._state)
            age = None if self._received is None else max(0, time.monotonic() - self._received)
        state.update(age_seconds=round(age, 1) if age is not None else None,
                     fresh=state['status'] == 'running' and age is not None and age <= 30,
                     condition_label=PHASES.get(state['condition'], PHASES['unknown']))
        return state
