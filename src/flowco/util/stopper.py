import threading


class Stopper:

    def __init__(self):
        self._stop_flag = False
        self._nesting_level = 0
        self._lock = threading.RLock()

    def __enter__(self):
        with self._lock:
            if self._nesting_level == 0:
                self._stop_flag = False
            self._nesting_level += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            self._nesting_level -= 1

    def stop(self):
        with self._lock:
            self._stop_flag = True

    def should_stop(self):
        with self._lock:
            return self._stop_flag
