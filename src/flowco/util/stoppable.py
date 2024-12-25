import threading

from flowco.util.output import log


class Stoppable:
    _stop_flag = False
    _nesting_level = 0
    _lock = threading.RLock()

    @classmethod
    def __enter__(cls):
        with cls._lock:
            cls._nesting_level += 1
        return cls

    @classmethod
    def __exit__(cls, exc_type, exc_value, traceback):
        with cls._lock:
            cls._nesting_level -= 1
            if cls._nesting_level == 0:
                cls._stop_flag = False

    @classmethod
    def stop(cls):
        with cls._lock:
            log("Stopping...")
            cls._stop_flag = True

    @classmethod
    def should_stop(cls):
        with cls._lock:
            # log("Should stop?", cls._stop_flag)
            return cls._stop_flag
