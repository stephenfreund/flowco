import threading
from flowco.util.output import log, session


class CostTracker:
    _lock = threading.RLock()  # Lock to ensure thread safety

    def __init__(self):
        self._total_cost = 0.0
        self._call_count = 0

    def total_cost(self):
        """Accessor for total cost."""
        with CostTracker._lock:
            return self._total_cost

    def call_count(self):
        """Accessor for call count."""
        with CostTracker._lock:
            return self._call_count

    def add_cost(self, amount, count=1):
        log(f"Adding cost: {amount:.3f}")
        with CostTracker._lock:
            self._total_cost += amount
            self._call_count += count


def add_cost(*args, **kwargs):
    return session.get("costs", CostTracker).add_cost(*args, **kwargs)


def total_cost(*args, **kwargs):
    return session.get("costs", CostTracker).total_cost(*args, **kwargs)


def call_count(*args, **kwargs):
    return session.get("costs", CostTracker).call_count(*args, **kwargs)
