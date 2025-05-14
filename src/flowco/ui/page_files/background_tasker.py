import threading
import queue

from flowco.util.output import log, warn


class Task:
    def run(self):
        pass

    def complete(self):
        pass


class BackgroundTaskRunner:
    def __init__(self):
        self.task_queue = queue.Queue[Task]()
        self.finished_queue = queue.Queue[Task]()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                try:
                    task.run()
                    self.finished_queue.put(task)
                except Exception as e:
                    warn(e)
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue

    def enqueue(self, task: Task) -> None:
        self.task_queue.put(task)
        log(f"Task enqueued: {task}")

    def dequeue(self, timeout=None) -> Task:
        try:
            task = self.finished_queue.get(timeout=timeout)
            self.finished_queue.task_done()
            log(f"Task dequeued: {task}")
            return result
        except queue.Empty:
            return None


# Example usage:
if __name__ == "__main__":
    import time

    # Sample task that sleeps for 1 second and returns a message
    class SimpleTask:
        def run(self):
            time.sleep(1)
            print("BEEP")

        def complete(self):
            print("Borp")

    # Create a background task runner instance
    runner = BackgroundTaskRunner()
    # Enqueue a few tasks
    for _ in range(3):
        runner.enqueue(SimpleTask())

    # Continuously check for finished tasks
    finished = 0
    while finished < 3:
        result = runner.dequeue(timeout=0.2)
        if result is not None:
            print("Got result:", result.complete())
            finished += 1
