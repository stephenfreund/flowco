import sys
import threading
import time
import traceback


def print_stack_for_thread(thread_id):
    """
    Print the stack trace for the thread with the given thread_id every second.
    """
    while True:
        time.sleep(5)  # Wait for 5 second before printing the stack trace

        for thread in threading.enumerate():
            if thread.ident == thread_id:
                print(f"Stack trace for thread {thread.name} (ID: {thread_id}):")
                stack = sys._current_frames()[thread_id]
                traceback.print_stack(stack, limit=3)
                print("\n")
                break
        else:
            print(f"No thread found with ID {thread_id}.")
            break


def watch_thread(thread: threading.Thread):
    print_stack_thread = threading.Thread(
        target=print_stack_for_thread, args=(thread.ident,)
    )
    print_stack_thread.start()
