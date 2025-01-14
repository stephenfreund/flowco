from io import StringIO
import re
import sys
import textwrap
import time
import traceback

from httpx import get
import termcolor
import threading

from flowco.util.config import config
import threading

from flowco.util.text import strip_ansi
from flowco.session.session import session


class LoggingFile:
    def __init__(self, file, log_file_path, prefix=""):
        """
        Wraps a file-like object and logs all writes to a specified log file.
        Prefixes each line of output with a given prefix.

        :param file: A file-like object to wrap (e.g., sys.stdout or any file object).
        :param log_file_path: Path to the log file where output will be mirrored.
        :param prefix: A string to prepend to each line of output.
        """
        self.file = file
        self.log_file = open(log_file_path, "a")
        self.prefix = prefix
        self.buffer = ""

    def write(self, message):
        """
        Buffers partial lines and writes full lines with the prefix to the wrapped file and log file.
        """
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._write_line(line + "\n")

    def flush(self):
        """
        Flushes both the wrapped file and the log file.
        If there's any remaining text in the buffer, it will be written.
        """
        if self.buffer:
            self._write_line(self.buffer)
            self.buffer = ""
        self.file.flush()
        self.log_file.flush()

    def close(self):
        """
        Flushes any remaining buffer and closes the log file (does not close the wrapped file).
        """
        self.flush()
        self.log_file.close()

    def isatty(self):
        """
        Checks if the wrapped file is a terminal.
        """
        return hasattr(self.file, "isatty") and self.file.isatty()

    def _write_line(self, line):
        """
        Writes a single line with the prefix to both the wrapped file and the log file.
        """
        line_with_prefix = self.prefix + line
        self.file.write(line_with_prefix)
        self.log_file.write(line_with_prefix)

    def __getattr__(self, name):
        """
        Delegates attribute access to the wrapped file.
        """
        return getattr(self.file, name)

    def __setattr__(self, name, value):
        """
        Sets attributes, handling internal attributes separately.
        """
        if name in {"file", "log_file", "prefix", "buffer"}:
            super().__setattr__(name, value)
        else:
            setattr(self.file, name, value)

    def __del__(self):
        """
        Ensures the log file is closed when the instance is deleted.
        """
        self.close()


class Output(threading.local):

    def __init__(
        self, max_log_depth=10, file=LoggingFile(sys.stdout, "logging.txt"), prefix=None
    ):
        self.max_log_depth = max_log_depth
        self.pending = None
        self.contexts = []
        self.file = file
        self.lock = threading.RLock()
        self.prefix = prefix

    def set_file(self, file):
        self.file = file

    ###

    def timing_context(self, key):
        if config.quiet:
            return self.NoOpContextManager()
        else:
            return self.OutputContext(
                self, key, color="cyan", start="[", time_prefix="", end="]"
            )

    def buffer(self, prefix):
        return self.BufferContext(self, prefix)

    class BufferContext:
        def __init__(self, outer, prefix):
            self.outer = outer
            self.file = StringIO()
            self.prefix = prefix

        def __enter__(self):
            with self.outer.lock:
                self.old_file = self.outer.file
                self.outer.file = self.file
                return self

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                self.outer.file = self.old_file
                self.text = textwrap.indent(
                    self.file.getvalue(),
                    termcolor.colored(f"[{self.prefix}]", "dark_grey"),
                )
                return False

        def get_text(self):
            return self.text

    ##
    class NoOpContextManager:
        def __enter__(self):
            # No setup code needed
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            # No teardown code needed
            return False  # Do not suppress exceptions

    class OutputContext:

        def __init__(
            self, outer, message, color="cyan", start="", time_prefix="", end=""
        ):
            self.outer = outer
            self.message = termcolor.colored(message + "...", color)
            self.color = color
            self.start = termcolor.colored(start, color)
            self.time_prefix = termcolor.colored(time_prefix, color)
            self.end = termcolor.colored(end, color)

        def __enter__(self):
            with self.outer.lock:
                self.outer.print_enter(self.message, start=self.start, end=self.end)
                self.outer.contexts += [self]
                self.start_time = time.time()

        def __exit__(self, exc_type, exc_value, traceback):
            with self.outer.lock:
                end_time = time.time()
                duration = int((end_time - self.start_time) * 1000)
                self.outer.contexts.pop()
                message = f"{self.time_prefix}{termcolor.colored(f'{duration} ms', self.color)}"
                self.outer.print_exit(message, start=self.start, end=self.end)

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        # Output.output_log.write(strip_ansi(message))
        # Output.output_log.flush()

    def flush(self):
        if self.pending is not None:
            self.file.write(self.pending)
            self.file.write("\n")
            # Output.output_log.write(strip_ansi(self.pending))
            # Output.output_log.flush()
            self.pending = None

    def get_prefixed_pad(self):
        """Return the prefix followed by spaces based on the current pad depth."""
        if self.prefix is None:
            return " " * self.pad_depth()
        else:
            return f"[{self.prefix}] {' ' * self.pad_depth()}"

    def pad_depth(self):
        return len(self.contexts) * 2

    def print_enter(self, message, start="", end=""):
        self.flush()
        pad = self.get_prefixed_pad()
        indented = textwrap.indent(f"{start}{message}", pad)
        self.write(indented)
        self.pending = end

    # assume message is one line
    def print_exit(self, message, start="", end=""):
        pad = self.get_prefixed_pad()
        if self.pending is not None:
            if self.pending != end:
                self.write(f"Pending does not match end: {self.pending} != {end}")
            # assert (
            #     self.pending == end
            # ), f"Pending does not match end: {self.pending} != {end}"
            self.write(" ")
            self.write(message)
            self.write(self.pending)
            self.write("\n")
        else:
            self.write(f"{pad}{start}{message}{end}\n")
        self.pending = None

    def _format_exception(self, e):
        # Extract the traceback from the exception
        tb = e.__traceback__

        # Format the exception type, value, and traceback
        formatted_exception = "\nException:\n"
        formatted_exception += "".join(traceback.format_exception(type(e), e, tb))

        return formatted_exception

    def _print(self, color, args, start="", end=""):

        self.flush()
        start_len = len(start)
        pad = self.get_prefixed_pad()

        message = " ".join(
            self.format_exception(a) if isinstance(a, Exception) else str(a)
            for a in args
        ).rstrip()

        lines = f"{start}{message}{termcolor.colored(end, color)}".split("\n")
        self.write(f"{pad}{termcolor.colored(lines[0], color)}")
        for line in lines[1:]:
            self.write("\n")
            self.write(f"{pad + (' ' * start_len)}{termcolor.colored(line, color)}")
        self.write("\n")

    def log(self, *args):
        with self.lock:
            if not config.quiet:
                self._print("cyan", args, start="[", end="]")

    def debug(self, *args):
        with self.lock:
            if config.debug:
                self._print("cyan", args, start="[", end="]")

    def message(self, *args):
        with self.lock:
            self._print(None, args, start="", end="")

    def error(self, *args):
        with self.lock:
            self._print("red", args, start="[", end="]")

    def warn(self, *args):
        with self.lock:
            self._print("green", args, start="[", end="]")


def log(*message):
    session.get("output", Output).log(*message)


def message(*message):
    session.get("output", Output).message(*message)


def error(*message):
    session.get("output", Output).error(*message)


def warn(*message):
    session.get("output", Output).warn(*message)


def debug(*message):
    session.get("output", Output).debug(*message)


def buffer_output(prefix):
    return session.get("output", Output).buffer(prefix)


def logger(key):
    return session.get("output", Output).timing_context(key)


if __name__ == "__main__":
    # Test the logger
    output = Output()
    with output.timing_context("Main"):
        output.log("This is a log message")

    with output.timing_context("No time"):
        output.log("This is a log message")
        with output.timing_context("Beep"):
            output.log("This is a log message")
            with output.timing_context("Boop"):
                output.message("Real messages")
                output.log("This is a log message")
                output.error("MOo")
            output.log("This is a log message")
        with output.timing_context("Bop"):
            pass
