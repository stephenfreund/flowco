from typing import Optional

from flowco.util.text import (
    format_key_lines,
    format_key_value,
    strip_ansi,
)


def error_message(message: str, e: Exception) -> str:
    return "\n".join(
        [
            message,
            format_key_lines("Details:", strip_ansi(str(e)).split("\n")),
        ]
    )


def execution_error_message(
    message: str, e: Exception, node: Optional["Node"] = None
) -> str:
    if node:
        source = node.code
        return "\n".join(
            [
                message,
                format_key_lines(
                    "Code:", ["Here is the node:", "------------------"] + source
                ),
                format_key_lines("Failure:", strip_ansi(str(e)).split("\n")),
            ]
        )
    else:
        return "\n".join(
            [
                message,
                format_key_lines("Failure:", strip_ansi(str(e)).split("\n")),
            ]
        )
