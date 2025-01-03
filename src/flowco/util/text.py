from collections import OrderedDict
from json import dumps
import difflib
import re
import textwrap
from typing import Callable, List
import deepdiff

import termcolor
from pydantic import BaseModel
import os
import re
import keyword
from pathlib import Path
import markdown

def format_key_value(key, value, width=15, indent=0) -> str:
    value = str(value)
    lines = textwrap.fill(
        value, initial_indent="", subsequent_indent=" " * width, width=75
    ).split("\n")
    first_line = lines[0]
    if len(lines) > 1:
        remaining = "\n" + "\n".join(lines[1:])
    else:
        remaining = ""

    return textwrap.indent(f"{key:<{width}}{first_line}{remaining}", " " * indent)


def format_key_lines(key, lines, width=15, indent=0) -> str:
    if len(lines) == 0:
        return textwrap.indent(f"{key:<{width}}", " " * indent)
    first_line = lines[0]
    if len(lines) > 1:
        remaining = "\n" + textwrap.indent("\n".join(lines[1:]), " " * width)
    else:
        remaining = ""

    return textwrap.indent(f"{key:<{width}}{first_line}{remaining}", " " * indent)


def format_key_line_list(list, width=15, indent=0) -> str:
    return "\n".join(
        [
            format_key_lines(key, lines, width=width, indent=indent)
            for key, lines in list
        ]
    )


def format_basemodel(
    obj: BaseModel,
    order: List[str] = [],
    drop_missing=False,
) -> str:
    fields = obj.model_fields.keys()
    assert set(order).issubset(
        fields
    ), f"Order {order} is not a subset of fields {fields}"

    keys = order
    if not drop_missing:
        remaining = set(fields) - set(order)
        keys += sorted(remaining)

    instance_dict = obj.model_dump()
    ordered = OrderedDict()
    for field in keys:
        ordered[field] = instance_dict[field]

    return dumps(ordered, indent=2)


def color_diff_line(line):
    if line.startswith("+"):
        return termcolor.colored(line, "green")
    elif line.startswith("-"):
        return termcolor.colored(line, "red")
    elif line.startswith("@@"):
        return termcolor.colored(line, "cyan")
    elif line.startswith(" "):
        return line
    else:
        return line  # File headers, etc.


def diff(from_lines, to_lines, from_version="", to_version="", n=3):
    diff = difflib.unified_diff(
        from_lines,
        to_lines,
        lineterm="",
        fromfile=from_version,
        tofile=to_version,
        n=n,
    )
    return "\n".join([color_diff_line(line) for line in diff])


def diff_page_versions(page, from_version, to_version):
    from_page = page.version(from_version)

    if to_version is None:
        to_page = page
    else:
        to_page = page.version(to_version)
    # to_lines = str(to_page).split("\n")

    return diff_pages(from_page, to_page)


def diff_pages(from_page, to_page):
    return deepdiff.DeepDiff(
        from_page.model_dump(exclude={"versions"}),
        to_page.model_dump(exclude={"versions"}),
        ignore_type_in_groups=[
            (str, None),
            (int, None),
            (float, None),
            (bool, None),
            (list, None),
            (dict, None),
            (tuple, None),
            (set, None),
            (type(None), None),
            (type, None),
            (BaseModel, None),
        ],
    )


import difflib
from typing import Callable


def string_diff(
    from_str,
    to_str,
    mark_removed: Callable[[str], str] = lambda x: f"~~{x}~~",
    mark_added: Callable[[str], str] = lambda x: f"**{x}**",
    mark_regular: Callable[[str], str] = lambda x: x,
) -> str:

    # List of list item prefixes
    list_item_prefixes = [r"-", r"*", r"+", r"\d+\.", r"\d+\)"]

    # Combine the prefixes into a regex pattern
    prefix_pattern = r"^\s*(?:(\-|\*|\+|\d+\.|\d+\)))\s*"

    # Function to process each line
    def process_line(line, f):
        # Find the prefix part
        match = re.match(prefix_pattern, line)
        if match:
            prefix = match.group(0)
            content = line[len(prefix) :]
            marked_content = f(content)
            return prefix + marked_content
        else:
            return f(line)

    # Function to apply mark_added to each line
    def mark_added_by_line(text):
        lines = text.split("\n")
        return "\n".join([process_line(line, mark_added) for line in lines])

    def mark_removed_by_line(text):
        lines = text.split("\n")
        return "\n".join([process_line(line, mark_removed) for line in lines])

    def mark_regular_by_line(text):
        lines = text.split("\n")
        return "\n".join([process_line(line, mark_regular) for line in lines])

    common = difflib.SequenceMatcher(None, from_str, to_str).get_matching_blocks()

    from_index = 0
    to_index = 0
    result = []

    for block in common:
        # Add removed text in chunks
        if from_index < block.a:
            result.append(mark_removed_by_line(from_str[from_index : block.a]))

        # Add added text in chunks
        if to_index < block.b:
            result.append(mark_added_by_line(to_str[to_index : block.b]))

        # Add common text
        if block.size > 0:
            result.append(
                mark_regular_by_line(from_str[block.a : block.a + block.size])
            )

        from_index = block.a + block.size
        to_index = block.b + block.size

    # Add any remaining removed text
    if from_index < len(from_str):
        result.append(mark_removed_by_line(from_str[from_index:]))

    # Add any remaining added text
    if to_index < len(to_str):
        result.append(mark_added_by_line(to_str[to_index:]))

    return "".join(result)


def string_lists_diff(
    from_lines: List[str],
    to_lines: List[str],
    mark_removed: Callable[[str], str] = lambda x: f"~~{x}~~",
    mark_added: Callable[[str], str] = lambda x: f"**{x}**",
    mark_regular: Callable[[str], str] = lambda x: x,
) -> List[str]:

    diff = difflib.ndiff(from_lines, to_lines)
    result = []

    for line in diff:
        if line[2:].isspace():
            result.append(line[2:])
        elif line.startswith("- "):
            result.append(mark_removed(line[2:]))
        elif line.startswith("+ "):
            result.append(mark_added(line[2:]))
        elif line.startswith("? "):
            pass
        else:
            result.append(mark_regular(line[2:]))

    return result


def function_name_to_title(function_name):
    """
    Convert a function name to a title by replacing underscores with spaces and capitalizing each word.
    """
    return function_name.replace("_", " ").title()


def strip_ansi(text: str) -> str:
    """
    Strip ANSI escape sequences from text.
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def strip_html(text: str) -> str:
    """
    replace all html tags with empty strings
    """
    return re.sub(r"<[^>]*>", " ", text)


def convert_file_path_to_python_name(file_path: str) -> str:
    """
    Convert a file path into a valid Python variable name.

    Steps:
    1. Extract the base file name without path and extension.
    2. Replace invalid characters with underscores.
    3. Ensure the name does not start with a digit.
    4. Avoid Python reserved keywords by appending an underscore if necessary.
    5. Handle edge cases where the name might be empty after processing.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - str: A valid Python variable name derived from the file name.
    """
    # Step 1: Extract base name without path and extension
    base_name = Path(file_path).stem  # 'example.txt' -> 'example'

    # Debug: Show the extracted base name
    # print(f"Base name: {base_name}")

    # Step 2: Replace invalid characters with underscores
    # Valid characters: letters, digits, and underscores
    # Replace any character that is not a letter, digit, or underscore with '_'
    valid_name = re.sub(r"\W", "_", base_name)

    # Debug: Show the name after replacing invalid characters
    # print(f"After replacing invalid characters: {valid_name}")

    # Step 3: Ensure the name does not start with a digit
    if re.match(r"^\d", valid_name):
        valid_name = f"_{valid_name}"
        # Debug: Show the name after handling leading digits
        # print(f"After handling leading digit: {valid_name}")

    # Step 4: Avoid Python reserved keywords
    if keyword.iskeyword(valid_name):
        valid_name += "_"
        # Debug: Show the name after handling reserved keyword
        # print(f"After handling reserved keyword: {valid_name}")

    # Step 5: Handle edge cases where the name might be empty
    if not valid_name:
        valid_name = "var"  # Default variable name
        # Debug: Show the default name assignment
        # print(f"Assigned default name: {valid_name}")

    return valid_name


def pill_to_python_name(pill: str) -> str:
    # Replace non-alphanumeric characters with underscores
    text = re.sub(r"[^0-9a-zA-Z]", "_", pill)

    # Ensure the variable name does not start with a digit
    if text and text[0].isdigit():
        text = "_" + text

    # Convert to lowercase
    text = text.lower()

    return text


def pill_to_result_var_name(pill: str) -> str:
    return f"{pill_to_python_name(pill)}"


def pill_to_function_name(pill: str) -> str:
    return f"{pill_to_python_name(pill)}"

def md_to_html(md):
    return markdown.markdown(
        md,
        extensions=[
            "extra",  # Includes several extensions like tables, fenced code, etc.
            "codehilite",  # Adds syntax highlighting to code blocks
            "toc",  # Generates a table of contents
            "sane_lists",  # Improves list handling
            "smarty",  # Converts quotes and dashes to smart quotes and dashes
        ],
    )
