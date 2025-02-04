import json
import re
from typing import Any, Dict, List


def process_chat_message(message) -> Dict[str, Any]:
    """
    Processes a ChatCompletionMessageParam (a dictionary) by searching through its string values
    for blocks delimited by triple backticks. Each string is split into an array of segments containing
    either raw text or parsed JSON objects (if the content within the backticks is valid JSON).
    """

    def parse_json_strings(obj: Any) -> Any:
        if isinstance(obj, dict):
            map = {key: parse_json_strings(value) for key, value in obj.items()}
            if "parsed" in map:
                del map["content"]
            return map
        elif isinstance(obj, list):
            return [parse_json_strings(item) for item in obj]
        elif isinstance(obj, str):
            if obj.startswith("data:image/png;base64,"):
                return "data:image/png;base64,..."  # Omit the base64 image data for brevity.
            if obj.startswith("{") and obj.endswith("}"):
                # If the string is a JSON object, parse it.
                try:
                    return parse_json_strings(json.loads(obj))
                except json.JSONDecodeError:
                    return obj

            segments: List[Any] = []
            # Regex pattern for blocks delimited by triple backticks.
            # Note: We assume the backticks are immediately followed by a newline,
            # then the content, then a newline and triple backticks.
            pattern = r"(```\n)(.*?)(\n```)"
            last_end = 0
            for match in re.finditer(pattern, obj, flags=re.DOTALL):
                # Append any text before the matched block as a raw string.
                segments.extend(obj[last_end : match.start()].splitlines())
                json_content = match.group(2)
                try:
                    # Attempt to parse the content within the backticks as JSON.
                    parsed = parse_json_strings(json.loads(json_content))
                    segments.append(parsed)
                except json.JSONDecodeError:
                    # If parsing fails, append the raw content.
                    segments.extend(json_content.splitlines())
                last_end = match.end()
            # Append any text remaining after the last match.
            segments.extend(obj[last_end:].splitlines())

            # If the string did not contain any backtick-delimited blocks,
            # return the string wrapped in a list for consistency.
            if len(segments) == 1:
                return segments[0]
            return segments
        else:
            return obj

    return {key: parse_json_strings(value) for key, value in message.items()}


# Example usage
message = {
    "role": "user",
    "content": 'Here is some text and a JSON block:\n```\n{"key": "value"}\n``` and some more text.',
    "extra": {
        "nested_content": 'Another block:\n```\n{"nested_key": "nested_value"}\n``` and plain text.',
        "more_text": "no json here",
    },
}

processed_message = process_chat_message(message)
formatted_message = json.dumps(processed_message, indent=2)
print(formatted_message)
