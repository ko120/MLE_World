import json
import re

import black


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text, fallback_to_invalid=False):
    """Extract python code blocks from the text.

    Returns only code that compiles. If nothing compiles and
    `fallback_to_invalid` is set, returns the best-effort code region anyway
    (fences stripped, surrounding NL prose excluded) — so a downstream executor
    surfaces the code's REAL SyntaxError instead of choking on a leaked
    ```python fence. Returns "" if no code region can be isolated.
    """
    if not text:
        return ""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    if not valid_code_blocks and fallback_to_invalid:
        # Nothing compiled: hand back the isolated (broken) code region so the
        # executor's traceback points at the model's actual bug, not the fence.
        return "\n\n".join(c for c in parsed_codes if c.strip())
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if not s or "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code
