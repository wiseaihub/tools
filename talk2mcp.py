
# talk2mcp.py
# MCP client that drives the Paint MCP server using an LLM (Gemini) to choose tools.
# Plan (short, fast):
#   1) strings_to_chars_to_int('INDIA')
#   2) int_list_to_exponential_sum([...])
#   3) open_paint
#   4) draw_rectangle|400|400|900|600  (canvas-relative)
#   5) add_text_in_paint|Result: <value>
#   6) FINAL_ANSWER
#
# Env:
#   GEMINI_API_KEY  (required)
#   MODEL           (default: gemini-2.5-flash)

import os
import re
import json
import asyncio
from typing import Any, Dict, Tuple

from mcp.client.stdio import stdio_client, StdioServerParameters
import sys, os
from pathlib import Path


# --- LLM (Gemini) ---
from google import genai

# --- MCP client ---
from mcp.client.session import ClientSession

# --- helpers for superscript scientific formatting ---
SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

def format_scientific_superscript(s: str) -> str:
    """
    Accepts a numeric string (with or without e-notation), returns 'mantissa × 10^{superscript exponent}'.
    Example: '7.59982224609308e+33' -> '7.59982224609308 × 10³³'
    """
    import re
    # Try to grab a number from the string (handles '... "7.59e+33" ...')
    m = re.search(r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)", s)
    if not m:
        return s  # give back whatever we got

    # Normalize via float -> canonical scientific -> split mantissa/exponent
    try:
        val = float(m.group(1))
    except Exception:
        return m.group(1)

    mantissa, exp_str = f"{val:.15e}".split("e")   # e.g. '7.599822246093080e+33'
    mantissa = mantissa.rstrip("0").rstrip(".")    # trim trailing zeros/dot
    exp_int = int(exp_str)                         # safe: keeps '33' not '3'
    exp_sup = f"{exp_int:+d}".translate(SUPERSCRIPT)
    return f"{mantissa} × 10{exp_sup}"


API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment.")

client = genai.Client(api_key=API_KEY)

AGENT_SYSTEM_PROMPT = (
    "You are an MCP-aware agent. You MUST solve the task by calling tools exposed "
    "by the MCP server. NEVER simulate tool results. Each step, output exactly ONE line:\n"
    "  FUNCTION_CALL: <tool_name>|<arg1>|<arg2>|...  (no JSON)\n"
    "or to finish: FINAL_ANSWER: <short message>.\n"
    "Rules:\n"
    "- Available tools: strings_to_chars_to_int(string), int_list_to_exponential_sum(int_list), "
    "open_paint(), draw_rectangle(x1,y1,x2,y2), add_text_in_paint(text[,x,y]).\n"
    "- Always call tools to accomplish real-world actions.\n"
    "- Draw the rectangle from (400,400) to (900,600) (canvas-relative), then write the result inside it.\n"
    "- Arguments are pipe-separated scalars; lists are comma-separated numbers with no spaces, e.g. 1,2,3.\n"
)

INITIAL_TASK = (
    "Compute the ASCII codes of the word INDIA, then compute the sum of e^n for each code.\n"
    "Open Microsoft Paint, draw a rectangle from (400,400) to (900,600) in canvas-relative coordinates, "
    "and write the final numeric result inside the rectangle. Then say FINAL_ANSWER: Done."
)

FUNCTION_CALL_RE = re.compile(r"^\s*(FUNCTION_CALL|FINAL_ANSWER)\s*:\s*(.+)$", re.IGNORECASE)

def parse_function_call(text: str) -> Tuple[str, str, list]:
    m = FUNCTION_CALL_RE.match(text.strip())
    if not m:
        return ("", "", [])
    kind = m.group(1).upper()
    payload = m.group(2).strip()
    if kind == "FINAL_ANSWER":
        return ("FINAL_ANSWER", payload, [])
    # tool call: name|arg1|arg2|...
    parts = [p.strip() for p in payload.split("|")]
    name = parts[0]
    args = parts[1:]
    return ("FUNCTION_CALL", name, args)

def _extract_text(resp) -> str:
    # Works across google-genai versions
    try:
        return resp.output_text.strip()
    except Exception:
        pass
    try:
        parts = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            if not content:
                continue
            for p in getattr(content, "parts", []) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass
    return str(resp).strip()

async def llm_next_line(history: list) -> str:
    # Flatten system + chat into one prompt string for compatibility
    convo_lines = [f"{m['role'].upper()}: {m['content']}" for m in history]
    full_prompt = (
        AGENT_SYSTEM_PROMPT
        + "\n\n"
        + "\n".join(convo_lines)
        + "\n\nYour next line (either FUNCTION_CALL: ... or FINAL_ANSWER: ...):"
    )
    resp = client.models.generate_content(model=MODEL, contents=full_prompt)
    return _extract_text(resp)


def to_int_list(csv: str):
    return [int(x) for x in csv.split(",") if x.strip()]

INTER_ITER_SLEEP = 0.2
MAX_STEPS = 8

async def main():
    import sys
    # Build an absolute path to the server file (so it works no matter where you run from)
    server_path = str(Path(__file__).with_name("mcp_paint_server.py"))

    server_params = StdioServerParameters(
        command=sys.executable,            # <-- must be a string: the Python exe
        args=[server_path],                # <-- pass the script path as an arg
        env={}                             # (optional) extra env for the server
    )

    history = [{"role": "user", "content": INITIAL_TASK}]
    math_value = None
    did_draw = False
    did_write = False

    # Count the number of steps taken so far

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            if hasattr(tools_result, "tools"):
                tools_list = tools_result.tools
            elif isinstance(tools_result, (list, tuple)):
                tools_list = list(tools_result)
            else:
                raise RuntimeError(f"Unexpected list_tools() shape: {type(tools_result)}")
            tool_names = {t.name for t in tools_list}

            required = {"strings_to_chars_to_int", "int_list_to_exponential_sum",
                        "open_paint", "draw_rectangle", "add_text_in_paint"}
            if not required.issubset(tool_names):
                raise RuntimeError("Server tools missing. Ensure mcp_paint_server.py exposes required tools.")

            # Progress flags MUST live inside the session block
            did_draw = False
            did_write = False

            print("\n=== Agent Execution Start ===")
            for step in range(1, MAX_STEPS + 1):
                line = await llm_next_line(history)
                print(f"[STEP {step}] LLM -> {line}")
                kind, name, args = parse_function_call(line)

                # Stop if already finished
                if did_draw and did_write:
                    print("[DONE] Already drew and wrote text. Finishing.")
                    break

                if kind == "FINAL_ANSWER":
                    print(f"[STEP {step}] DONE: {name}")
                    break

                if kind != "FUNCTION_CALL":
                    history.append({"role": "assistant", "content": "Please output a valid FUNCTION_CALL."})
                    await asyncio.sleep(INTER_ITER_SLEEP)
                    continue

                # Skip duplicate draws
                if name == "draw_rectangle" and did_draw:
                    print(f"[STEP {step}] (skipping duplicate draw request; rectangle already drawn)")
                    history.append({"role": "assistant",
                                    "content": "Rectangle already drawn. Proceed to add_text_in_paint and then FINAL_ANSWER."})
                    await asyncio.sleep(INTER_ITER_SLEEP)
                    continue

                # Deterministic constraints
                if name == "draw_rectangle":
                    args = ["300", "300", "1300", "650"]
                    print(f"[STEP {step}] (forcing rectangle to 300,300→1300,650)")

                if name == "add_text_in_paint" and math_value is not None:
                    header = "Sum of exponentials of ASCII values of INDIA"
                    args = [f"{header}\n{math_value}"]
                    print(f"[STEP {step}] (forcing add_text_in_paint to two-line header + result)")

                # Prepare arguments
                call_args = {}
                if name == "strings_to_chars_to_int":
                    call_args = {"string": args[0] if args else "INDIA"}
                elif name == "int_list_to_exponential_sum":
                    csv = args[0] if args else ""
                    csv = csv.strip().strip("[]").replace(" ", "")
                    call_args = {"int_list": to_int_list(csv)}
                elif name == "open_paint":
                    call_args = {}
                elif name == "draw_rectangle":
                    x1, y1, x2, y2 = map(int, args[:4])
                    call_args = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                elif name == "add_text_in_paint":
                    call_args = {"text": args[0] if args else "Result: <value>"}
                else:
                    print(f"[STEP {step}] Unknown tool: {name}")
                    history.append({"role": "assistant", "content": "Unknown tool; please call a known tool."})
                    await asyncio.sleep(INTER_ITER_SLEEP)
                    continue

                # Execute tool BEFORE setting flags/breaking
                try:
                    result = await session.call_tool(name, arguments=call_args)
                except Exception as e:
                    print(f"[STEP {step}] TOOL {name} ERROR: {e}")
                    history.append({"role": "assistant", "content": f"TOOL_ERROR: {e}"})
                    await asyncio.sleep(INTER_ITER_SLEEP)
                    continue

                # Parse tool output
                text_bits = []
                try:
                    for c in result.content:
                        if getattr(c, "type", "") == "text":
                            text_bits.append(c.text)
                except Exception:
                    pass
                tool_text = "\n".join(text_bits) if text_bits else json.dumps(result.model_dump(), default=str)
                print(f"[STEP {step}] TOOL {name} -> {tool_text}")

                # Capture math in superscript scientific form
                if name == "int_list_to_exponential_sum":
                    math_value = format_scientific_superscript(tool_text)

                # Update flags and possibly exit
                if name == "draw_rectangle":
                    did_draw = True
                if name == "add_text_in_paint":
                    did_write = True
                    print(f"[STEP {step}] Completed writing text; exiting run loop.")
                    break

                # Feed tool result back to LLM
                history.append({"role": "assistant", "content": f"TOOL {name} -> {tool_text}"})
                await asyncio.sleep(INTER_ITER_SLEEP)

            print("\n=== Agent Execution Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
