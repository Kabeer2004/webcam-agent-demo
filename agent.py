"""
Webcam Agent Demo — CLI Chat Interface

Run:  python agent.py

Requires:
  1. Ollama running locally with qwen3:4b
  2. Face model trained (python train_model.py)
"""

import config

# Import tools so they get registered with @register_tool before creating the agent
import tools  # noqa: F401

from qwen_agent.agents import Assistant


def build_system_prompt() -> str:
    """Build the system prompt with person facts baked in."""
    facts_block = ""
    for label_id, name in config.PERSON_NAMES.items():
        facts = config.PERSON_FACTS.get(label_id, [])
        facts_str = "\n".join(f"  - {f}" for f in facts)
        facts_block += f"\n{name}:\n{facts_str}\n"

    return f"""You are a friendly AI assistant with access to a webcam and face recognition tools.

BEHAVIOUR RULES:
1. When a user sends their FIRST message (like "hi", "hello", "hey", or any greeting),
   you MUST use the `take_photo` tool to capture a photo from the webcam.
2. After taking the photo, IMMEDIATELY use the `recognize_face` tool with the image path
   returned by `take_photo` to identify who is in front of the camera.
3. Based on the recognition result, greet the person warmly and personally using the facts
   you know about them (listed below).
4. If the person is not recognised, greet them as a guest and be friendly.
5. For ALL subsequent messages in the conversation, do NOT use the webcam or face
   recognition tools again — just chat normally.
6. You are helpful, witty, and concise.

KNOWN PERSONS AND FACTS:
{facts_block}
Use these facts creatively to make the greeting feel personal and fun.
"""


import re
import sys


# ANSI color codes for terminal styling
class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def extract_thinking(text: str) -> tuple[str, str]:
    """
    Separate Qwen3's <think>...</think> reasoning from the visible reply.
    Returns (thinking_text, reply_text).
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    reply = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return thinking, reply


def main():
    print(f"{Colors.BOLD}{'=' * 60}")
    print("  Webcam Agent Demo  —  Powered by Qwen Agent + Ollama")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f"  Model : {config.LLM_MODEL}")
    print(f"  Server: {config.LLM_SERVER}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print("Type 'quit' or 'exit' to end the session.\n")

    llm_cfg = {
        "model": config.LLM_MODEL,
        "model_server": config.LLM_SERVER,
        "api_key": config.LLM_API_KEY,
    }

    bot = Assistant(
        llm=llm_cfg,
        name="Webcam Agent",
        description="An AI assistant that can see you through the webcam.",
        system_message=build_system_prompt(),
        function_list=["take_photo", "recognize_face"],
    )

    messages = []

    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}You:{Colors.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        full_response = []
        seen_tools = set()
        last_text_len = 0
        shown_thinking = set()
        
        for response in bot.run(messages):
            full_response = response
            
            # ── 1. Show tool results ────────────────────────────────
            for msg in response:
                if msg.get("role") == "tool":
                    tool_id = msg.get("name") or "unknown_tool"
                    content = msg.get("content", "")
                    call_sig = f"result_{tool_id}_{content}"
                    if call_sig not in seen_tools:
                        print(f"\n{Colors.GREEN}  ↳ Tool Result ({tool_id}):{Colors.RESET} {content}", flush=True)
                        seen_tools.add(call_sig)
            
            # ── 2. Find latest assistant message ────────────────────
            assistant_msg = None
            for msg in reversed(response):
                if msg.get("role") == "assistant":
                    assistant_msg = msg
                    break
            
            if not assistant_msg:
                continue
            
            # ── 3. Show tool call initiation ────────────────────────
            if "function_call" in assistant_msg and assistant_msg["function_call"]:
                call = assistant_msg["function_call"]
                call_name = call.get("name")
                call_args = call.get("arguments")
                call_sig = f"call_{call_name}_{call_args}"
                
                # Before the tool call, the model might have thinking in the content
                raw_content = assistant_msg.get("content", "")
                if isinstance(raw_content, str) and raw_content.strip():
                    thinking, _ = extract_thinking(raw_content)
                    if thinking and thinking not in shown_thinking:
                        print(f"\n{Colors.DIM}{Colors.CYAN}  💭 Thinking: {thinking}{Colors.RESET}", flush=True)
                        shown_thinking.add(thinking)
                
                if call_sig not in seen_tools:
                    print(f"\n{Colors.YELLOW}  🔧 Calling tool `{call_name}`{Colors.RESET}", end="", flush=True)
                    if call_args:
                        print(f" {Colors.DIM}args: {call_args}{Colors.RESET}", end="", flush=True)
                    print(flush=True)
                    seen_tools.add(call_sig)
                    last_text_len = -1
                continue

            # ── 4. Stream text content ──────────────────────────────
            raw_content = assistant_msg.get("content", "")
            if isinstance(raw_content, str):
                text = raw_content
            elif isinstance(raw_content, list):
                text = "".join([p.get("text", "") for p in raw_content if isinstance(p, dict)])
            else:
                text = ""

            if not text:
                continue

            # Parse out thinking vs reply
            thinking, reply = extract_thinking(text)
            
            # Show thinking if we haven't already
            if thinking and thinking not in shown_thinking:
                print(f"\n{Colors.DIM}{Colors.CYAN}  💭 Thinking: {thinking}{Colors.RESET}", flush=True)
                shown_thinking.add(thinking)
            
            # Stream the visible reply
            if reply:
                if last_text_len <= 0:
                    print(f"\n{Colors.BOLD}{Colors.MAGENTA}Assistant:{Colors.RESET} ", end="", flush=True)
                    last_text_len = 0
                
                if len(reply) > last_text_len:
                    new_text = reply[last_text_len:]
                    print(new_text, end="", flush=True)
                    last_text_len = len(reply)

        print()  # Newline after response
        if full_response:
            messages.extend(full_response)
        else:
            print("[No response from agent]")


if __name__ == "__main__":
    main()
