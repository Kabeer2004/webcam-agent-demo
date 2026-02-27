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


def main():
    print("=" * 60)
    print("  Webcam Agent Demo  —  Powered by Qwen Agent + Ollama")
    print("=" * 60)
    print(f"  Model : {config.LLM_MODEL}")
    print(f"  Server: {config.LLM_SERVER}")
    print("=" * 60)
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
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("\nAgent: ", end="", flush=True)

        response = []
        for response in bot.run(messages):
            pass  # consume the generator to get the final response

        # Extract the assistant's final text reply
        if response:
            # The last message with role 'assistant' contains the final answer
            for msg in reversed(response):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        print(content)
                        break
                    elif isinstance(content, list):
                        # content can be a list of dicts like [{"text": "..."}]
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                print(part["text"])
                        break

            messages.extend(response)
        else:
            print("[No response from agent]")


if __name__ == "__main__":
    main()
