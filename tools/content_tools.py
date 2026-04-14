import os
import json
from openai import OpenAI
from anthropic import Anthropic
from langchain_core.tools import tool

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None


def _safe_json_loads(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)


def _build_article_prompt(
    final_title: str,
    final_topic: str,
    teen_style_notes: str,
    research_text: str,
) -> str:
    return f"""
Write a factual blog article in clean WordPress HTML.

Title:
{final_title}

Chosen angle:
{final_topic}

Audience:
Indian teenagers

Style notes:
{teen_style_notes}

Rules:
- Write in a neutral article style
- Do NOT address the reader directly as "you"
- Do NOT use phrases like "Imagine", "Have you ever", "Let's", or "Hey"
- Use facts supported by the research context
- Output clean HTML only
- Use tags like <h2>, <p>, <ul>, <li> where useful
- Do not include markdown or code fences
- Do not include <html> or <body> tags

Research context:
{research_text}
""".strip()


@tool
def choose_best_topic_tool(user_topic: str, research_text: str, recent_posts_text: str) -> str:
    """
    Choose the best final article angle based on the user topic, web research,
    and recent WordPress posts. Returns JSON text with:
    final_title, final_topic, why_selected, teen_style_notes, excerpt.
    """
    prompt = f"""
You are a content strategist for a science, discovery, history, and news blog aimed at Indian teenagers.

User topic:
{user_topic}

Web research:
{research_text}

Recent WordPress posts:
{recent_posts_text}

Task:
- Choose the best article angle
- Avoid repeating existing recent blog topics
- Keep it factual and easy to understand
- Borad topic keeping fact intact
- Return STRICT JSON only

Return this exact JSON shape:
{{
  "final_title": "A strong title",
  "final_topic": "Short chosen angle",
  "why_selected": "Reason this angle was selected",
  "teen_style_notes": "Style guidance",
  "excerpt": "Short excerpt"
}}
""".strip()

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. No markdown fences. No explanation outside JSON."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    content = (response.choices[0].message.content or "").strip()

    try:
        parsed = _safe_json_loads(content)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        fallback = {
            "final_title": user_topic.strip().title(),
            "final_topic": user_topic.strip(),
            "why_selected": "Fallback used because structured JSON parsing failed.",
            "teen_style_notes": "Use simple, factual, engaging language for Indian teenagers.",
            "excerpt": f"A simple and engaging article about {user_topic.strip()}."
        }
        return json.dumps(fallback, ensure_ascii=False)


@tool
def write_article_gpt_tool(
    final_title: str,
    final_topic: str,
    teen_style_notes: str,
    research_text: str,
) -> str:
    """
    Generate the final article with GPT and return clean WordPress-ready HTML.
    """
    prompt = _build_article_prompt(
        final_title=final_title,
        final_topic=final_topic,
        teen_style_notes=teen_style_notes,
        research_text=research_text,
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": "You write clean WordPress-ready HTML articles."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    return (response.choices[0].message.content or "").strip()


@tool
def write_article_claude_tool(
    final_title: str,
    final_topic: str,
    teen_style_notes: str,
    research_text: str,
) -> str:
    """
    Generate the final article with Claude and return clean WordPress-ready HTML.
    """
    if not anthropic_client:
        return "<p>Claude generation is unavailable because ANTHROPIC_API_KEY is missing.</p>"

    prompt = _build_article_prompt(
        final_title=final_title,
        final_topic=final_topic,
        teen_style_notes=teen_style_notes,
        research_text=research_text,
    )

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2500,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()
