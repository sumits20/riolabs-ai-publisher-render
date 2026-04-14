import json
import os
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

try:
    import streamlit as st
except Exception:
    st = None


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    if st is not None:
        try:
            return st.secrets[key]
        except Exception:
            pass
    return os.getenv(key, default)


def try_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)


def generate_image_prompt(
    openai_client: OpenAI,
    final_title: str,
    final_topic: str,
    excerpt: str,
    research_text: str,
) -> Dict[str, str]:
    """
    Ask GPT to produce a strong editorial image prompt for the chosen article.
    """
    prompt = f"""
You are creating a blog header image prompt for a science, discovery, history, or news article.

Title:
{final_title}

Chosen topic:
{final_topic}

Excerpt:
{excerpt}

Research context:
{research_text}

Requirements:
- Create a realistic, high-quality editorial blog image prompt
- Suitable for Indian teenagers, but not childish
- No text in the image
- No watermark
- No logos
- No UI elements
- Prefer one strong clear visual concept
- Keep it factual in mood
- Good as a website featured image / hero image
- Return STRICT JSON only

Return this exact JSON shape:
{{
  "image_prompt": "Detailed prompt for image generation",
  "alt_text": "Short accessible alt text",
  "caption": "Short caption for WordPress",
  "style_notes": "Short style note"
}}
""".strip()

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.6,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. No markdown fences. No explanation outside JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    raw = response.choices[0].message.content or ""

    try:
        return try_parse_json(raw)
    except Exception:
        return {
            "image_prompt": (
                f"Editorial science blog header image for '{final_title}', "
                f"showing {final_topic}, realistic lighting, clean composition, "
                "high detail, no text, no watermark, no logo."
            ),
            "alt_text": final_title,
            "caption": final_title,
            "style_notes": "Realistic editorial illustration"
        }


def generate_grok_image(
    prompt: str,
    aspect_ratio: str = "16:9",
    timeout: int = 180,
) -> Dict[str, Any]:
    """
    Generate an image using xAI's image model through the OpenAI-compatible SDK.
    Returns:
      {
        "image_url": str,
        "image_bytes": bytes,
        "content_type": str,
        "filename": str
      }
    """
    xai_api_key = get_secret("XAI_API_KEY")
    if not xai_api_key:
        raise ValueError("Missing XAI_API_KEY")

    client = OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=xai_api_key,
    )

    response = client.images.generate(
        model="grok-imagine-image",
        prompt=prompt,
        size=aspect_ratio,
    )

    if not response.data or not response.data[0]:
        raise ValueError("No image returned from xAI")

    image_url = getattr(response.data[0], "url", None)
    b64_json = getattr(response.data[0], "b64_json", None)

    if image_url:
        img_resp = requests.get(image_url, timeout=timeout)
        img_resp.raise_for_status()

        content_type = img_resp.headers.get("Content-Type", "image/png")
        ext = ".png"
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "webp" in content_type:
            ext = ".webp"

        return {
            "image_url": image_url,
            "image_bytes": img_resp.content,
            "content_type": content_type,
            "filename": f"generated_featured_image{ext}",
        }

    if b64_json:
        import base64

        image_bytes = base64.b64decode(b64_json)
        return {
            "image_url": None,
            "image_bytes": image_bytes,
            "content_type": "image/png",
            "filename": "generated_featured_image.png",
        }

    raise ValueError("xAI response did not include a URL or base64 image payload")
