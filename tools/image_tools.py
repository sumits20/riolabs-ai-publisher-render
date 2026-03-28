# tools/image_tools.py

import os
import base64
from openai import OpenAI
from langchain_core.tools import tool

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_featured_image_prompt(title: str, topic: str, article_excerpt: str = "") -> str:
    """
    Create a clean featured-image prompt for a science/history/news article.
    """
    return f"""
Create a high-quality featured image for a blog article.

Article title:
{title}

Article topic:
{topic}

Article excerpt:
{article_excerpt}

Style requirements:
- modern editorial illustration or realistic digital art
- visually striking
- suitable for a science, discovery, history, or educational news blog
- appealing to Indian teenagers
- bright, engaging, clean composition
- no text, no captions, no labels, no watermark
- landscape orientation
- suitable as a WordPress featured image
""".strip()


def generate_featured_image_bytes(
    title: str,
    topic: str,
    article_excerpt: str = "",
    size: str = "1536x1024"
) -> dict:
    """
    Generate an image using OpenAI and return bytes plus metadata.
    """
    prompt = build_featured_image_prompt(
        title=title,
        topic=topic,
        article_excerpt=article_excerpt
    )

    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size
    )

    image_b64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_b64)

    return {
        "prompt": prompt,
        "image_bytes": image_bytes,
        "mime_type": "image/png",
        "filename": "featured_image.png"
    }


@tool
def make_image_prompt_tool(title: str, topic: str, article_excerpt: str = "") -> str:
    """
    Create a featured-image prompt for a blog article.
    Use this when you want to review or inspect the image prompt before generating the image.
    """
    return build_featured_image_prompt(
        title=title,
        topic=topic,
        article_excerpt=article_excerpt
    )
