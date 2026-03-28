import json
import os
from openai import OpenAI
from langchain_core.tools import tool

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@tool
def choose_best_topic_tool(user_topic: str, research_text: str, recent_posts_text: str) -> str:
    """
    Choose the best final article angle based on user topic, web research, and recent WordPress posts.
    Use this to avoid duplicate topics and select an attractive topic for Indian teenagers.
    Returns JSON as text with final_title, final_topic, why_selected, teen_style_notes, and excerpt.
    """
    prompt = f"""
You are a content strategist for a blog aimed at Indian teenagers.

User topic:
{user_topic}

Web research:
{research_text}

Recent WordPress posts:
{recent_posts_text}

Task:
- choose the best article angle
- avoid repeating existing recent blog topics
- make it engaging for Indian teens
- keep it factual and easy to understand

Return STRICT JSON only in this format:
{{
  "final_title": "A strong title",
  "final_topic": "Short chosen angle",
  "why_selected": "Reason",
  "teen_style_notes": "Style guidance",
  "excerpt": "Short excerpt"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content.strip()


@tool
def write_article_tool(final_title: str, final_topic: str, teen_style_notes: str, research_text: str) -> str:
    """
    Write a WordPress-ready article in simple English for Indian teenagers.
    Use this after the final topic has been selected.
    Returns clean HTML suitable for WordPress.
    """
    prompt = f"""
Write a blog article using ONLY the research context below.

Title:
{final_title}

Chosen topic:
{final_topic}

Audience:
Indian teenagers

Style guidance:
{teen_style_notes}

Rules:
- use simple English
- easy to read
- short paragraphs
- engaging introduction
- clear subheadings
- factual only
- do not invent facts not supported by the research
- output clean HTML for WordPress
- use tags like <h2>, <p>, <ul>, <li>
- do not include markdown fences
- do not include <html> or <body>

Research context:
{research_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You write clean WordPress-ready HTML articles."},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content.strip()
