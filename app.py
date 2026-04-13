import os
import json
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic

from tools.tavily_research import tavily_search
from tools.wordpress_tools import get_recent_posts, create_draft_post


st.set_page_config(page_title="AI WordPress Publisher", layout="wide")
st.title("AI WordPress Publisher")
st.caption("Research a topic, choose a better angle, generate GPT and Claude versions, then save the selected one to WordPress as a draft.")


# ----------------------------
# Environment checks
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")

missing = []
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")
if not TAVILY_API_KEY:
    missing.append("TAVILY_API_KEY")
if not WP_USER:
    missing.append("WORDPRESS_USERNAME")
if not WP_PASS:
    missing.append("WORDPRESS_APP_PASSWORD")

if missing:
    st.error("Missing environment variables: " + ", ".join(missing))
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ----------------------------
# Helper functions
# ----------------------------
def build_research_text(results: list[dict]) -> str:
    if not results:
        return "No research results found."

    blocks = []
    for idx, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        content = r.get("content", "").strip()

        blocks.append(
            f"Result {idx}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Summary: {content}"
        )

    return "\n\n".join(blocks)


def build_recent_posts_text(posts: list[dict]) -> str:
    if not posts:
        return "No recent posts found on the website."

    return "\n".join(
        f"- {p.get('title', '')} ({p.get('link', '')})"
        for p in posts
    )


def try_parse_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)


def choose_best_topic(user_topic: str, research_text: str, recent_posts_text: str) -> dict:
    prompt = f"""
You are a content strategist for a science, discovery, history, and news blog.

User's starting topic:
{user_topic}

Recent web research:
{research_text}

Recent posts already on the user's WordPress site:
{recent_posts_text}

Your task:
1. Suggest the BEST article angle based on recent web interest.
2. Avoid repeating topics that are too similar to existing WordPress posts.
3. Make the topic attractive for Indian teenagers.
4. Keep it factual, easy to understand, and interesting.
5. Prefer a focused angle rather than a very broad topic.
6. Return STRICT JSON only.

Return JSON with this structure:
{{
  "final_title": "A strong blog title",
  "final_topic": "Short description of chosen angle",
  "why_selected": "Why this is a good angle",
  "teen_style_notes": "How the article should feel for Indian teens",
  "excerpt": "A short 1-2 sentence excerpt for the WordPress draft"
}}
"""

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
            }
        ]
    )

    raw_text = response.choices[0].message.content or ""

    try:
        return try_parse_json(raw_text)
    except Exception:
        return {
            "final_title": user_topic.strip().title(),
            "final_topic": user_topic.strip(),
            "why_selected": "Fallback title used because structured parsing failed.",
            "teen_style_notes": "Use simple, engaging language for Indian teens.",
            "excerpt": f"A simple and engaging article about {user_topic.strip()}."
        }


def build_article_prompt(final_title: str, final_topic: str, teen_style_notes: str, research_text: str) -> str:
    return f"""
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
- Use simple English
- Easy to read for teenagers
- Short paragraphs
- Engaging but neutral article style
- Use clear subheadings
- Keep it factual
- Do not invent facts not supported by the research context
- Do not use heavy jargon
- Include a short conclusion
- Output clean HTML suitable for WordPress
- Use tags like <h2>, <p>, <ul>, <li> where useful
- Do not include <html>, <body>, or markdown code fences
- Do not include fake citations like [1] or source lists at the end
- Do not directly address the reader as "you"

Research context:
{research_text}
""".strip()


def write_article_gpt(final_title: str, final_topic: str, teen_style_notes: str, research_text: str) -> str:
    prompt = build_article_prompt(final_title, final_topic, teen_style_notes, research_text)

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
            }
        ]
    )

    return (response.choices[0].message.content or "").strip()


def write_article_claude(final_title: str, final_topic: str, teen_style_notes: str, research_text: str) -> str:
    if not anthropic_client:
        return "<p>Claude generation is unavailable because ANTHROPIC_API_KEY is missing.</p>"

    prompt = build_article_prompt(final_title, final_topic, teen_style_notes, research_text)

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2500,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.subheader("Settings")
    max_results = st.slider("Tavily results", min_value=3, max_value=10, value=5)
    recent_post_limit = st.slider("Recent WordPress posts to compare", min_value=3, max_value=20, value=10)
    show_html_code = st.checkbox("Show raw HTML", value=False)
    preferred_publish_version = st.selectbox(
        "Default draft selection",
        ["Do not create draft", "GPT", "Claude"],
        index=0
    )


# ----------------------------
# Main form
# ----------------------------
topic = st.text_input(
    "Enter a topic",
    value="Latest space discovery"
)

run_btn = st.button("Research, Compare, and Draft")


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    try:
        with st.spinner("Searching the web..."):
            research_results = tavily_search(query=topic, max_results=max_results)
            research_text = build_research_text(research_results)

        with st.spinner("Checking your recent WordPress posts..."):
            recent_posts = get_recent_posts(limit=recent_post_limit)
            recent_posts_text = build_recent_posts_text(recent_posts)

        with st.spinner("Choosing the best angle..."):
            topic_choice = choose_best_topic(
                user_topic=topic,
                research_text=research_text,
                recent_posts_text=recent_posts_text
            )

        final_title = topic_choice.get("final_title", topic.strip().title())
        final_topic = topic_choice.get("final_topic", topic.strip())
        why_selected = topic_choice.get("why_selected", "")
        teen_style_notes = topic_choice.get("teen_style_notes", "Use simple, engaging language.")
        excerpt = topic_choice.get("excerpt", "")

        with st.spinner("Writing GPT version..."):
            article_html_gpt = write_article_gpt(
                final_title=final_title,
                final_topic=final_topic,
                teen_style_notes=teen_style_notes,
                research_text=research_text
            )

        with st.spinner("Writing Claude version..."):
            article_html_claude = write_article_claude(
                final_title=final_title,
                final_topic=final_topic,
                teen_style_notes=teen_style_notes,
                research_text=research_text
            )

        st.success("Both article versions generated successfully.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Chosen Topic", "GPT Article", "Claude Article", "Research", "Recent Posts"]
        )

        with tab1:
            st.subheader(final_title)
            st.write("**Chosen angle:**", final_topic)
            st.write("**Why selected:**", why_selected)
            st.write("**Tone guidance:**", teen_style_notes)
            st.write("**Excerpt:**", excerpt)

        with tab2:
            st.subheader("GPT Version")

            if show_html_code:
                st.code(article_html_gpt, language="html")
                st.markdown("---")

            st.markdown("### Rendered Preview")
            st.components.v1.html(article_html_gpt, height=700, scrolling=True)

        with tab3:
            st.subheader("Claude Version")

            if show_html_code:
                st.code(article_html_claude, language="html")
                st.markdown("---")

            st.markdown("### Rendered Preview")
            st.components.v1.html(article_html_claude, height=700, scrolling=True)

        with tab4:
            st.subheader("Web Research Results")
            for idx, item in enumerate(research_results, start=1):
                st.markdown(f"**{idx}. {item.get('title', '')}**")
                st.write(item.get("url", ""))
                st.write(item.get("content", ""))
                st.markdown("---")

        with tab5:
            st.subheader("Recent WordPress Posts")
            for p in recent_posts:
                st.markdown(f"- **{p.get('title', '')}**")
                st.write(p.get("link", ""))
                st.caption(p.get("date", ""))
                st.markdown("---")

        st.markdown("## Choose version to send to WordPress")

        selected_version = st.radio(
            "Which version should be drafted?",
            ["Do not create draft", "GPT", "Claude"],
            index=["Do not create draft", "GPT", "Claude"].index(preferred_publish_version)
        )

        selected_html = None
        if selected_version == "GPT":
            selected_html = article_html_gpt
        elif selected_version == "Claude":
            selected_html = article_html_claude

        if selected_html:
            if st.button("Create selected WordPress draft"):
                with st.spinner("Creating WordPress draft..."):
                    post = create_draft_post(
                        title=final_title,
                        content=selected_html,
                        excerpt=excerpt
                    )

                st.success("Draft created in WordPress.")
                st.write(f"**Post ID:** {post.get('id')}")
                st.write(f"**Status:** {post.get('status')}")
                st.write(f"**Title:** {post.get('title')}")
                st.write(f"**Link:** {post.get('link')}")
                st.json(post)

    except Exception as e:
        st.error(f"Error: {str(e)}")
