import os
import json
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic

from tools.tavily_research import tavily_search
from tools.wordpress_tools import (
    get_recent_posts,
    create_draft_post,
    update_post,
    upload_media_to_wordpress,
)
from tools.image_tools import (
    generate_image_prompt,
    generate_grok_image,
)


st.set_page_config(page_title="AI WordPress Publisher", layout="wide")
st.title("AI WordPress Publisher")
st.caption(
    "Research a topic, choose a better angle, generate GPT and Claude versions, "
    "generate a featured image with Grok, then save the selected result to WordPress as a draft."
)


def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
XAI_API_KEY = get_secret("XAI_API_KEY")
WP_BASE_URL = get_secret("WORDPRESS_BASE_URL")
WP_USER = get_secret("WORDPRESS_USERNAME")
WP_PASS = get_secret("WORDPRESS_APP_PASSWORD")

missing = []
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")
if not TAVILY_API_KEY:
    missing.append("TAVILY_API_KEY")
if not WP_BASE_URL:
    missing.append("WORDPRESS_BASE_URL")
if not WP_USER:
    missing.append("WORDPRESS_USERNAME")
if not WP_PASS:
    missing.append("WORDPRESS_APP_PASSWORD")

if missing:
    st.error("Missing environment variables: " + ", ".join(missing))
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


DEFAULTS = {
    "generated": False,
    "final_title": "",
    "final_topic": "",
    "why_selected": "",
    "teen_style_notes": "",
    "excerpt": "",
    "article_html_gpt": "",
    "article_html_claude": "",
    "research_results": [],
    "recent_posts": [],
    "judge_result": None,
    "selected_version": "Do not create draft",
    "image_prompt_data": None,
    "generated_image": None,
    "uploaded_media": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def clear_results():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


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


def judge_articles(final_title: str, final_topic: str, research_text: str, article_html_gpt: str, article_html_claude: str) -> dict:
    prompt = f"""
You are evaluating two blog article drafts for WordPress.

Title:
{final_title}

Chosen topic:
{final_topic}

Research context:
{research_text}

Draft A (GPT):
{article_html_gpt}

Draft B (Claude):
{article_html_claude}

Evaluate both drafts using these criteria:
1. Factual accuracy based on research context
2. Clarity for Indian teenagers
3. Structure and readability
4. WordPress suitability
5. Engagement without sounding childish
6. Avoiding unsupported claims

Return STRICT JSON only:
{{
  "winner": "GPT or Claude",
  "reason": "Short explanation",
  "scores": {{
    "GPT": {{
      "accuracy": 0,
      "clarity": 0,
      "structure": 0,
      "wordpress_readiness": 0,
      "engagement": 0,
      "total": 0
    }},
    "Claude": {{
      "accuracy": 0,
      "clarity": 0,
      "structure": 0,
      "wordpress_readiness": 0,
      "engagement": 0,
      "total": 0
    }}
  }}
}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
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
            "winner": "GPT",
            "reason": "Fallback winner because judge JSON parsing failed.",
            "scores": {
                "GPT": {"accuracy": 0, "clarity": 0, "structure": 0, "wordpress_readiness": 0, "engagement": 0, "total": 0},
                "Claude": {"accuracy": 0, "clarity": 0, "structure": 0, "wordpress_readiness": 0, "engagement": 0, "total": 0},
            },
        }


with st.sidebar:
    st.subheader("Settings")
    max_results = st.slider("Tavily results", min_value=3, max_value=10, value=5)
    recent_post_limit = st.slider("Recent WordPress posts to compare", min_value=3, max_value=20, value=10)
    show_html_code = st.checkbox("Show raw HTML", value=False)

    st.markdown("### Article selection")
    auto_judge = st.checkbox("Let AI compare GPT vs Claude automatically", value=True)
    auto_pick_winner = st.checkbox("Automatically select the judged winner", value=True)
    preferred_publish_version = st.selectbox(
        "Default draft selection",
        ["Do not create draft", "GPT", "Claude"],
        index=0
    )

    st.markdown("### Image generation")
    auto_generate_image = st.checkbox("Generate featured image with Grok", value=True)
    upload_image_to_wp = st.checkbox("Upload image to WordPress Media", value=True)
    set_as_featured_image = st.checkbox("Use uploaded image as featured image", value=True)
    insert_image_into_article = st.checkbox("Insert image at top of article HTML", value=False)
    image_aspect_ratio = st.selectbox(
        "Image aspect ratio",
        ["16:9", "4:3", "3:2", "1:1"],
        index=0
    )

    st.markdown("---")
    if st.button("Clear generated results"):
        clear_results()
        st.rerun()


topic = st.text_input("Enter a topic", value="Latest space discovery")
run_btn = st.button("Research, Compare, Generate Image, and Draft")

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

        judge_result = None
        if auto_judge:
            with st.spinner("Judging both versions..."):
                judge_result = judge_articles(
                    final_title=final_title,
                    final_topic=final_topic,
                    research_text=research_text,
                    article_html_gpt=article_html_gpt,
                    article_html_claude=article_html_claude,
                )

        selected_version = preferred_publish_version
        if auto_judge and auto_pick_winner and judge_result:
            winner = judge_result.get("winner", "").strip()
            if winner in ["GPT", "Claude"]:
                selected_version = winner

        image_prompt_data = None
        generated_image = None
        uploaded_media = None

        if auto_generate_image:
            with st.spinner("Creating image prompt..."):
                image_prompt_data = generate_image_prompt(
                    openai_client=openai_client,
                    final_title=final_title,
                    final_topic=final_topic,
                    excerpt=excerpt,
                    research_text=research_text,
                )

            if XAI_API_KEY:
                with st.spinner("Generating image with Grok..."):
                    generated_image = generate_grok_image(
                        prompt=image_prompt_data["image_prompt"],
                        aspect_ratio=image_aspect_ratio,
                    )

                if upload_image_to_wp and generated_image:
                    with st.spinner("Uploading image to WordPress..."):
                        uploaded_media = upload_media_to_wordpress(
                            image_bytes=generated_image["image_bytes"],
                            filename=generated_image["filename"],
                            content_type=generated_image["content_type"],
                            alt_text=image_prompt_data.get("alt_text", ""),
                            caption=image_prompt_data.get("caption", ""),
                        )

        st.session_state.generated = True
        st.session_state.final_title = final_title
        st.session_state.final_topic = final_topic
        st.session_state.why_selected = why_selected
        st.session_state.teen_style_notes = teen_style_notes
        st.session_state.excerpt = excerpt
        st.session_state.article_html_gpt = article_html_gpt
        st.session_state.article_html_claude = article_html_claude
        st.session_state.research_results = research_results
        st.session_state.recent_posts = recent_posts
        st.session_state.judge_result = judge_result
        st.session_state.selected_version = selected_version
        st.session_state.image_prompt_data = image_prompt_data
        st.session_state.generated_image = generated_image
        st.session_state.uploaded_media = uploaded_media

        st.success("Generation completed successfully.")

    except Exception as e:
        st.error(f"Error: {str(e)}")


if st.session_state.generated:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Chosen Topic", "GPT Article", "Claude Article", "Research", "Recent Posts", "Judge Result", "Image"]
    )

    with tab1:
        st.subheader(st.session_state.final_title)
        st.write("**Chosen angle:**", st.session_state.final_topic)
        st.write("**Why selected:**", st.session_state.why_selected)
        st.write("**Tone guidance:**", st.session_state.teen_style_notes)
        st.write("**Excerpt:**", st.session_state.excerpt)

    with tab2:
        st.subheader("GPT Version")
        if show_html_code:
            st.code(st.session_state.article_html_gpt, language="html")
            st.markdown("---")
        st.components.v1.html(st.session_state.article_html_gpt, height=700, scrolling=True)

    with tab3:
        st.subheader("Claude Version")
        if show_html_code:
            st.code(st.session_state.article_html_claude, language="html")
            st.markdown("---")
        st.components.v1.html(st.session_state.article_html_claude, height=700, scrolling=True)

    with tab4:
        st.subheader("Web Research Results")
        for idx, item in enumerate(st.session_state.research_results, start=1):
            st.markdown(f"**{idx}. {item.get('title', '')}**")
            st.write(item.get("url", ""))
            st.write(item.get("content", ""))
            st.markdown("---")

    with tab5:
        st.subheader("Recent WordPress Posts")
        for p in st.session_state.recent_posts:
            st.markdown(f"- **{p.get('title', '')}**")
            st.write(p.get("link", ""))
            st.caption(p.get("date", ""))
            st.markdown("---")

    with tab6:
        st.subheader("AI Judge")
        if st.session_state.judge_result:
            st.json(st.session_state.judge_result)
            st.success(f"Recommended winner: {st.session_state.judge_result.get('winner', 'Unknown')}")
            st.write(st.session_state.judge_result.get("reason", ""))
        else:
            st.info("Judge was not run for this generation.")

    with tab7:
        st.subheader("Generated Image")
        if st.session_state.image_prompt_data:
            st.write("**Prompt:**")
            st.code(st.session_state.image_prompt_data.get("image_prompt", ""), language="text")
            st.write("**Alt text:**", st.session_state.image_prompt_data.get("alt_text", ""))
            st.write("**Caption:**", st.session_state.image_prompt_data.get("caption", ""))
            st.write("**Style notes:**", st.session_state.image_prompt_data.get("style_notes", ""))

        if st.session_state.generated_image:
            st.image(
                st.session_state.generated_image["image_bytes"],
                caption="Generated featured image",
                use_container_width=True,
            )

        if st.session_state.uploaded_media:
            st.success(f"Image uploaded to WordPress Media. Media ID: {st.session_state.uploaded_media.get('id')}")
            source_url = st.session_state.uploaded_media.get("source_url")
            if source_url:
                st.write(source_url)
        elif st.session_state.image_prompt_data and not XAI_API_KEY:
            st.warning("Image prompt created, but XAI_API_KEY is missing so no image was generated.")
        else:
            st.info("No image was generated for this run.")

    st.markdown("## Choose version to send to WordPress")

    options = ["Do not create draft", "GPT", "Claude"]
    current_selection = st.session_state.selected_version
    if current_selection not in options:
        current_selection = "Do not create draft"

    selected_version = st.radio(
        "Which version should be drafted?",
        options,
        index=options.index(current_selection),
        key="selected_version_radio"
    )
    st.session_state.selected_version = selected_version

    selected_html = None
    if selected_version == "GPT":
        selected_html = st.session_state.article_html_gpt
    elif selected_version == "Claude":
        selected_html = st.session_state.article_html_claude

    if selected_html:
        uploaded_media = st.session_state.uploaded_media
        featured_media_id = None
        image_url = None
        alt_text = ""

        if uploaded_media:
            featured_media_id = uploaded_media.get("id")
            image_url = uploaded_media.get("source_url")

        if st.session_state.image_prompt_data:
            alt_text = st.session_state.image_prompt_data.get("alt_text", "")

        st.info(f"Selected version: {selected_version}")
        st.write("Featured media ID:", featured_media_id)
        st.write("Will set featured image:", set_as_featured_image and bool(featured_media_id))
        st.write("Will insert image into content:", insert_image_into_article and bool(image_url))

        if st.button("Create selected WordPress draft"):
            try:
                with st.spinner("Step 1/3: Creating text-only draft..."):
                    post = create_draft_post(
                        title=st.session_state.final_title,
                        content=selected_html,
                        excerpt=st.session_state.excerpt,
                    )

                post_id = post["id"]

                with st.spinner("Step 2/3: Updating featured image..."):
                    if set_as_featured_image and featured_media_id:
                        update_post(
                            post_id=post_id,
                            featured_media=featured_media_id,
                        )

                final_updated_post = post

                with st.spinner("Step 3/3: Updating content with inline image..."):
                    if insert_image_into_article and image_url:
                        image_block = f'<p><img src="{image_url}" alt="{alt_text}"></p>'
                        final_html = image_block + "\n" + selected_html

                        final_updated_post = update_post(
                            post_id=post_id,
                            content=final_html,
                        )
                    elif set_as_featured_image and featured_media_id:
                        final_updated_post = update_post(
                            post_id=post_id,
                            featured_media=featured_media_id,
                        )

                st.success("Draft created in WordPress.")
                st.write(f"**Post ID:** {final_updated_post.get('id')}")
                st.write(f"**Status:** {final_updated_post.get('status')}")
                st.write(f"**Title:** {final_updated_post.get('title')}")
                st.write(f"**Link:** {final_updated_post.get('link')}")
                st.json(final_updated_post["raw"])

            except Exception as e:
                st.error(f"Draft creation failed: {str(e)}")
