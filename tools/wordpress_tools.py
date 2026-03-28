# wordpress_tools.py

import os
import html
import requests
from langchain_core.tools import tool

WP_BASE_URL = os.getenv("WORDPRESS_BASE_URL", "https://riolabs.in").rstrip("/")
WP_API_POSTS = f"{WP_BASE_URL}/wp-json/wp/v2/posts"
WP_API_MEDIA = f"{WP_BASE_URL}/wp-json/wp/v2/media"

WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")


def _wp_auth():
    if not WP_USER or not WP_PASS:
        raise ValueError(
            "Missing WORDPRESS_USERNAME or WORDPRESS_APP_PASSWORD in environment variables."
        )
    return (WP_USER, WP_PASS)


def _wp_headers() -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; RiolabsContentAgent/1.0)"
    }


def get_recent_posts(limit: int = 10) -> list[dict]:
    """
    Fetch recent published posts from WordPress.
    """
    params = {
        "per_page": limit,
        "_fields": "id,date,slug,title,link"
    }

    response = requests.get(
        WP_API_POSTS,
        params=params,
        headers=_wp_headers(),
        timeout=(10, 30)
    )
    response.raise_for_status()

    posts = response.json()

    cleaned_posts = []
    for post in posts:
        cleaned_posts.append({
            "id": post.get("id"),
            "date": post.get("date"),
            "slug": post.get("slug"),
            "title": html.unescape(post.get("title", {}).get("rendered", "")),
            "link": post.get("link")
        })

    return cleaned_posts


@tool
def get_recent_posts_tool(limit: int = 10) -> str:
    """
    Fetch recent posts already published on the user's WordPress website.
    Use this tool when you need to know what topics already exist on the user's blog,
    avoid duplicate ideas, compare against existing content, or understand what has already been published.
    Returns recent post titles and links.
    """
    try:
        posts = get_recent_posts(limit=limit)

        if not posts:
            return "No recent posts were found on the website."

        return "\n".join(
            f"{p['title']} ({p['link']})"
            for p in posts
        )

    except requests.exceptions.ConnectTimeout:
        return "Could not connect to the WordPress website in time. The site may be slow or temporarily unreachable."

    except requests.exceptions.ReadTimeout:
        return "The WordPress website took too long to respond."

    except requests.exceptions.RequestException as e:
        return f"Failed to fetch recent posts from WordPress: {str(e)}"


def create_draft_post(
    title: str,
    content: str,
    excerpt: str = "",
    categories: list[int] | None = None,
    tags: list[int] | None = None
) -> dict:
    """
    Create a WordPress draft post.
    """
    payload = {
        "title": title,
        "content": content,
        "status": "draft"
    }

    if excerpt:
        payload["excerpt"] = excerpt
    if categories:
        payload["categories"] = categories
    if tags:
        payload["tags"] = tags

    response = requests.post(
        WP_API_POSTS,
        json=payload,
        auth=_wp_auth(),
        headers=_wp_headers(),
        timeout=(10, 30)
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    if response.status_code != 201:
        raise RuntimeError(f"WordPress draft creation failed: {response.status_code} - {data}")

    return {
        "id": data.get("id"),
        "link": data.get("link"),
        "slug": data.get("slug"),
        "status": data.get("status"),
        "title": html.unescape(data.get("title", {}).get("rendered", title))
    }


@tool
def create_draft_post_tool(title: str, content: str, excerpt: str = "") -> str:
    """
    Create a draft post in WordPress.
    Use this tool after article generation is complete and ready to be saved as a draft on the user's website.
    """
    try:
        post = create_draft_post(title=title, content=content, excerpt=excerpt)

        return (
            f"Draft created successfully.\n"
            f"ID: {post['id']}\n"
            f"Status: {post['status']}\n"
            f"Title: {post['title']}\n"
            f"Link: {post['link']}"
        )

    except ValueError as e:
        return f"Configuration error: {str(e)}"

    except requests.exceptions.ConnectTimeout:
        return "Could not connect to WordPress in time while creating the draft."

    except requests.exceptions.ReadTimeout:
        return "WordPress took too long to respond while creating the draft."

    except requests.exceptions.RequestException as e:
        return f"HTTP error while creating draft post: {str(e)}"

    except Exception as e:
        return f"Failed to create draft post: {str(e)}"


def upload_image_to_wordpress(
    image_bytes: bytes,
    filename: str,
    mime_type: str = "image/png"
) -> dict:
    """
    Upload an image to the WordPress media library.
    """
    headers = _wp_headers()
    headers.update({
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": mime_type
    })

    response = requests.post(
        WP_API_MEDIA,
        data=image_bytes,
        auth=_wp_auth(),
        headers=headers,
        timeout=(20, 60)
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    if response.status_code not in (200, 201):
        raise RuntimeError(f"WordPress media upload failed: {response.status_code} - {data}")

    return {
        "id": data.get("id"),
        "link": data.get("link"),
        "source_url": data.get("source_url")
    }


def set_featured_image(post_id: int, media_id: int) -> dict:
    """
    Attach uploaded media as featured image to an existing post.
    """
    payload = {
        "featured_media": media_id
    }

    response = requests.post(
        f"{WP_API_POSTS}/{post_id}",
        json=payload,
        auth=_wp_auth(),
        headers=_wp_headers(),
        timeout=(10, 30)
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    if response.status_code not in (200, 201):
        raise RuntimeError(f"Setting featured image failed: {response.status_code} - {data}")

    return data
