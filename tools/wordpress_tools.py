import os
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

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


def get_wp_base_url() -> str:
    base_url = get_secret("WORDPRESS_BASE_URL") or get_secret("WORDPRESS_SITE_URL")
    if not base_url:
        raise ValueError("Missing WORDPRESS_BASE_URL")
    return base_url.rstrip("/")


def get_wp_auth() -> HTTPBasicAuth:
    username = get_secret("WORDPRESS_USERNAME")
    app_password = get_secret("WORDPRESS_APP_PASSWORD")

    if not username or not app_password:
        raise ValueError("Missing WORDPRESS_USERNAME or WORDPRESS_APP_PASSWORD")

    return HTTPBasicAuth(username, app_password)


def get_recent_posts(limit: int = 10) -> list[dict]:
    """
    Returns recent published posts for duplicate checking.
    """
    base_url = get_wp_base_url()
    auth = get_wp_auth()

    response = requests.get(
        f"{base_url}/wp-json/wp/v2/posts",
        auth=auth,
        params={
            "per_page": limit,
            "status": "publish",
            "_fields": "id,date,link,title",
        },
        timeout=60,
    )
    response.raise_for_status()

    posts = response.json()
    cleaned = []

    for p in posts:
        title_obj = p.get("title", {})
        title = title_obj.get("rendered", "") if isinstance(title_obj, dict) else str(title_obj)

        cleaned.append(
            {
                "id": p.get("id"),
                "date": p.get("date", ""),
                "link": p.get("link", ""),
                "title": title,
            }
        )

    return cleaned


def upload_media_to_wordpress(
    image_bytes: bytes,
    filename: str,
    content_type: str = "image/png",
    alt_text: str = "",
    caption: str = "",
) -> dict:
    """
    Upload an image to the WordPress media library, then update alt text/caption.
    """
    base_url = get_wp_base_url()
    auth = get_wp_auth()

    upload_response = requests.post(
        f"{base_url}/wp-json/wp/v2/media",
        auth=auth,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": content_type,
        },
        data=image_bytes,
        timeout=120,
    )
    upload_response.raise_for_status()
    media = upload_response.json()

    media_id = media["id"]

    meta_payload = {}
    if alt_text:
        meta_payload["alt_text"] = alt_text
    if caption:
        meta_payload["caption"] = caption

    if meta_payload:
        meta_response = requests.post(
            f"{base_url}/wp-json/wp/v2/media/{media_id}",
            auth=auth,
            json=meta_payload,
            timeout=60,
        )
        meta_response.raise_for_status()
        media = meta_response.json()

    return media


def create_draft_post(
    title: str,
    content: str,
    excerpt: str = "",
    featured_media: Optional[int] = None,
) -> dict:
    """
    Create a WordPress post draft. Optionally set featured image.
    """
    base_url = get_wp_base_url()
    auth = get_wp_auth()

    payload = {
        "title": title,
        "content": content,
        "excerpt": excerpt,
        "status": "draft",
    }

    if featured_media is not None:
        payload["featured_media"] = featured_media

    response = requests.post(
        f"{base_url}/wp-json/wp/v2/posts",
        auth=auth,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    post = response.json()

    title_obj = post.get("title", {})
    clean_title = title_obj.get("rendered", "") if isinstance(title_obj, dict) else str(title_obj)

    return {
        "id": post.get("id"),
        "status": post.get("status"),
        "link": post.get("link"),
        "title": clean_title,
        "raw": post,
    }
