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

    if not upload_response.ok:
        raise Exception(
            f"WordPress media upload failed.\n"
            f"Status: {upload_response.status_code}\n"
            f"Response: {upload_response.text}"
        )

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

        if not meta_response.ok:
            raise Exception(
                f"WordPress media metadata update failed.\n"
                f"Status: {meta_response.status_code}\n"
                f"Response: {meta_response.text}\n"
                f"Payload: {meta_payload}"
            )

        media = meta_response.json()

    return media


def create_draft_post(
    title: str,
    content: str,
    excerpt: str = "",
) -> dict:
    """
    Safest version: create the post with text only.
    No featured_media during creation.
    """
    base_url = get_wp_base_url()
    auth = get_wp_auth()

    payload = {
        "title": title,
        "content": content,
        "excerpt": excerpt,
        "status": "draft",
    }

    response = requests.post(
        f"{base_url}/wp-json/wp/v2/posts",
        auth=auth,
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise Exception(
            f"WordPress post creation failed.\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}\n"
            f"Payload: {payload}"
        )

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


def update_post(
    post_id: int,
    featured_media: Optional[int] = None,
    content: Optional[str] = None,
) -> dict:
    """
    Update an existing post after creation.
    Use this to attach featured image and/or update content safely.
    """
    base_url = get_wp_base_url()
    auth = get_wp_auth()

    payload = {}
    if featured_media is not None:
        payload["featured_media"] = featured_media
    if content is not None:
        payload["content"] = content

    if not payload:
        raise ValueError("update_post called with no fields to update")

    response = requests.post(
        f"{base_url}/wp-json/wp/v2/posts/{post_id}",
        auth=auth,
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise Exception(
            f"WordPress post update failed.\n"
            f"Post ID: {post_id}\n"
            f"Status: {response.status_code}\n"
            f"Response: {response.text}\n"
            f"Payload: {payload}"
        )

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
