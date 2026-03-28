import streamlit as st
import requests
import os

st.title("WP Post Creator Test")

WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")
WP_API_URL = "https://riolabs.in/wp-json/wp/v2/posts"

title = st.text_input("Post title", "Test post from Render")
content = st.text_area("Post content", "This is a test post created from Render.")

if st.button("Create Draft Post"):
    try:
        payload = {
            "title": title,
            "content": content,
            "status": "draft"
        }

        r = requests.post(
            WP_API_URL,
            json=payload,
            auth=(WP_USER, WP_PASS),
            timeout=(10, 30)
        )

        st.write("Status:", r.status_code)
        st.json(r.json())

    except Exception as e:
        st.error(str(e))
