import streamlit as st
import requests
import os

st.title("WP Connectivity Test")

WP_USER = os.getenv("WORDPRESS_USERNAME")
WP_PASS = os.getenv("WORDPRESS_APP_PASSWORD")

if st.button("Test WordPress"):
    try:
        r = requests.get(
            "https://riolabs.in/wp-json/wp/v2/posts",
            params={"per_page": 3},
            timeout=(10, 30)
        )
        st.write("Status:", r.status_code)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))
