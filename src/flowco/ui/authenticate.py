import os
import textwrap
from typing import Dict
import uuid
import streamlit as st
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import json
from datetime import datetime, timedelta
from dataclasses import dataclass

from flowco.session import session_file_system


# Constants
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
FOLDER_NAME = "Flowco"
FILE_EXTENSION = ".flowco"

# Initialize Streamlit session state for credentials
if "credentials" not in st.session_state:
    st.session_state.credentials = None


@dataclass
class CacheEntry:
    credentials: Credentials
    user_email: str
    timestamp: datetime


@st.cache_resource
def cache() -> Dict[str, CacheEntry]:
    return dict()


def purge_stale_entries(cache_dict):
    now = datetime.now()
    for key, entry in list(cache_dict.items()):
        if now - entry.timestamp > timedelta(hours=24):
            del cache_dict[key]


def sign_in(authorization_url: str):

    release = os.getenv("RELEASE_VERSION", "unknown")

    instructions = textwrap.dedent(
        f"""\
        * **Signing in with Google** creates an account on our server associated with your email address.
        * **Signing in as Guest** creates a temporary account for the current session that will be deleted when you close the browser tab.
        * Click "Report Bug" whenever you see something fishy!
        """
    )

    st.write(f"# Flowco {release}!")
    st.write(instructions)


    st.link_button("Sign In With Google", authorization_url)
    with st.sidebar:
        st.image("static/flowco.png")


# Function to fetch user information from id_token
def fetch_user_info_from_id_token(id_token_str):
    try:
        google_client_config = st.secrets["google_client_secrets"]

        # Specify the CLIENT_ID of the app that accesses the backend:
        CLIENT_ID = google_client_config["client_id"]
        # Verify the integrity of the id_token
        id_info = id_token.verify_oauth2_token(
            id_token_str, google_requests.Request(), CLIENT_ID
        )

        # ID token is valid. Get the user's Google Account information from the decoded token.
        return {
            "email": id_info.get("email"),
            "name": id_info.get("name"),
            "picture": id_info.get("picture"),
            "given_name": id_info.get("given_name"),
            "family_name": id_info.get("family_name"),
            "locale": id_info.get("locale"),
        }
    except ValueError as e:
        # Invalid token
        st.error(f"Invalid token")
        st.exception(e)
        return None


# Function to initialize OAuth Flow
def oauth_authenticate():
    if st.session_state.credentials is None:

        cache_dict = cache()
        purge_stale_entries(cache_dict)

        key = st.context.cookies["_streamlit_xsrf"].split("|")[-1]
        print("_streamlit_xsrf key", key, cache_dict.keys())
        if key in cache_dict:
            print("Cache hit")
            st.session_state.credentials = cache_dict[key].credentials
            st.session_state.user_email = cache_dict[key].user_email
            return

        if "auth_state" not in st.session_state:
            st.session_state.auth_state = "initial"

        environment = st.secrets["FLOWCO_ENVIRONMENT"]
        google_client_config = st.secrets["google_client_secrets"]

        if environment == "production":
            redirect_uri = google_client_config["redirect_uris"][1]  # Production URI
        else:
            redirect_uri = google_client_config["redirect_uris"][0]  # Localhost URI

        client_config = {
            "web": google_client_config,
        }

        # print(st.context.cookies["_streamlit_xsrf"], st.session_state.auth_state)

        if st.session_state.auth_state == "initial":
            flow = Flow.from_client_config(
                client_config=client_config,
                scopes=SCOPES,
                redirect_uri=redirect_uri,
            )
            authorization_url, _ = flow.authorization_url(prompt="consent")
            sign_in(authorization_url)
            st.session_state.auth_state = "waiting_for_code"
            return None

        elif st.session_state.auth_state == "waiting_for_code":
            if "code" in st.query_params:
                flow = Flow.from_client_config(
                    client_config=client_config,
                    scopes=SCOPES,
                    redirect_uri=redirect_uri,
                )
                flow.fetch_token(code=st.query_params["code"])
                st.session_state.token = flow.credentials.to_json()
                st.session_state.auth_state = "authenticated"

                session = flow.authorized_session()

                profile_info = session.get(
                    "https://www.googleapis.com/userinfo/v2/me"
                ).json()
                st.session_state.user_email = profile_info["email"]

                st.session_state.credentials = Credentials.from_authorized_user_info(
                    json.loads(st.session_state.token),
                )
                cache_dict[key] = CacheEntry(
                    credentials=st.session_state.credentials,
                    user_email=st.session_state.user_email,
                    timestamp=datetime.now(),
                )

                st.query_params.clear()  # Clear the query parameters after use
                st.rerun()
            else:
                return None

        elif st.session_state.auth_state == "authenticated":
            st.session_state.credentials = Credentials.from_authorized_user_info(
                json.loads(st.session_state.token),
            )
            cache_dict[key] = CacheEntry(
                credentials=st.session_state.credentials,
                user_email=st.session_state.user_email,
                timestamp=datetime.now(),
            )


def sign_out():
    key = st.context.cookies["_streamlit_xsrf"].split("|")[-1]
    del cache()[key]
    st.session_state.credentials = None
    st.session_state.user_email = None
    st.session_state.auth_state = "initial"
    st.session_state.token = None
    st.session_state.query_params = None
    st.stop()


def authenticate():

    if st.session_state.credentials is None:
        oauth_authenticate()
        # while st.session_state.credentials is None:
        oauth_authenticate()

        if st.button("Sign In As Guest"):
            cache_dict = cache()
            purge_stale_entries(cache_dict)
            key = st.context.cookies["_streamlit_xsrf"].split("|")[-1]
            st.session_state.credentials = "Guest"
            st.session_state.user_email = session_file_system.SessionFileSystem.make_unique_path("s3://go-flowco/", "guest")
            st.session_state.auth_state = "authenticated"
            cache_dict[key] = CacheEntry(
                credentials=st.session_state.credentials,
                user_email=st.session_state.user_email,
                timestamp=datetime.now(),
            )       
            st.rerun()

        if st.session_state.credentials is None:
            st.stop()
