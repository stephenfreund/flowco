from argparse import Namespace
from typing import Tuple
from flowco.util.config import Config
from flowco.util.output import log
import streamlit as st
import sys


@st.cache_data
def parse_args() -> Tuple[Config, Namespace]:
    config = Config()
    parser = config.parser()
    parser.add_argument("page", type=str, help="Name of the page")
    parser.add_argument(
        "--user_email",
        default=None,
        type=str,
    )
    args = parser.parse_args(sys.argv[1:])
    return config, args
