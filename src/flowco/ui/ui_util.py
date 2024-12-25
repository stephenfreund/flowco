import functools
import pandas as pd
import streamlit as st

from typing import Any, Callable, Iterator, List
import uuid

from streamlit_extras.stylable_container import stylable_container

from flowco.dataflow.dfg import Node
from flowco.dataflow.phase import Phase
from flowco.util.config import AbstractionLevel


def editable_list(
    title: str,
    items: List[str],
    on_refresh: Callable[[], None],
    on_change: Callable[[List[str]], None],
    disabled: bool = False,
):
    def internal_on_change():
        new_list = []
        for i in range(len(items) + 1):
            # be careful -- some of the text_areas may be gone...
            if f"{uid}_{i}" in st.session_state.keys():
                item = st.session_state[f"{uid}_{i}"]
                # print(item)
                if item != "":
                    new_list.append(item)
        # print(new_list)
        # traceback.print_stack()
        on_change(new_list)

    def internal_refresh():
        pass

    uid = str(uuid.uuid4())

    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    st.button(
        ":material/refresh:", key=f"{uid}", on_click=internal_refresh, disabled=disabled
    )

    st.markdown(f"<span style='font-size:14px;'>{title}</span>", unsafe_allow_html=True)

    for i, s in enumerate(items + [""]):
        st.text_area(
            f"Item {i+1}",
            value=s,
            key=f"{uid}_{i}",
            label_visibility="collapsed",
            height=56,
            on_change=internal_on_change,
            disabled=disabled,
            placeholder="Enter requirement here...",
        )


def editable_text(
    title: str,
    text: str,
    on_refresh: Callable[[], None],
    on_change: Callable[[str], None],
    disabled: bool = False,
):
    def internal_on_change():
        on_change(st.session_state[f"{uid}_text"])

    def internal_refresh():
        pass

    uid = str(uuid.uuid4())

    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    st.button(
        ":material/refresh:", key=f"{uid}", on_click=internal_refresh, disabled=disabled
    )

    st.markdown(f"<span style='font-size:14px;'>{title}</span>", unsafe_allow_html=True)

    st.text_area(
        f"Item",
        value=text,
        key=f"{uid}_text",
        label_visibility="collapsed",
        height=300,
        on_change=internal_on_change,
        disabled=disabled,
        placeholder="Enter text here...",
    )


def editable_code(
    title: str,
    text: str,
    on_refresh: Callable[[], None],
    on_change: Callable[[str], None],
    disabled: bool = False,
):
    uid = str(uuid.uuid4())
    with stylable_container(
        key=uid,
        css_styles="""
        textarea {
            font-family: monospace;
            font-size: 12px !important;    
        }
    """,
    ):
        editable_text(
            title=title,
            text=text,
            on_refresh=on_refresh,
            on_change=on_change,
            disabled=disabled,
        )


def flex_columns():
    return stylable_container(
        key="command_sidebar",
        css_styles="""
            /* Rule 1: Apply the styles only if the element is not inside a .main section */
            div[data-testid="column"]:not(.main div[data-testid="column"]):not(div[data-testid="stForm"] div[data-testid="column"]) {
                width: fit-content !important;
                flex: unset;
            }

            /* Rule 2: Apply the styles only if the element is not inside a .main section */
            div[data-testid="column"]:not(.main div[data-testid="column"]):not(div[data-testid="stForm"] div[data-testid="column"]) div {
                width: fit-content !important;
            }
                """,
    )


def toggle(key):
    st.session_state[key] = not st.session_state[key]


def set_session_state(key, value):
    st.session_state[key] = value
    return value


def show_requirements():
    return st.session_state.show_requirements and AbstractionLevel.show_requirements(
        st.session_state.abstraction_level
    )


def show_algorithm():
    return st.session_state.show_algorithm and AbstractionLevel.show_algorithm(
        st.session_state.abstraction_level
    )


def show_code():
    return st.session_state.show_code and AbstractionLevel.show_code(
        st.session_state.abstraction_level
    )


def visible_phases() -> Iterator[Phase]:
    if show_requirements():
        yield Phase.requirements
    if show_algorithm():
        yield Phase.algorithm
    if show_code():
        yield Phase.code


def phase_for_last_shown_part() -> Phase:
    if show_code():
        return Phase.code
    elif show_algorithm():
        return Phase.algorithm
    elif show_requirements():
        return Phase.requirements
    else:
        return Phase.clean
