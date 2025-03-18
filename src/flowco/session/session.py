from __future__ import annotations
import os
from threading import current_thread
from typing import Type, TypeVar, cast
from abc import ABC, abstractmethod


class FlowcoSession(ABC):
    T = TypeVar("T")

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self._set(key, value)

    T = TypeVar("T")

    def _set(self, key: str, value: T):
        self.get_session()[key] = value

    T = TypeVar("T")

    def get(self, key: str, type_: Type[T]) -> T:
        return cast(type_, self.get_session()[key])

    @abstractmethod
    def get_session(self) -> dict:
        pass


if os.environ.get("FLOWCO_ENVIRONMENT", None) == None:

    print("[Starting command line session]\n")

    class TerminalSession(FlowcoSession):
        def __init__(self):
            self._session = {}

        def get_session(self):
            return self._session

    session = TerminalSession()

else:

    import streamlit as st

    print("[Starting Streamlit session]\n")

    class StreamlitSession(FlowcoSession):
        def __init__(self):
            st.session_state["flowco_session"] = {}

        def get_session(self) -> dict:
            thread_attr = getattr(current_thread(), "flowco_session", None)
            if thread_attr is not None:
                return thread_attr
            else:
                return st.session_state["flowco_session"]

    session = StreamlitSession()
