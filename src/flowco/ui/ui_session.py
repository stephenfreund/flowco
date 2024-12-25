# from typing import Type, TypeVar, cast
# from flowco.session.session import FlowcoSession
# import streamlit as st


# class StreamlitSession(FlowcoSession):
#     def __init__(self):
#         pass

#     T = TypeVar("T")

#     def _set(self, key: str, value: T):
#         full_key = self.full_key(key)
#         print(f"Setting {full_key} to {value}")
#         st.session_state[full_key] = value

#     T = TypeVar("T")

#     def get(self, key: str, type_: Type[T]) -> T:
#         full_key = self.full_key(key)
#         return cast(type_, st.session_state[full_key])
