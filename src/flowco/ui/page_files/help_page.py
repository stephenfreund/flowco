import os
import textwrap
import streamlit as st
import yaml


class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h3")

    def header(self, text):
        self._markdown(text, "h4", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h5", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

    def entry(self, question, answer):
        key = "".join(filter(str.isalnum, question)).lower()
        self._items.append(f"    * <a href='#{key}'>{question}</a>")
        return f"<li id={key} style='line-height:1.3;'><b>{question}</b><br><br>\n\n{answer}\n\n</li>"


# toc = Toc()

# st.title("Table of contents")
# toc.placeholder()

# toc.title("Title")

# for a in range(10):
#     st.write("Blabla...")

# toc.header("Header 1")

# for a in range(10):
#     st.write("Blabla...")

# toc.header("Header 2")

# for a in range(10):
#     st.write("Blabla...")

# toc.subheader("Subheader 1")

# for a in range(10):
#     st.write("Blabla...")

# toc.subheader("Subheader 2")

# for a in range(10):
#     st.write("Blabla...")

# toc.generate()


class HelpPage:

    def main(self):
        with st.container(key="help_page"):
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.markdown("## Tutorial")

            # st.link_button("Watch", "https://youtu.be/q0eAJv1vhAQ", type="primary")

            st.markdown("## FAQ")
            toc = Toc()
            toc.placeholder(True)

            # load the yaml from the "faq.yaml" file in this files directory
            faq = yaml.safe_load(
                open(os.path.join(os.path.dirname(__file__), "faq.yaml"))
            )
            for section in faq:
                toc.header(section["section"])
                entries = [
                    toc.entry(entry["question"], entry["answer"])
                    for entry in section["questions"]
                ]
                st.markdown(f"<ul>{''.join(entries)}</ul>", unsafe_allow_html=True)

            toc.generate()
