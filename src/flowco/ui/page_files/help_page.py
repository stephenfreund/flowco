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

            st.write("## Welcome to Flowco!")

            st.image("static/flowco.png", width=100)


            st.write(
                textwrap.dedent(
                    """\
                ### Getting started
                * Begin by editing a simple diagram named "welcome.flowco".  
                * To select that project, select the **Projects** view in the top-left corner and click on the **welcome** project.
                * Then switch to the **Edit** view in the top-left corner.
                * Follow the instructions on the right-hand side of the screen to get started.

                After working on that project, we recommend following the numbered tutorials to learn more about Flowco.
                
                ### OpenAI API Key
                * For the first hour, you can use Flowco without providing an OpenAI API key.  After that, you'll need to provide an API key in Settings to continue using Flowco.

                ### Bugs
                * While many users have worked in Flowco, there are undoubtedly still bugs.  
                * Please report any bugs you find by clicking the "Report Bug" button at the bottom of the screen.
                """
                )
            )

            st.markdown("### Tutorials and Videos")

            st.link_button("Tutorial", "https://youtu.be/q0eAJv1vhAQ", type="primary")
            
            st.markdown("### FAQ")
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
