import textwrap
import streamlit as st


@st.dialog("Help", width="large")
def help_dialog():
    st.write(
        textwrap.dedent(
            """\
        ## Dataflow Graph
        * **Select node:** Click on it
        * **Create node:** Shift-click on canvas
        * **Move node:** Click and drag
        * **Delete node:** Click on node and press delete
        * **Add edge:** Hover over label for green halo and drag to target
        * **Add edge and node:** Hover over label for green halo and drag to empty space
        * **Delete edge:** Click on edge and press delete

        ## Sidebar
        * The sidebar shows the current node.
        * Anything LLM generated is in orange. 
        * **Build:** Build the whole graph.
        * **Test:** Write and run tests. 
        * **Run:** Run the graph. (Must build first)
        * **Choose what to view** through the check boxes.
        * **Report:** Generate a report for the graph.  (Should run first)
        * **Settings:** Change settings, including LLM.
                                
        ## Node Colors
        * **Gray:** Clean, all parts must be checked or generated.
        * **Red:** Requirements are ready.
        * **Pink:** Algorithm is ready.
        * **Orange:** Code generated but not checked to be runnable.
        * **Yellow:** Code is ready. (Can run it)
        * **Green:** Tests are ready. (Can run them)
        * **Blue:** Tests are passing.
        """
        )
    )
