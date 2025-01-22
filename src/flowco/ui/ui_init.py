import streamlit as st

css = """
.stMainBlockContainer {
    padding: 0rem;
    max-width: unset !important;
}
.stSidebar {
    min-width: 415px;
    max-width:1000px;
}

.stSidebar .st-key-node_sidepanel *,
.stSidebar .st-key-node_sidepanel {
    gap: 0rem !important;
    min-height: 0rem;
}

.stSidebar .st-key-masthead {
    gap: 0.25rem !important;
}

[data-testid="stLogoSpacer"] {
    height: 1rem;
}

[data-testid="stSidebarHeader"] {
    padding: 0rem;
}

[data-testid="stSidebarNav"] * {
    line-height: 1.5;
}

[data-testid="stSidebarNav"] li {
    padding: 0rem;
    font-size: 0.8rem;
    line-height: 1.5;
}

[data-testid="stSidebarUserContent"] {
    padding: 0rem 0.75rem 6rem;
}

.stExpander summary {
    padding: 0.5rem;
}

[data-testid="stExpanderDetails"]  {
    padding: 0.5rem;
}

/* New overriding rule */
.st-key-node_description .stMarkdown *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-node_output .stMarkdown *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-node_requirements .stMarkdown *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-node_algorithm .stMarkdown *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-node_code .stMarkdown *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-node_code .stCode *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-node_checks .stMarkdown *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-right-panel .stCode *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6),
.st-key-right-panel p, ul, ol, li
.st-key-right-panel .stMarkdown *:not(h1):not(h2):not(h3):not(h4):not(h5):not(h6)
 {
    font-size: 12px !important;
}
.st-key-right-panel h3 {
    font-size: 1.25rem !important;
}

header {
    background: rgb(247, 249, 253) !important;
}

.st-key-command_sidebar .stColumn {
    width: fit-content !important;
    flex: unset !important;
}

.st-key-abstraction_level * {
    font-size: 12px !important;
    font-family: "Source Sans Pro", sans-serif;    
}

.stMain {
    padding-top: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
    padding-bottom: 0rem;
    width: 100%;
    height: 100%;
    overflow: hidden;
    min-width: auto;
    max-width: initial;
}

/* Hide the "Hide" button in the sidebar */
[data-testid="stSidebarContent"] [data-testid="stBaseButton-headerNoPadding"] {
    display: none;
}

/* When .st-key-chat_container is a grandchild */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .st-key-chat_container) {
    background-color: white;
    padding: 0rem;
}


.st-key-button_bar .stHorizontalBlock {
    gap: 0rem !important;
    
}
.st-key-undo {
    margin-left: 1rem !important;
    padding-left: 1rem !important;
}


.st-key-button_bar .stColumn {
    width: fit-content !important;
    flex: unset !important;
}

.st-key-zoom_button_bar .stHorizontalBlock {
    gap: 0rem !important;
    
}
.st-key-zoom_button_bar .stColumn {
    width: fit-content !important;
    flex: unset !important;
    font-size: 16px !important;
}

/* Edit Dialog */

[aria-label="dialog"]:has(.st-key-edit_dialog) {
    width: 90% !important;
}

.st-key-current_label textarea,
.st-key-current_requirements textarea,
.st-key-current_algorithm textarea,
.st-key-current_code textarea,
.st-key-generated_label .stMarkdown *,
.st-key-generated_requirements .stMarkdown *,
.st-key-generated_algorithm .stMarkdown *,
.st-key-generated_code .stCode *,
.st-key-generated_code .stMarkdown *
{
    font-size: 12px !important;
}

.st-key-current_label textarea,
.st-key-current_requirements textarea,
.st-key-current_algorithm textarea,
.st-key-current_code textarea {
    min-height: 40px;
}

.st-key-generated_code *,
.st-key-current_code textarea {
    font-family: "Source Code Pro", monospace;
}

.st-key-current_dialog {
    gap: 0rem !important;
}

.st-key-edit_node_commands {
    padding-bottom: 1rem;
}

.st-key-edit_node_commands .stColumn {
    width: fit-content !important;
    flex: unset !important;
}

.st-key-controls .stColumn {
    width: fit-content !important;
    flex: unset !important;
}

.st-key-controls * {
    font-size: 12px !important;
}

.st-key-page_controls .stColumn {
    width: fit-content !important;
    flex: unset !important;
}

.st-key-bottom_bar .stColumn {
    width: fit-content !important;
    flex: unset !important;
}

.st-key-description_dialog .stColumn {
    width: fit-content !important;
    flex: unset !important;
}


/* Chat */

.stChatMessage {
    padding: 0.25rem;
}
.stChatMessage .stMarkdown * {
    font-size: 0.85rem !important;
}

.stChatMessage .stMarkdown code,
.stChatMessage .stMarkdown code * {
    font-size: 12px !important;
}

.st-key-voice_input,
.st-key-voice_input_node {
    margin-top: -3.6rem;
    margin-left: 0.1rem;
    width: 48px;

}

.st-key-ama_input,
.st-key-ama_input_node {
    width: calc(200% - 48px);
    min-width: 400px;
} 

.st-key-voice_input *,
.st-key-voice_input_node * {
    height: 2.5rem !important;
    background-color: unset !important;
}

.st-key-voice_input [data-testid="stAudioInputWaveformTimeCode"],
.st-key-voice_input [data-testid="stAudioInputWaveSurfer"],
.st-key-voice_input_node [data-testid="stAudioInputWaveformTimeCode"],
.st-key-voice_input_node [data-testid="stAudioInputWaveSurfer"] {

    display: none;
}

.st-key-voice_input div > div > div:not(:nth-child(2)),
.st-key-voice_input_node div > div > div:not(:nth-child(2)) {
    display: none;
}

.st-key-voice_input > div > div > div > span:nth-child(2),
.st-key-voice_input_node > div > div > div > span:nth-child(2) {
    display: none;
}

[aria-label="dialog"]:has(.st-key-code_editor) {
    width: 90% !important;
}

.stMainBlockContainer * {
    gap: 0.4rem !important;
}

.stMainBlockContainer {
    background-color: #F0F2F6;
}
.st-key-right-panel {
    height: calc(100vh - 75px) !important;
    margin-top: 2.25rem;
    padding: 0rem;
    padding-top: 0.5rem !important;
    background-color: #F0F2F6 !important;
}


div[data-testid="stVerticalBlock"]:has( > div > div > div.st-key-right-panel) {
    overflow-y: scroll !important;
    overflow-x: hidden !important;

div:has(> .st-key-right-panel) {
    display: block !important;
} 


div:has(> div > .st-key-right-panel) {
    background-color: #F0F2F6;
    padding: 0rem !important;
    border: none !important;
}

.st-key-right-panel img {
/*    max-width: 200px !important; */
}

.st-key-right_panel_width {
    padding-right: 0rem !important;
    padding-left: 0rem !important;
}
.st-key-right_panel_width * {
    line-height: 0.5;
    min-height: 0rem;
    padding: 0.1rem;
    border:none;
}

.st-key-node_header .stColumn {
    width: fit-content !important;
    flex: unset !important;
}

.react-json-view {
    font-size: 12px !important;
}

.st-key-right-panel-size-button .stButton * {
    font-size: 16px !important;
}

.st-key-lock button[data-testid="stBaseButton-segmented_control"],
.st-key-lock button[data-testid="stBaseButton-segmented_controlActive"] {
    padding: 4px;
}

"""


# # @st.cache_resource
# def custom_css():
#     script_path = os.path.abspath(__file__)
#     script_dir = os.path.dirname(script_path)
#     filename = "special.css"
#     file_path = os.path.join(script_dir, filename)
#     with open(file_path, "r") as file:
#         return file.read()


def st_init(page_config=True):
    if page_config:
        st.set_page_config(layout="wide", page_icon=":material/account_tree:")

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    if "init" not in st.session_state:
        import uuid

        from flowco.util.config import config

        st.session_state.init = True
        st.session_state.last_sequence_number = -1

        st.session_state.nonce = uuid.uuid4().hex

        st.session_state.selected_node = None

        st.session_state.ama_responding = False

        st.session_state.trigger_build_toggle = None
        st.session_state.builder = None
        st.session_state.builder_progress = 0.0

        st.session_state.force_update = False
        st.session_state.clear_graph = False
        st.session_state.zoom = None

        st.session_state.abstraction_level = config.abstraction_level

        st.session_state.wide_right_panel = False
        st.session_state.pinned_nodes = []

        st.session_state.current_page = None

        st.session_state.edited_node = None
        st.session_state.edit_assistant = None
        st.session_state.edit_assistant_history = None

        st.session_state.generate = False
        st.session_state.copy_generated = False
        st.session_state.chat_command = None

        st.session_state.ama = None

        st.session_state.code_editor = None

        st.session_state.just_created_project = False

        st.session_state.make_suggestions = False
