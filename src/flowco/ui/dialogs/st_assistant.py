# from dataclasses import dataclass
# import json
# from queue import Queue
# from typing import List, TypeVar
# from litellm import ChatCompletionMessageToolCall
# from openai import OpenAI
# from pydantic import BaseModel, Field
# import streamlit as st
# import litellm

# from flowco.util.output import log


# class StreamlitQuestioningAssistant:
#     def __init__(
#         self,
#         system_prompt,
#         model="gpt-4o",
#         timeout=30,
#     ):

#         self._model = model
#         self._timeout = timeout
#         self._conversation = [{"role": "system", "content": system_prompt}]
#         self._question_queue: List[ChatCompletionMessageToolCall] = []

#     def add_message(self, role: str, content: str | list[dict[str, any]]):
#         self._conversation.append({"role": role, "content": content})

#     T = TypeVar("T", bound=BaseModel)

#     def query(self, prompt: str, t: T) -> T:
#         self._conversation.append({"role": "user", "content": prompt})
#         client = OpenAI()

#         completion = self._completion(t, client)

#         response_message = completion.choices[0].message
#         self._conversation.append(response_message)

#         if completion.choices[0].finish_reason == "tool_calls":
#             self._add_questions_to_queue(response_message)
#             return None
#         else:
#             # cost = litellm.completion_cost(model=self._model, messages=self._conversation, completion=response_message)
#             # add_cost(cost)

#             return response_message.parsed

#     def _completion(self, t, client: OpenAI, stream=False):
#         completion = client.beta.chat.completions.parse(
#             model=self._model,
#             messages=self._conversation,
#             tools=[
#                 {
#                     "type": "function",
#                     "function": {
#                         "name": "answer_question",
#                         "description": "Ask the user a question and get the answer.",
#                         "parameters": {
#                             "type": "object",
#                             "required": [
#                                 "phase",
#                                 "question",
#                             ],
#                             "properties": {
#                                 "phase": {
#                                     "type": "string",
#                                     "description": "The phase of the question.  Either 'Requirements', 'Algorithm', or 'Code'.",
#                                 },
#                                 "question": {
#                                     "type": "string",
#                                     "description": "The question to be answered.",
#                                 },
#                             },
#                             "additionalProperties": False,
#                         },
#                         "strict": True,
#                     },
#                 }
#             ],
#             timeout=self._timeout,
#             response_format=t,
#             stream_options={"include_usage": True},
#         )

#         return completion
#         # with completion as stream:
#         #     print("BeeP")
#         #     for event in stream:
#         #         print("BOOP", event.type)
#         #         if event.type == "content.delta":
#         #             print(event.delta, flush=True, end="")

#         # print(stream)
#         # return stream

#     def _streamed_query(self, prompt: str, user_text):
#         cost = 0

#         self._conversation.append({"role": "user", "content": prompt})

#         while True:
#             stream = self._completion(stream=True)

#             # litellm.stream_chunk_builder is broken for new GPT models
#             # that have content before calls, so...

#             # stream the response, collecting the tool_call parts separately
#             # from the content
#             try:
#                 self._broadcast("on_begin_stream")
#                 chunks = []
#                 tool_chunks = []
#                 for chunk in stream:
#                     self._log({"chunk": chunk})
#                     chunks.append(chunk)
#                     if chunk.choices[0].delta.content != None:
#                         self._broadcast(
#                             "on_stream_delta", chunk.choices[0].delta.content
#                         )
#                     else:
#                         tool_chunks.append(chunk)
#             finally:
#                 self._broadcast("on_end_stream")

#             # then compute for the part that litellm gives back.
#             completion = litellm.stream_chunk_builder(
#                 chunks, messages=self._conversation
#             )
#             cost += litellm.completion_cost(completion)

#             # add content to conversation, but if there is no content, then the message
#             # has only tool calls, and skip this step
#             response_message = completion.choices[0].message
#             if response_message.content != None:
#                 self._conversation.append(response_message.json())

#             if response_message.content != None:
#                 self._broadcast("on_response", response_message.content)

#             if completion.choices[0].finish_reason == "tool_calls":
#                 # create a message with just the tool calls, append that to the conversation, and generate the responses.
#                 tool_completion = litellm.stream_chunk_builder(
#                     tool_chunks, self._conversation
#                 )

#                 # this part wasn't counted above...
#                 cost += litellm.completion_cost(tool_completion)

#                 tool_message = tool_completion.choices[0].message

#                 tool_json = tool_message.json()

#                 # patch for litellm sometimes putting index fields in the tool calls it constructs
#                 # in stream_chunk_builder.  gpt-4-turbo-2024-04-09 can't handle those index fields, so
#                 # just remove them for the moment.
#                 for tool_call in tool_json.get("tool_calls", []):
#                     _ = tool_call.pop("index", None)

#                 tool_json["role"] = "assistant"
#                 self._conversation.append(tool_json)
#                 self._add_function_results_to_conversation(tool_message)
#             else:
#                 break

#         stats = {
#             "cost": cost,
#             "tokens": completion.usage.total_tokens,
#             "prompt_tokens": completion.usage.prompt_tokens,
#             "completion_tokens": completion.usage.completion_tokens,
#         }
#         return stats

#     def _add_questions_to_queue(self, response_message):
#         tool_calls = response_message.tool_calls
#         for tool_call in tool_calls:
#             if tool_call.function.name == "answer_question":
#                 self._question_queue.append(tool_call)

#     def peek_question(self) -> str | None:
#         if len(self._question_queue) == 0:
#             return None
#         else:
#             print(self._question_queue[0])
#             arguments = self._question_queue[0].function.parsed_arguments
#             return f"**{arguments['phase']}**: {arguments['question']}"

#     def has_qustions(self) -> bool:
#         return len(self._question_queue) > 0

#     def answer_question(self, answer: str):
#         tool_call = self._question_queue.pop(0)
#         response = {
#             "tool_call_id": tool_call.id,
#             "role": "tool",
#             "name": tool_call.function.name,
#             "content": answer,
#         }
#         self._conversation.append(response)


# class Response(BaseModel):
#     reasoning_for_requirements: str
#     requirements: List[str]
#     reasoning_for_algorithm: str
#     algorithm: str
#     reasoning_for_code: str
#     code: List[str] = Field(
#         default_factory=list,
#         description="The code to be checked for consistency with the requirements and algorithm.",
#     )


# system = """
# You will take a list of requirements, algorithm pseudocode, or code.
# * Create a list of consistency problems in the requirements.  For each one design a question to resolve it,
#     and ask the user for an answer using the `ask_question` function call.  Use the answers to create a list of
#     consistent requirements in line with the user's answers.  Ask followups as needed.
# * Modify the algorithm pseudocode to match the corrected requirements.  If the pseudocode is not consistent with
#     the requirements, create a list of inconsistencies and ask the user for answers to resolve them.  Ask followups as needed.
# * Modify the code to match the corrected pseudocode.  If the code is not consistent with the pseudocode or requirements, create a
#     list of inconsistencies and ask the user for answers to resolve them.  Ask followups as needed.
# """

# test = """
# * load_finch_result is a pandas DataFrame.
# * The DataFrame contains columns: 'species', 'Beak length, mm', 'Beak depth, mm'.
# * All rows with missing or NA values are removed from load_finch_result.
# * The data types for the columns are: 'species' as string, 'Beak length, mm' and 'Beak depth, mm' as int.
# * measurements are in tenths of mm.
# * species are encoded as ints.
# """

# test2 = """
#     {
#     "label": "Select fortis",
#     "requirements": [
#         "select_fortis_result is a pandas DataFrame.",
#         "select_fortis_result contains no missing or obviously bad values.",
#         "select_fortis_result has columns: 'species', 'Beak length, mm', 'Beak depth, mm'.",
#         "select_fortis_result has more than 5 rows.",
#         "select_fortis_result only contains rows where the species is 'fortis'."
#     ],
#     "algorithm": "1. Start with the input DataFrame 'finch_data_result'.\n2. Filter the DataFrame to include only rows where the 'species' column is equal to 'fortis'.\n3. Store the filtered DataFrame in a new variable 'select_fortis_result'.\n4. Ensure 'select_fortis_result' contains no missing or obviously bad values.\n5. Verify that 'select_fortis_result' has columns: 'species', 'Beak length, mm', 'Beak depth, mm'.\n6. Verify that 'select_fortis_result' has more than 5 rows.\n7. Verify that 'select_fortis_result' only contains rows where the species is 'fortis'.",
#     "function_parameters": [
#         {
#         "name": "finch_data_result",
#         "type": {
#             "type": "pd.DataFrame['species': str, 'Beak length, mm': float, 'Beak depth, mm': float]"
#         }
#         }
#     ],
#     "function_return_type": {
#         "type": "pd.DataFrame['species': str, 'Beak length, mm': float, 'Beak depth, mm': float]"
#     },
#     "function_computed_value": "The function returns a cleaned pandas DataFrame containing only rows where the species is 'fortis', with columns for 'Beak length, mm' and 'Beak depth, mm', and ensures there are no missing or obviously bad values and more than 5 rows.",
#     "code": [
#         "import pandas as pd",
#         "import numpy as np",
#         "import seaborn as sns",
#         "import matplotlib.pyplot as plt",
#         "from sklearn.model_selection import train_test_split",
#         "",
#         "def compute_select_fortis(finch_data_result: pd.DataFrame) -> pd.DataFrame:",
#         "    # Step 1: Filter the DataFrame to include only rows where the 'species' column is equal to 'fortis'",
#         "    select_fortis_result = finch_data_result[finch_data_result['species'] == 'fortis']",
#         "",
#         "    # Step 2: Ensure 'select_fortis_result' contains no missing or obviously bad values",
#         "    select_fortis_result = select_fortis_result.dropna()",
#         "    assert not select_fortis_result.isnull().values.any(), 'select_fortis_result contains missing values'",
#         "",
#         "    # Step 3: Verify that 'select_fortis_result' has columns: 'species', 'Beak length, mm', 'Beak depth, mm'",
#         "    required_columns = {'species', 'Beak length, mm', 'Beak depth, mm'}",
#         "    assert required_columns.issubset(select_fortis_result.columns), f'select_fortis_result does not have the required columns: {required_columns}'",
#         "",
#         "    # Step 4: Verify that 'select_fortis_result' has more than 5 rows",
#         "    assert len(select_fortis_result) > 5, 'select_fortis_result does not have more than 5 rows'",
#         "",
#         "    # Step 5: Verify that 'select_fortis_result' only contains rows where the species is 'fortis'",
#         "    assert (select_fortis_result['species'] == 'fortis').all(), 'select_fortis_result contains rows where species is not fortis'",
#         "",
#         "    return select_fortis_result"
#     ]
#     },
# """


# if "assistant" not in st.session_state:
#     st.session_state.assistant = None


# @st.fragment
# def answer():
#     response = st.session_state.response
#     if response is None:
#         if assistant.has_qustions():
#             question = assistant.peek_question()
#             if question is not None:
#                 with st.chat_message("ai"):
#                     st.markdown(question)

#                 answer = st.chat_input("Answer")
#                 if answer is not None:
#                     if answer:
#                         assistant.answer_question(answer)
#                         st.session_state.messages.append(
#                             {"role": "user", "content": answer}
#                         )
#                     st.rerun(scope="fragment")
#         else:
#             st.session_state.response = assistant.query(test2, Response)
#             st.rerun(scope="fragment")
#     else:
#         st.write(response)


# if st.button("Go"):
#     assistant = st.session_state.assistant
#     if assistant is None:
#         assistant = StreamlitQuestioningAssistant(system_prompt=system)
#         st.session_state.response: Response | None = assistant.query(test2, Response)
#     answer()


# # from openai import OpenAI
# # import streamlit as st

# # st.title("ChatGPT-like clone")

# # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # if "openai_model" not in st.session_state:
# #     st.session_state["openai_model"] = "gpt-3.5-turbo"

# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])

# # if prompt := st.chat_input("What is up?"):
# #     st.session_state.messages.append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)

# #     with st.chat_message("assistant"):
# #         stream = client.chat.completions.create(
# #             model=st.session_state["openai_model"],
# #             messages=[
# #                 {"role": m["role"], "content": m["content"]}
# #                 for m in st.session_state.messages
# #             ],
# #             stream=True,
# #         )
# #         response = st.write_stream(stream)
# #     st.session_state.messages.append({"role": "assistant", "content": response})
