import asyncio
import json
import os
from dataclasses import dataclass
from typing import List, Literal, TypedDict
from dotenv import load_dotenv
import pandas as pd
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
import logfire
import logging
import logging.config
import yaml
from supabase import create_client, Client
import google.generativeai as genai
from retrieval_agent_old import retrieval_agent, helper_agent, RetrievalAgentDeps, process_attachment, AttachmentProcessor
import streamlit as st
import etl



load_dotenv()

logfire.configure()
with open("config/logging.yml", "r") as logging_config_file:
    logging.config.dictConfig(yaml.load(logging_config_file, Loader=yaml.FullLoader))

main_logger = logging.getLogger('main')


supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def invoke_agent(prompt: str, attachment_processors: List[AttachmentProcessor] = None):
    print(f"INVOKE AGENT PROMPT: {prompt}")
    async with retrieval_agent.run_stream(prompt, deps=RetrievalAgentDeps(supabase, attachment_processors), message_history=st.session_state.message_history) as result:
        partial_text = ""
        structured_data = ""
        message_placeholder = st.empty()
        async for chunk in result.stream_text(delta=True):
            index = chunk.find('```')
            if index != -1:
                if structured_data == "":
                    structured_data += chunk[index:]
                    partial_text += chunk[:index]
                else:
                    structured_data += chunk[:index+1]
                    partial_text += chunk[index+1:]
                    print(f"Structured data: {structured_data}")
                    st.table(pd.dataframe(json.loads(structured_data[structured_data.find('['):structured_data.rfind(']')+1])))
                    # message_placeholder.markdown(structured_data)
                    structured_data = ""
            else:
                partial_text += chunk                
            message_placeholder.markdown(partial_text)

        print(f"Streamlit received data: {partial_text}")

        # Add the final response to the messages
        st.session_state.message_history.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.title("Dixit's agentic RAG")
    st.write("Ask me a question about {{something}}. You can provide a url, text document or an image as additional information source.")

    # Initialize chat history in session state if not present
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if "attachments_processed" not in st.session_state:
        st.session_state.attachments_processed = {}

    for msg in st.session_state.message_history:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # prompt = st.chat_input("Ask me...")
    prompt = st.chat_input("Ask me...")
    attachments = st.file_uploader("Upload a file.", type=("txt", "csv", "md", "pdf", "jpg", "jpeg", "png"), accept_multiple_files=True)
    
    if attachments:
        once = True
        for attachment in attachments:
            if attachment.file_id not in st.session_state.attachments_processed:
                with st.spinner("Processing attached document..."):
                    processor = await process_attachment(attachment=attachment)
                    st.session_state.attachments_processed[attachment.file_id] = processor
                if once:
                    st.success("Successfully processed attached document.")
                    once = False

    
    for file_id in list(st.session_state.attachments_processed.keys()):
        found = False
        for attachment in attachments:
            if attachment.file_id == file_id:
                found = True
        if not found:
            st.session_state.attachments_processed.pop(file_id)


    if prompt:
        st.session_state.message_history.append(
            ModelRequest(parts=[UserPromptPart(content=prompt)])
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            print("INVOKING AGENT")
            with st.spinner("Thinking..."):
                await invoke_agent(prompt, list(st.session_state.attachments_processed.values()))
            print("INVOKE CALL RETURNED")
    # print(f"STATE: {st.session_state}")


if __name__ == "__main__":
    # asyncio.run(main())
    if not hasattr(st.session_state, "loop"):
        st.session_state.loop = asyncio.new_event_loop()
    st.session_state.loop.run_until_complete(main())

