import json
import os
from dataclasses import dataclass
from typing import List, Union
from dotenv import load_dotenv
import pandas as pd
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import UserPromptPart, ModelRequest
import logfire
import concurrent
import logging
import logging.config
import yaml
from supabase import create_client, Client
import google.generativeai as genai
import etl_old
from streamlit.runtime.uploaded_file_manager import UploadedFile
import PyPDF2
import docx
import faiss
import numpy as np


load_dotenv()

logfire.configure()
main_logger = logging.getLogger('main')

executor = concurrent.futures.ThreadPoolExecutor(max_workers=20, thread_name_prefix=__name__)

model = GeminiModel('gemini-2.0-flash-exp')
# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# openai_client = None
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class AttachmentProcessor:
    def __init__(self, attachment: UploadedFile):
        self.attachment = attachment
        self.index = faiss.IndexFlatIP(1536)
        self.transformed_chunks = []

    async def parse_pdf(self):
        text = ""
        pdf_reader = PyPDF2.PdfReader(self.attachment)
        # TODO 1: extract image
        # TODO 2: use llamaparse to parse pdf
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"

        self.transformed_chunks = await etl_old.transform_text_doc(url=None, source_name=self.attachment.name, text=text, build_index=True, index=self.index)
    
    async def parse_txt(self):
        text = self.attachment.read().decode()
        self.transformed_chunks = await etl_old.transform_text_doc(url=None, source_name=self.attachment.name, text=text, build_index=True, index=self.index)

    async def parse_csv(self):
        # df = pd.read_csv(attachment)
        # text = df.to_string()
        text = self.attachment.read().decode()
        self.transformed_chunks = await etl_old.transform_text_doc(url=None, source_name=self.attachment.name, text=text, build_index=True, index=self.index)

    async def parse_doc(self):
        doc = docx.Document(self.attachment)
        text = ""
        for para in doc.paragraphs:
            text += para.text
        self.transformed_chunks = await etl_old.transform_text_doc(url=None, source_name=self.attachment.name, text=text, build_index=True, index=self.index)

    async def parse_image(self):
        pass


async def match_query_embedding(processor: AttachmentProcessor, prompt: str) -> str:
    top_k = 10
    query_embedding = np.array(await etl_old.get_embeddings(prompt, is_document=False), dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    _, I = processor.index.search(query_embedding, top_k)
    print(f"Top {top_k} matches: {I}")
    matched_chunks = [processor.transformed_chunks[i].content for i in I[0] if i >= 0]
    return "\n\n".join(matched_chunks)


async def process_attachment(attachment: UploadedFile) -> AttachmentProcessor:
    processor = AttachmentProcessor(attachment)
    print(attachment.type)
    if 'png' in attachment.type or\
        'jpeg' in attachment.type or\
        'jpg' in attachment.type :
        pass
    elif 'pdf' in attachment.type:
        #TODO: also extract images from pdf and process them 
        await processor.parse_pdf()
    elif 'csv' in attachment.type:
        await processor.parse_csv()
    elif 'txt' in attachment.type or 'text' in attachment.type:
        await processor.parse_txt()
    elif 'doc' in attachment.type or 'docx' in attachment.type:
        await processor.parse_doc()
    return processor


@dataclass
class RetrievalAgentDeps:
    supabase: Client = None
    attachment_processors: List[AttachmentProcessor] = None

retrieval_agent = Agent(
    model=model,
    system_prompt=(
        'You are an AI assistant that retrieves information either from the attached documents or by querying the database.'
        'Always call at least one tool before responding to the user.'
        'If the user has attached documents and the documents are relevant to the user prompt, use the `query_attached_file` tool.'
        'If the user prompt is not related to the attached documents, use the `query_database` tool.'
        'Answer any question to the best of your abilities, and always return a reasonable response.'
        'Return data as an array of json objects.'
    ),
    deps_type=RetrievalAgentDeps,
    result_type=str,
    retries=3
)


helper_agent = Agent(
    model=model, 
    system_prompt=(
        'You are an AI assistant your task is to assist the user with general questions and simple tasks.'
        'Only do or answer what you are asked for.'
    ),
    retries=3
)


validation_agent = Agent(
    model=model,
    system_prompt=(
        'You are an assistant that validates the final output of a given prompt.'
        'Valid means that the output is relevant to the prompt'
        'Just indicate True for valid results and False for invalid results.'
    ),
    result_type=bool,
    retries=3,
)


@retrieval_agent.tool(retries=3)
async def query_attached_file(ctx: RunContext[RetrievalAgentDeps], user_prompt: str) -> str:
    """ Process the attached document(s) and return the relevant content based on the user's prompt.

    Args:
        ctx: The context including the Supabase client
        user_prompt: The user's prompt
    Returns:
        A list of DocumentProcessor objects for the attached document(s).
    """
    print(f"Document processors: {ctx.deps.attachment_processors}")
    print(f"User prompt: {user_prompt}")
    if ctx.deps.attachment_processors:
        result = ""
        for processor in ctx.deps.attachment_processors:
            result += await match_query_embedding(processor, user_prompt)                
        print(f"Result: {result}")
        return result
    return "No document attached."


@retrieval_agent.tool(retries=3)
async def query_database(ctx: RunContext[RetrievalAgentDeps]) -> str:
    """ Determine the most relevant source_name(s), based on the user's prompt.

    Args:
        ctx: The context including the Supabase client

    Returns:
        A list of strings which could be the potential source_names for the user prompt.
    """
    user_prompt = None
    for i in range(len(ctx.messages)-1, -1, -1):
        if isinstance(ctx.messages[i], ModelRequest):
            user_prompt = ctx.messages[i].parts[-1].content
            break
        
    # print(f"Document processors: {ctx.deps.attachment_processors}")
    # if ctx.deps.attachment_processors:
    #     result = (await match_query_embedding(ctx.deps.attachment_processors[0], user_prompt)).data
    #     return result
    
    # try:
    # get all the unique source_name's from the table
    possible_source_names = ctx.deps.supabase.table('agentic_rag')\
        .select('source_name')\
        .execute().data
    unique_src_names = set()
    for d in possible_source_names:
        unique_src_names.add(d["source_name"])
    
    engineered_prompt = (
        f'Help me determine the most relevant source_name from a prompt, i.e. the subject of the prompt.'
        f'Choose from the following list of source_name\'s: `{list(unique_src_names)}`.'
        f'You can choose multiple source_name\'s if you think they are relevant.'
        f'If none of the source_name\'s are relevant, return `None`.'
        f'Return a list of the choices made, strictly use double quotes to wrap each source_name.'
        f'Here is the prompt: {user_prompt}'
    )

    result = await helper_agent.run(engineered_prompt)
    if 'None' in result.data:
        return ""
    result = result.data.strip()
    source_names = json.loads(result[result.find("["):result.rfind("]")+1])
    print(f"Most relevant source names {type(source_names)}: {source_names}")
    
    # ------------------------------------------------------------------------------------------------------------
    urls = ctx.deps.supabase.from_('agentic_rag')\
        .select('url')\
        .in_('source_name', source_names)\
        .execute().data

    unique_urls = set()
    for d in urls:
        unique_urls.add(d["url"])
    unique_urls = list(unique_urls)

    # try:
    user_query = ctx.messages[-1].parts[-1].content
    print(f"User query: {user_query}")
    query_embedding = await etl_old.get_embeddings(user_query, is_document=False)
    print(f"Source names: {source_names}")
    print(f"URLs: {unique_urls}")
    # get the most relevant content from the table
    relevant_content = ctx.deps.supabase.rpc(
        'match_agentic_rag',
        {
            'query_embedding': query_embedding,
            'source_names': source_names,
            'urls': unique_urls,
            'match_count': 10
        }
    ).execute().data

    print(f"Relevant content: {relevant_content}")
    if not relevant_content:
        return "No relevant content found."
    
    new_content = ''
    for content in relevant_content:
        new_content += f"#{content['title']}\n{content['content']}\n\n"
    print(f"NEW CONTENT: {new_content}")
    return new_content
    # except Exception as e:
    #     raise ModelRetry(f"Error retrieving the most relevant content: {e}")
    
    # return source_names
    # except Exception as e:
    #     raise ModelRetry(f"Error determining the most relevant source_name(s): {e}")


# @retrieval_agent.tool(retries=3)
# async def retrieve_relevant_content(ctx: RunContext[RetrievalAgentDeps], source_names: List[str]) -> str:
#     '''Retrieve the most relevant document based on the source_names, from the agent's knowledge base.

#     Args:
#         ctx: The context including the Supabase client
#         source_names: A list of strings which could be the potential source_names for the user prompt.

#     Returns:
#         A string containing the content fetched using RAG.
#     '''
#     urls = ctx.deps.supabase.from_('agentic_rag')\
#         .select('url')\
#         .in_('source_name', source_names)\
#         .execute().data
#     print(urls)
#     unique_urls = set()
#     for d in urls:
#         unique_urls.add(d["url"])
#     print(f"URLs: {unique_urls}")

#     # try:
#     user_query = ctx.messages[-1].parts[-1].content
#     print(f"User query: {user_query}")
#     query_embedding = await etl_old.get_embeddings(user_query)
#     # get the most relevant content from the table
#     relevant_content = ctx.deps.supabase.rpc(
#         'match_agentic_rag',
#         {
#             'query_embedding': query_embedding,
#             'source_names': source_names,
#             'urls': urls,
#             'match_count': 10
#         }
#     ).execute().data
#     print(f"Relevant content: {relevant_content}")
#     if not relevant_content:
#         return "No relevant content found."

#     new_content = ''
#     for content in relevant_content:
#         new_content += f"#{content['title']}\n{content['content']}\n\n"
#     return new_content
#     # except Exception as e:
#     #     raise ModelRetry(f"Error retrieving the most relevant content: {e}")


# @retrieval_agent.result_validator
async def validate_result(ctx: RunContext[RetrievalAgentDeps], final_output: str):
    """ The validator for all of the retrieval_agent's final result.
        Always called before returning the response to user.
    """
    engineered_prompt = (
        f"Prompt: {ctx.prompt.strip()}, "
        f"Final Output: {final_output.strip()}"
    )
    print(f"Validating: {engineered_prompt}")
    result = await validation_agent.run(engineered_prompt)
    if result.data:
        return final_output
    raise ModelRetry("The result is invalid.")


if __name__ == "__main__":
    history = None
    while True:
        prompt = input("Enter a prompt: ")
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        result = retrieval_agent.run_sync(prompt, deps=RetrievalAgentDeps(supabase, []), message_history=history)
        history = result.all_messages()
        print(f"HISTORY: {history}")
        print(f"RESULT: {result.data}")

