# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_openai import (
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_community.document_loaders import (
    YoutubeLoader, 
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotPromptTemplate,
    StringPromptTemplate,
)
from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessageGraph,
)
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableSequence,
    RunnableBranch,
    RunnableConfig,
    Runnable,
    RunnableSerializable,
    RunnableMap,
    RunnableWithFallbacks,
    chain,
)
from typing import (
    List, Dict, Tuple, Set, Optional, Union, Any, Callable, Iterable, Sequence, Mapping, 
    TypeVar, Generic, NewType, cast, overload, final, Protocol, runtime_checkable, 
    TypedDict, Literal, ClassVar, NoReturn, Text, IO, ContextManager, AsyncIterable, 
    Coroutine, Generator, NamedTuple, Annotated
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    ChatMessage,
    MessageLikeRepresentation,
)
from langgraph.checkpoint.memory import (
    InMemorySaver,
    MemorySaver,
)     
from langgraph.graph.message import (
    add_messages,
    MessageGraph,
) 
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    XMLOutputParser,
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    BaseOutputParser,
)
from langchain_core.language_models.llms import (
    LLM,
    BaseLLM,
)
# import faiss
from langchain_community.vectorstores import (
    FAISS,
    Chroma,
    Pinecone,
    VectorStore,
    Qdrant,
    Weaviate,
    Redis,
    Milvus,
    ElasticsearchStore,
)
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
    PubmedQueryRun,
    GoogleSearchRun,
    Tool,
    StructuredTool,
    BaseTool,
)
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

#----------------------------------------------------------------------------------#
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
llm = ChatOpenAI(model="meta-llama/llama-3.3-70b-instruct")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#Youtube loader
video_url = "https://www.youtube.com/watch?v=msHyYioAyNE&list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&index=2"
loader = YoutubeLoader.from_youtube_url(video_url, language="en")
docs1 = loader.load()
# docs2 = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.split_documents(docs1 + docs2)
chunks = splitter.split_documents(docs1)
vector_store = FAISS.from_documents(chunks, embeddings)
# print(vector_store.index.reconstruct_n(0, 2))
# print(vector_store.docstore._dict)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        {context}
        Question: {question}
    """,
    input_variables = ['context', 'question']
)
parser = StrOutputParser()
chain = prompt | llm | parser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text
query = "summerize the document"
context = format_docs(retriever.invoke(query))
response = chain.invoke({"context": context,"question": query})
# print(response)

#------------------------------------------------------------------------------------#
# Define custom tools using the @tool decorator
from langchain_core.tools import tool
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when the user asks about weather."""
    # This is a mock weather function. In production, you'd call a real weather API like OpenWeatherMap
    import random
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "stormy"]
    temp = random.randint(15, 35)
    condition = random.choice(weather_conditions)
    return f"The weather in {city} is {condition} with a temperature of {temp}°C."

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input should be a valid Python mathematical expression like '2+2' or '10*5'."""
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# Initialize search tool
search = DuckDuckGoSearchRun()

# Bind all tools to the LLM
tools = [search, get_weather, calculator, get_current_time]
llm_bind_tools = llm.bind_tools(tools)

user_query = 'what is the current time'
system_message = SystemMessage(content="""You are a helpful assistant with access to the following tools:
- Web search (for recent news and information)
- Weather lookup (for current weather in any city)
- Calculator (for mathematical calculations)
- Current time (for getting current date and time)

ONLY use tools when necessary:
- Use search for recent events or news
- Use weather tool when asked about weather
- Use calculator for math problems
- Use time tool when asked about current time

For general knowledge questions you can answer directly without tools.""")

messages = [system_message, HumanMessage(content=user_query)]
result = llm_bind_tools.invoke(messages)
print("Initial LLM Response:")
print(result)

if result.tool_calls:
    # Create a tool map for easy lookup
    tool_map = {tool.name: tool for tool in tools}
    
    # Add the AI message with tool calls once
    messages.append(result)
    
    for tool_call in result.tool_calls:
        tool_name = tool_call['name']
        print(f"\nExecuting tool: {tool_name}")
        print(f"With arguments: {tool_call['args']}")
        
        # Get the appropriate tool and execute it
        if tool_name in tool_map:
            tool_result = tool_map[tool_name].invoke(tool_call['args'])
        else:
            tool_result = f"Error: Tool {tool_name} not found"
        
        print(f"Tool result: {tool_result}\n")
        
        # Add tool response to messages
        messages.append(ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call['id']
        ))
    
    final_result = llm_bind_tools.invoke(messages)
    print("Final LLM Response:")
    print(final_result.content)
else:
    print("Direct Response:")
    print(result)