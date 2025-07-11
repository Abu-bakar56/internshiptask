import os
from datetime import datetime
from typing import List, Dict, Any, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import gradio as gr
import requests
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

# load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME")
CHAT_COLLECTION_NAME = os.getenv("CHAT_COLLECTION_NAME")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
vector_collection = client[DB_NAME][VECTOR_COLLECTION_NAME]
chat_collection = client[DB_NAME][CHAT_COLLECTION_NAME]

llm = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=0.2)

retriever_obj = None

DOCUMENT_PATH = "FY24_Q1_Consolidated_Financial_Statements.pdf"

class AgentState(TypedDict):
    messages: List[AnyMessage]
    session_id: str

def openai_embedding():
    """Initialize HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def document_loader(file_path):
    """Load PDF document."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {file_path}")
    return documents

def text_splitter(data):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(data)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def vector_database(chunks):
    """Create or reuse vector database from document chunks."""
    global retriever_obj
    
    if vector_collection.count_documents({}) > 0:
        print("Reusing existing vector database.")
        embedding = openai_embedding()
        vectordb = MongoDBAtlasVectorSearch(
            collection=vector_collection,
            embedding=embedding,
            index_name=INDEX_NAME
        )
    else:
        print("Creating new vector database.")
        vector_collection.delete_many({})  
        embedding = openai_embedding()
        vectordb = MongoDBAtlasVectorSearch.from_documents(
            documents=chunks,
            embedding=embedding,
            collection=vector_collection,
            index_name=INDEX_NAME
        )
    print(f"Vector DB has {vector_collection.count_documents({})} documents")
    retriever_obj = vectordb.as_retriever(search_kwargs={"k": 10})
    return vectordb

def process_document():
    """Process the provided document and initialize retriever."""
    global retriever_obj
    try:
        splits = document_loader(DOCUMENT_PATH)
        chunks = text_splitter(splits)
        vectordb = vector_database(chunks)
        print(f"Retriever initialized. Sample chunk: {chunks[0].page_content[:80]}...")
        return True
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return False

def save_chat_history(query, response, session_id="default"):
    """Save chat history to MongoDB."""
    chat_collection.insert_one({
        "session_id": session_id,
        "query": query,
        "response": response,
        "timestamp": datetime.utcnow()
    })

def retrieve_chat_history(session_id="default", limit=3):
    """Retrieve recent chat history."""
    history = chat_collection.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
    return [(entry["query"], entry["response"]) for entry in history]

@tool
def serper_search(query: str) -> str:
    """Searches the web using the Serper API with timezone-aware date queries."""
    try:
        if "date" in query.lower():
            query = f"{query} in PKT"
            local_date = datetime.now().strftime("%Y-%m-%d")
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        response = requests.post(url, json={"q": query}, headers=headers)
        data = response.json()
        snippets = [item.get("snippet", "") for item in data.get("organic", [])]
        result = "\n".join(snippets[:3]) or "No results found."
       
        if "date" in query.lower():
            if local_date not in result:
                return f"Local date (PKT): {local_date}"
        return result
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def document_retriever_tool(query: str, session_id: str = "default") -> str:
    """Retrieves relevant chunks from the document or falls back to Serper search."""
    global retriever_obj
    if not retriever_obj:
        print("No retriever initialized, falling back to search.")
        return serper_search.invoke({"query": query})

    try:
        base_docs = retriever_obj.get_relevant_documents(query, search_kwargs={"k": 10})
        print(f"Query: {query}")
        if not base_docs:
            print("No relevant documents found, falling back to Serper search.")
            return serper_search.invoke({"query": query})

        print(f"Retrieved {len(base_docs)} documents")
        for i, doc in enumerate(base_docs):
            print(f"Document {i+1}: {doc.page_content[:100]}...")

        context = "\n".join([doc.page_content for doc in base_docs])
        history = retrieve_chat_history(session_id)
        history_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])

        prompt_template = PromptTemplate(
            template="""You are an assistant. Answer using the context from the provided financial document and chat history. If the context is insufficient, indicate that the information is not in the document.

Context:
{context}

History:
{history_context}

Question:
{question}

Answer:""",
            input_variables=["context", "history_context", "question"]
        )

        chain = prompt_template | llm
        result = chain.invoke({"context": context, "history_context": history_context, "question": query})
        return result.content
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return f"Error: {str(e)}"

def create_react_agent():
    """Create ReAct agent with tools."""
    tools = [serper_search, document_retriever_tool]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [response], "session_id": state["session_id"]}

    def tool_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        session_id = state["session_id"]
        last = messages[-1]
        results = []
        for call in getattr(last, "tool_calls", []):
            for tool in tools:
                if tool.name == call["name"]:
                    args = call["args"]
                    if call["name"] == "document_retriever_tool":
                        res = tool.invoke({"query": args["query"], "session_id": session_id})
                    else:
                        res = tool.invoke(args)
                    results.append(ToolMessage(content=res, name=call["name"], tool_call_id=call["id"]))
        return {"messages": messages + results, "session_id": session_id}

    def should_continue(state: AgentState) -> str:
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge("agent", "tools")
    graph.add_conditional_edges("tools", should_continue, {"tools": "agent", END: END})
    graph.set_entry_point("agent")
    return graph.compile()

def rag_react_agent(query, session_id="default"):
    """Main function to process query using document or Serper search."""
    global retriever_obj
    try:
        if not retriever_obj:
            # if not process_document():
            #     return "File processing failed."

                 embedding = openai_embedding()
                 vectordb = MongoDBAtlasVectorSearch(
        collection=vector_collection,
        embedding=embedding,
        index_name=INDEX_NAME
    )
                 retriever_obj = vectordb.as_retriever(search_kwargs={"k": 10})

        
        if not query.strip():
            return "Enter a valid query."

        agent = create_react_agent()
        initial = HumanMessage(content=f"""Use tools to answer:
# - document_retriever_tool for document-based queries
# - serper_search for general queries
Question: {query}""")
        
        result = agent.invoke({"messages": [initial], "session_id": session_id})
        final = result["messages"][-1].content

        save_chat_history(query, final, session_id)
        return final
    except Exception as e:
        print(f"Error in rag_react_agent: {str(e)}")
        return f"Error: {str(e)}"

process_document()

def chat_interface_fn(message, history, session_id):
    """Wrapper function for Gradio ChatInterface."""
    return rag_react_agent(message, session_id)

with gr.Blocks(theme=gr.themes.Soft()) as rag_app:
    gr.Markdown("# RAG AND WEB ChatBOT")
    session_input = gr.Textbox(label="Session ID", value="default", placeholder="Enter session ID")
    chat_interface = gr.ChatInterface(
        fn=chat_interface_fn,
        additional_inputs=[session_input],
        title=None,
        chatbot=gr.Chatbot(height=400),
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    rag_app.launch()