import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.chat_models import ChatAnthropic
from typing import TypedDict, Annotated, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, END, START
import gseapy as gp
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
import json, os
import requests
import pandas as pd
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ConversationBufferMemory
import uuid
from typing import Literal, Dict, List
memory = MemorySaver()
# Set page configuration
st.set_page_config(
    page_title="Gene Analysis Chatbot",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Your existing State and LLM initialization
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",
                    anthropic_api_key='')

# EnrichR components
class EnrichrInput(BaseModel):
    input_data: List[str] = Field(desc="Input data")

def enrichr_api(input_data: List[str]) -> dict:
    """
    Perform enrichment analysis using the Enrichr API.
  
    Args:
        input_data (List[str]): Input data with a list of gene symbols.

    Returns:
        dict: Enrichment results or error details.
    """
    gene_list = input_data
    try:
        enr = gp.enrichr(
            gene_list=gene_list, 
            gene_sets='KEGG_2016', 
            outdir='test/enrichr_kegg_tumor'
        )
        
        # Extract top results as JSON
        enrichment_results = enr.res2d.head().to_dict(orient='records')
        return {"enrichment_results": enrichment_results}
    except Exception as e:
        return {"error": str(e)}

# Tool setup
enrichr_tool = Tool.from_function(
    name="enrichr_api",
    func=enrichr_api,
    description="Perform enrichment analysis using the Enrichr API. Input: gene list as a list of strings. Output: Enrichment results.",
    args_schema=EnrichrInput
)

# Bind tool to LLM
llm_with_tools = llm.bind_tools([enrichr_tool])
import json
from langchain_core.messages import ToolMessage
class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}



tool_node = ToolNode([enrichr_api])

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def should_fallback(state: State) -> Literal["agent", "remove_failed_tool_call_attempt"]:
    messages = state["messages"]
    failed_tool_messages = [
        msg for msg in messages
        if isinstance(msg, ToolMessage) and msg.additional_kwargs.get("error") is not None
    ]
    return "remove_failed_tool_call_attempt" if failed_tool_messages else "agent"

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def remove_failed_tool_call_attempt(state: MessagesState):
    messages = state["messages"]
    # Remove all messages from the most recent
    # instance of AIMessage onwards.
    last_ai_message_index = next(
        i
        for i, msg in reversed(list(enumerate(messages)))
        if isinstance(msg, AIMessage)
    )
    messages_to_remove = messages[last_ai_message_index:]
    return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}


# Fallback to a better model if a tool call fails
def call_fallback_model(state: MessagesState):
    messages = state["messages"]
    response = better_model_with_tools.invoke(messages)
    return {"messages": [response]}



workflow = StateGraph(MessagesState)

########
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("remove_failed_tool_call_attempt", remove_failed_tool_call_attempt)
workflow.add_node("fallback_agent", call_fallback_model)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", should_fallback)
workflow.add_edge("remove_failed_tool_call_attempt", "fallback_agent")
workflow.add_edge("fallback_agent", "tools")

app = workflow.compile()

def main():
    st.title("🧬 BioInformatics Agent: Pathway Explorer")
    
    with st.sidebar:
        st.markdown("## About")
        st.markdown("This intelligent agent leverages bioinformatics tools to analyze gene pathways and molecular signatures.")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your gene list or analysis query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing genes..."):
                try:
                    # Use your workflow
                    result = app.invoke({
                        "messages": [{"role": "user", "content": prompt}]
                    })
                    response = result["messages"][-1].content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()