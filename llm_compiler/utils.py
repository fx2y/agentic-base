import logging
from typing import Any
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

@tool
def db_query_tool(query: str) -> str:
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    result = db.run_no_throw(query)
    if not result:
        logging.error(f"Query failed: {query}")
        return "Error: Query failed. Please rewrite your query and try again."
    logging.info(f"Query executed successfully: {query}")
    return result
