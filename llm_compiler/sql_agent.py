from typing import Annotated, Literal, List, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from llm_compiler.utils import create_tool_node_with_fallback
from llm_compiler.planner import create_planner, stream_plan
from llm_compiler.scheduler import schedule_tasks
from llm_compiler.output_parser import LLMCompilerPlanParser

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class SubmitFinalAnswer(BaseModel):
    final_answer: str = Field(..., description="The final answer to the user")

@tool
def db_query_tool(query: str) -> str:
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result

def create_sql_agent(llm: ChatOpenAI):
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

    query_check_system = """You are a SQL expert with a strong attention to detail.
    Double check the SQLite query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    You will call the appropriate tool to execute the query after running this check."""

    query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
    )
    query_check = query_check_prompt | llm.bind_tools([db_query_tool], tool_choice="required")

    query_gen_system = """You are a SQL expert with a strong attention to detail.

    Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

    DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

    When generating the query:

    Output the SQL query that answers the input question without a tool call.

    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.

    If you get an error while executing a query, rewrite the query and try again.

    If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
    NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

    If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
    query_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", query_gen_system), ("placeholder", "{messages}")]
    )
    query_gen = query_gen_prompt | llm.bind_tools([SubmitFinalAnswer])

    def query_gen_node(state: State):
        message = query_gen.invoke(state)
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            tool_messages = []
        return {"messages": [message] + tool_messages}

    workflow = StateGraph(State)
    workflow.add_node("first_tool_call", lambda state: {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    })
    workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool]))
    workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
    workflow.add_node("model_get_schema", lambda state: {
        "messages": [llm.bind_tools([get_schema_tool]).invoke(state["messages"])],
    })
    workflow.add_node("query_gen", query_gen_node)
    workflow.add_node("correct_query", lambda state: {
        "messages": [query_check.invoke({"messages": [state["messages"][-1]]})]
    })
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

    def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
        messages = state["messages"]
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return END
        if last_message.content.startswith("Error:"):
            return "query_gen"
        else:
            return "correct_query"

    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "query_gen")
    workflow.add_conditional_edges(
        "query_gen",
        should_continue,
    )
    workflow.add_edge("correct_query", "execute_query")
    workflow.add_edge("execute_query", "query_gen")

    return workflow.compile()
