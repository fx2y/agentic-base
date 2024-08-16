from typing import Annotated, TypedDict

from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

from joiner import create_joiner
from math_tools import get_math_tool
from planner import create_planner, stream_plan
from utils import get_pass


class State(TypedDict):
    messages: Annotated[list, add_messages]


def main():
    get_pass("LANGCHAIN_API_KEY")
    get_pass("OPENAI_API_KEY")
    get_pass("TAVILY_API_KEY")

    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    calculate = get_math_tool(llm)
    search = TavilySearchResults(
        max_results=1,
        description='tavily_search_results_json(query="the search query") - a search engine.',
    )
    tools = [search, calculate]

    prompt = hub.pull("wfh/llm-compiler")
    planner = create_planner(llm, tools, prompt)
    joiner = create_joiner(llm)

    graph_builder = StateGraph(State)
    graph_builder.add_node("plan_and_schedule", lambda state: {
        "messages": state["messages"],
        "tasks": stream_plan(planner, state["messages"])
    })
    graph_builder.add_node("join", joiner)
    graph_builder.add_edge("plan_and_schedule", "join")

    def should_continue(state):
        messages = state["messages"]
        if isinstance(messages[-1], AIMessage):
            return END
        return "plan_and_schedule"

    graph_builder.add_conditional_edges(
        start_key="join",
        condition=should_continue,
    )
    graph_builder.add_edge(START, "plan_and_schedule")
    chain = graph_builder.compile()

    example_question = "What's the temperature in SF raised to the 3rd power?"
    for step in chain.stream({"messages": [HumanMessage(content=example_question)]}):
        print(step)
        print("---")


if __name__ == "__main__":
    main()
