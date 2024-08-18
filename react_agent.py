import uuid
from typing import Annotated, List, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    error: str


class ReActAgent:
    def __init__(self, llm, retriever, web_search_tool):
        self.llm = llm
        self.retriever = retriever
        self.web_search_tool = web_search_tool
        self.tools = self._create_tools()
        self.graph = self._create_graph()

    def _create_tools(self):
        @tool
        def retrieve_documents(query: str) -> list:
            """Retrieve documents from the vector store based on the query."""
            return self.retriever.invoke(query)

        @tool
        def grade_document_retrieval(step_by_step_reasoning: str, score: int) -> str:
            """Grade the relevance of retrieved documents."""
            if score == 1:
                return "Docs are relevant. Generate the answer to the question."
            return "Docs are not relevant. Use web search to find more documents."

        @tool
        def web_search(query: str) -> list[Document]:
            """Run web search on the question."""
            web_results = self.web_search_tool.invoke({"query": query})
            return [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]

        @tool
        def generate_answer(answer: str) -> str:
            """Generate the final answer."""
            return f"Here is the answer to the user question: {answer}"

        return [retrieve_documents, grade_document_retrieval, web_search, generate_answer]

    def _create_assistant(self) -> Runnable:
        primary_assistant_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant tasked with answering user questions using the provided vector store. "
             "Use the provided vector store to retrieve documents. Then grade them to ensure they are relevant before answering the question."),
            ("placeholder", "{messages}"),
        ])
        assistant_runnable = primary_assistant_prompt | self.llm.bind_tools(self.tools)
        return RunnableLambda(assistant_runnable)

    def _create_graph(self):
        builder = StateGraph(State)
        builder.add_node("assistant", self._create_assistant())
        builder.add_node("tools", self._create_tool_node())
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def _create_tool_node(self):
        return ToolNode(self.tools).with_fallbacks(
            [RunnableLambda(self._handle_tool_error)],
            exception_key="error"
        )

    @staticmethod
    def _handle_tool_error(state: State) -> Dict[str, List[ToolMessage]]:
        error = state.get("error", "Unknown error occurred.")
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

    def predict(self, question: str) -> Dict[str, Any]:
        """Predict the answer to a given question."""
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        messages = self.graph.invoke({"messages": ("user", question)}, config)
        return {
            "response": messages["messages"][-1].content,
            "messages": messages
        }

    @staticmethod
    def find_tool_calls(messages: Dict[str, List[Any]]) -> List[str]:
        """Find all tool calls in the messages returned from the ReAct agent."""
        return [
            tc["name"]
            for m in messages["messages"]
            for tc in getattr(m, "tool_calls", [])
        ]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}