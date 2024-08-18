import logging
import uuid
from typing import Annotated, List, Dict, Any

from langchain.pydantic_v1 import BaseModel, Field, ValidationError
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    error: str


class QuestionInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class ReActAgent:
    def __init__(self, llm, retriever, web_search_tool, config: Dict[str, Any] = None):
        self.llm = llm
        self.retriever = retriever
        self.web_search_tool = web_search_tool
        self.config = config or {}
        self.tools = self._create_tools()
        self.graph = self._create_graph()
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_response_time": 0,
        }

    def _create_tools(self):
        @tool
        def retrieve_documents(query: str) -> list:
            """Retrieve documents from the vector store based on the query."""
            logger.info(f"Retrieving documents for query: {query}")
            return self.retriever.invoke(query)

        @tool
        def grade_document_retrieval(step_by_step_reasoning: str, score: int) -> str:
            """Grade the relevance of retrieved documents."""
            logger.info(f"Grading document retrieval. Score: {score}")
            if score == 1:
                return "Docs are relevant. Generate the answer to the question."
            return "Docs are not relevant. Use web search to find more documents."

        @tool
        def web_search(query: str) -> list[Document]:
            """Run web search on the question."""
            logger.info(f"Performing web search for query: {query}")
            web_results = self.web_search_tool.invoke({"query": query})
            return [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]

        @tool
        def generate_answer(answer: str) -> str:
            """Generate the final answer."""
            logger.info("Generating final answer")
            return f"Here is the answer to the user question: {answer}"

        return [retrieve_documents, grade_document_retrieval, web_search, generate_answer]

    def _create_assistant(self) -> Runnable:
        primary_assistant_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.get("system_prompt",
                                       "You are a helpful assistant tasked with answering user questions using the provided vector store. "
                                       "Use the provided vector store to retrieve documents. Then grade them to ensure they are relevant before answering the question.")),
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
        logger.error(f"Tool error occurred: {error}")
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    async def predict_async(self, question: str) -> Dict[str, Any]:
        """Asynchronously predict the answer to a given question."""
        try:
            input_data = QuestionInput(question=question)
        except ValidationError as e:
            logger.error(f"Input validation error: {e}")
            raise ValueError("Invalid input question") from e

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        messages = await self.graph.ainvoke({"messages": ("user", input_data.question)}, config)
        return {
            "response": messages["messages"][-1].content,
            "messages": messages
        }

    def predict(self, question: str) -> Dict[str, Any]:
        """Predict the answer to a given question."""
        import time
        start_time = time.time()

        try:
            input_data = QuestionInput(question=question)
        except ValidationError as e:
            logger.error(f"Input validation error: {e}")
            raise ValueError("Invalid input question") from e

        self.metrics["total_queries"] += 1

        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            messages = self.graph.invoke({"messages": ("user", input_data.question)}, config)
            self.metrics["successful_queries"] += 1
            response = {
                "response": messages["messages"][-1].content,
                "messages": messages
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            self.metrics["failed_queries"] += 1
            raise

        end_time = time.time()
        response_time = end_time - start_time
        self.metrics["total_response_time"] += response_time

        logger.info(f"Query processed in {response_time:.2f} seconds")
        return response

    @staticmethod
    def find_tool_calls(messages: Dict[str, List[Any]]) -> List[str]:
        """Find all tool calls in the messages returned from the ReAct agent."""
        return [
            tc["name"]
            for m in messages["messages"]
            for tc in getattr(m, "tool_calls", [])
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics of the agent."""
        metrics = self.metrics.copy()
        if metrics["total_queries"] > 0:
            metrics["average_response_time"] = int(
                metrics["total_response_time"] / metrics["total_queries"])  # Cast to int
        return metrics


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


# Version information
__version__ = "0.1.0"
