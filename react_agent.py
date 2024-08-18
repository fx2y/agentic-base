import logging
import time
import uuid
from functools import lru_cache
from typing import Annotated, List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    """
    ReActAgent is a class that implements a ReAct (Retrieve, Act) agent for question answering.
    
    This agent uses a combination of document retrieval, web search, and language model
    to answer questions. It employs a graph-based workflow to manage the question-answering process.

    Attributes:
        llm: The language model used for generating responses.
        retriever: The document retriever for fetching relevant documents.
        web_search_tool: The tool used for web searches.
        config: Configuration dictionary for the agent.
        tools: List of tools available to the agent.
        graph: The workflow graph of the agent.
        metrics: Dictionary to store performance metrics.

    """

    def __init__(self, llm, retriever, web_search_tool, config: Dict[str, Any] = None):
        """
        Initialize the ReActAgent.

        Args:
            llm: The language model to use.
            retriever: The document retriever.
            web_search_tool: The web search tool.
            config: Optional configuration dictionary.
        """
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
            "cache_hits": 0,
        }
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 5))

    def _create_tools(self):
        """Create and return the list of tools available to the agent."""
        @tool
        @lru_cache(maxsize=self.config.get("cache_size", 100))
        def retrieve_documents(query: str) -> list:
            """Retrieve documents from the vector store based on the query."""
            logger.info(f"Retrieving documents for query: {query}")
            self.metrics["cache_hits"] += 1
            return self.retriever.invoke(query)

        @tool
        def grade_document_retrieval(step_by_step_reasoning: str, score: int) -> str:
            """Grade the relevance of retrieved documents."""
            logger.info(f"Grading document retrieval. Score: {score}")
            if score == 1:
                return "Docs are relevant. Generate the answer to the question."
            return "Docs are not relevant. Use web search to find more documents."

        @tool
        @lru_cache(maxsize=self.config.get("cache_size", 100))
        def web_search(query: str) -> list[Document]:
            """Run web search on the question."""
            logger.info(f"Performing web search for query: {query}")
            self.metrics["cache_hits"] += 1
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
        """Create and return the assistant runnable."""
        primary_assistant_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.get("system_prompt",
                                       "You are a helpful assistant tasked with answering user questions using the provided vector store. "
                                       "Use the provided vector store to retrieve documents. Then grade them to ensure they are relevant before answering the question.")),
            ("placeholder", "{messages}"),
        ])
        assistant_runnable = primary_assistant_prompt | self.llm.bind_tools(self.tools)
        return RunnableLambda(assistant_runnable)

    def _create_graph(self):
        """Create and return the workflow graph."""
        builder = StateGraph(State)
        builder.add_node("assistant", self._create_assistant())
        builder.add_node("tools", self._create_tool_node())
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def _create_tool_node(self):
        """Create and return the tool node for the graph."""
        return ToolNode(self.tools).with_fallbacks(
            [RunnableLambda(self._handle_tool_error)],
            exception_key="error"
        )

    @staticmethod
    def _handle_tool_error(state: State) -> Dict[str, List[ToolMessage]]:
        """Handle errors that occur during tool execution."""
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
        """
        Asynchronously predict the answer to a given question.

        Args:
            question: The input question.

        Returns:
            A dictionary containing the response and messages.

        Raises:
            ValueError: If the input question is invalid.
        """
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
        """
        Predict the answer to a given question.

        Args:
            question: The input question.

        Returns:
            A dictionary containing the response and messages.

        Raises:
            ValueError: If the input question is invalid.
        """
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
        """
        Find all tool calls in the messages returned from the ReAct agent.

        Args:
            messages: The messages to search for tool calls.

        Returns:
            A list of tool call names.
        """
        return [
            tc["name"]
            for m in messages["messages"]
            for tc in getattr(m, "tool_calls", [])
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current metrics of the agent.

        Returns:
            A dictionary of metrics.
        """
        metrics = self.metrics.copy()
        if metrics["total_queries"] > 0:
            metrics["average_response_time"] = round(
                metrics["total_response_time"] / metrics["total_queries"], 2)
        return metrics

    def clear_caches(self):
        """Clear the LRU caches for retrieve_documents and web_search."""
        self.tools[0].clear_cache()  # retrieve_documents
        self.tools[2].clear_cache()  # web_search

    async def process_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of questions asynchronously.

        Args:
            questions: A list of questions to process.

        Returns:
            A list of response dictionaries.
        """
        tasks = [self.predict_async(question) for question in questions]
        return await asyncio.gather(*tasks)

    def run_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Run a batch of questions using a thread pool.

        Args:
            questions: A list of questions to process.

        Returns:
            A list of response dictionaries.
        """
        return list(self.executor.map(self.predict, questions))


class Assistant:
    """
    Assistant class that wraps a runnable object and handles its invocation.

    This class is responsible for executing the assistant's logic and handling
    cases where the assistant needs to be re-prompted for a real output.

    Attributes:
        runnable (Runnable): The runnable object that implements the assistant's logic.
    """

    def __init__(self, runnable: Runnable):
        """
        Initialize the Assistant with a runnable object.

        Args:
            runnable (Runnable): The runnable instance to invoke.
        """
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        """
        Call method to invoke the LLM and handle its responses.

        This method will re-prompt the assistant if the response is not a tool call
        or does not contain meaningful text.

        Args:
            state (State): The current state containing messages.
            config (RunnableConfig): The configuration for the runnable.

        Returns:
            dict: The final state containing the updated messages.
        """
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
__version__ = "0.2.0"