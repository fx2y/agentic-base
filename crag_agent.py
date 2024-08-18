import logging
import time
from functools import lru_cache
from typing import List, TypedDict, Optional

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END


class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}

    def track(self, key, value):
        self.metrics[key] = value

    def get_metrics(self):
        self.metrics['total_time'] = time.time() - self.start_time
        return self.metrics


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    question: str
    generation: str
    search: str
    documents: List[Document]
    steps: List[str]


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class CRAGAgentConfig:
    def __init__(self, relevance_threshold: float = 0.5, max_web_searches: int = 3):
        self.relevance_threshold = relevance_threshold
        self.max_web_searches = max_web_searches


class CRAGAgent:
    def __init__(self, llm, retriever, config: Optional[CRAGAgentConfig] = None):
        self.llm = llm
        self.retriever = retriever
        self.web_search_tool = TavilySearchResults()
        self.rag_chain = self._create_rag_chain()
        self.retrieval_grader = self._create_retrieval_grader()
        self.graph = self._create_graph()
        self.config = config or CRAGAgentConfig()
        self.retrieve_documents = lru_cache(maxsize=100)(self.retrieve_documents)
        self.metrics_tracker = MetricsTracker()

    @lru_cache(maxsize=100)
    def retrieve_documents(self, question: str):
        return self.retriever.invoke(question)

    def _create_rag_chain(self):
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 
            Use the following documents to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise:
            Question: {question} 
            Documents: {documents} 
            Answer: 
            """,
            input_variables=["question", "documents"],
        )
        return prompt | self.llm | StrOutputParser()

    def _create_retrieval_grader(self):
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are a teacher grading a quiz. You will be given: 
        1/ a QUESTION 
        2/ a set of comma separated FACTS provided by the student

        You are grading RELEVANCE RECALL:
        A score of 1 means that ANY of the FACTS are relevant to the QUESTION. 
        A score of 0 means that NONE of the FACTS are relevant to the QUESTION. 
        1 is the highest (best) score. 0 is the lowest score you can give. 

        Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "FACTS: \n\n {documents} \n\n QUESTION: {question}"),
            ]
        )
        return grade_prompt | structured_llm_grader

    def _create_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("web_search", self.web_search)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "search": "web_search",
                "generate": "generate",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def retrieve(self, state: GraphState) -> GraphState:
        try:
            start_time = time.time()
            question = state["question"]
            documents = self.retrieve_documents(question)
            steps = state["steps"]
            steps.append("retrieve_documents")
            logger.info(f"Retrieved {len(documents)} documents for question: {question}")
            self.metrics_tracker.track('retrieve_time', time.time() - start_time)
            self.metrics_tracker.track('num_documents_retrieved', len(documents))
            return {"documents": documents, "question": question, "steps": steps}
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            return {"error": str(e), "question": state["question"], "steps": state["steps"]}

    def generate(self, state: GraphState) -> GraphState:
        try:
            question = state["question"]
            documents = state["documents"]
            generation = self.rag_chain.invoke({"documents": documents, "question": question})
            steps = state["steps"]
            steps.append("generate_answer")
            logger.info(f"Generated answer for question: {question}")
            return {
                "documents": documents,
                "question": question,
                "generation": generation,
                "steps": steps,
            }
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            return {"error": str(e), "question": state["question"], "steps": state["steps"]}

    def grade_documents(self, state: GraphState) -> GraphState:
        try:
            question = state["question"]
            documents = state["documents"]
            steps = state["steps"]
            steps.append("grade_document_retrieval")
            filtered_docs = []
            search = "No"
            for d in documents:
                score = self.retrieval_grader.invoke(
                    {"question": question, "documents": d.page_content}
                )
                grade = score.binary_score
                if grade == "yes" and float(score.confidence) >= self.config.relevance_threshold:
                    filtered_docs.append(d)
                else:
                    search = "Yes"
                    continue
            logger.info(f"Graded documents: {len(filtered_docs)} relevant out of {len(documents)}")
            return {
                "documents": filtered_docs,
                "question": question,
                "search": search,
                "steps": steps,
            }
        except Exception as e:
            logger.error(f"Error in grade_documents: {str(e)}")
            return {"error": str(e), "question": state["question"], "steps": state["steps"]}

    def web_search(self, state: GraphState) -> GraphState:
        try:
            question = state["question"]
            documents = state.get("documents", [])
            steps = state["steps"]
            steps.append("web_search")
            web_results = self.web_search_tool.invoke(
                {"query": question, "max_results": self.config.max_web_searches}
            )
            documents.extend(
                [
                    Document(page_content=d["content"], metadata={"url": d["url"]})
                    for d in web_results
                ]
            )
            logger.info(f"Performed web search for question: {question}")
            return {"documents": documents, "question": question, "steps": steps}
        except Exception as e:
            logger.error(f"Error in web_search: {str(e)}")
            return {"error": str(e), "question": state["question"], "steps": state["steps"]}

    @staticmethod
    def decide_to_generate(state: GraphState) -> str:
        search = state["search"]
        return "search" if search == "Yes" else "generate"

    def run(self, question: str, config: RunnableConfig) -> dict:
        try:
            logger.info(f"Starting CRAG agent for question: {question}")
            result = self.graph.invoke(
                {"question": question, "steps": []},
                config
            )
            logger.info(f"CRAG agent completed for question: {question}")
            result['metrics'] = self.metrics_tracker.get_metrics()
            return result
        except Exception as e:
            logger.error(f"Error in CRAG agent run: {str(e)}")
            return {"error": str(e), "question": question}

# Usage
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# retriever = vectorstore.as_retriever(k=4)
# config = CRAGAgentConfig(relevance_threshold=0.6, max_web_searches=2)
# crag_agent = CRAGAgent(llm, retriever, config)
# response = crag_agent.run("What are the types of agent memory?", {"configurable": {"thread_id": str(uuid.uuid4())}})
