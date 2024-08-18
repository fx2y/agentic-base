import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Dict, Any, Optional

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

    def track(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        self.metrics['total_time'] = time.time() - self.start_time
        return self.metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Task(BaseModel):
    tool: str
    args: Dict[str, Any]
    idx: int
    dependencies: List[int]


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class LLMCompilerCRAGAgent:
    def __init__(self, llm, retriever, config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.retriever = retriever
        self.web_search_tool = TavilySearchResults()
        self.config = config or {}
        self.metrics_tracker = MetricsTracker()
        self.graph = self._create_graph()

    @lru_cache(maxsize=100)
    def retrieve_documents(self, question: str) -> List[Document]:
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
        workflow = StateGraph(Dict[str, Any])
        workflow.add_node("planner", self.planner)
        workflow.add_node("task_fetching_unit", self.task_fetching_unit)
        workflow.add_node("joiner", self.joiner)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("web_search", self.web_search)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "task_fetching_unit")
        workflow.add_edge("task_fetching_unit", "joiner")
        workflow.add_conditional_edges(
            "joiner",
            self.decide_next_action,
            {
                "retrieve": "retrieve",
                "grade": "grade_documents",
                "generate": "generate",
                "search": "web_search",
                "end": END,
            },
        )
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", "joiner")

        return workflow.compile()

    def planner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = state["question"]
            plan_prompt = PromptTemplate(
                template="""Given the question: {question}
                Create a plan to answer this question. The plan should be a list of tasks.
                Each task should be in the format:
                task_name(arg1="value1", arg2="value2")
                Available tasks are: retrieve, grade_documents, generate, web_search
                Plan:
                """,
                input_variables=["question"],
            )
            plan = self.llm.invoke(plan_prompt.format(question=question))
            tasks = self._parse_plan(plan)
            state["tasks"] = tasks
            state["steps"].append("create_plan")
            logger.info(f"Created plan with {len(tasks)} tasks for question: {question}")
            return state
        except Exception as e:
            logger.error(f"Error in planner: {str(e)}")
            state["error"] = str(e)
            return state

    def _parse_plan(self, plan: str) -> List[Task]:
        tasks = []
        for idx, line in enumerate(plan.split('\n')):
            if '(' in line and ')' in line:
                tool = line.split('(')[0].strip()
                args_str = line.split('(')[1].split(')')[0]
                args = {arg.split('=')[0].strip(): arg.split('=')[1].strip('"') for arg in args_str.split(',')}
                tasks.append(Task(tool=tool, args=args, idx=idx, dependencies=[]))
        return tasks

    def task_fetching_unit(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tasks = state["tasks"]
            results = {}

            def execute_task(task):
                if task.tool == "retrieve":
                    return self.retrieve({"question": state["question"]})
                elif task.tool == "grade_documents":
                    return self.grade_documents(
                        {"question": state["question"], "documents": state.get("documents", [])})
                elif task.tool == "generate":
                    return self.generate({"question": state["question"], "documents": state.get("documents", [])})
                elif task.tool == "web_search":
                    return self.web_search({"question": state["question"]})

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(execute_task, task) for task in tasks]
                for future in futures:
                    result = future.result()
                    state.update(result)

            state["steps"].append("execute_tasks")
            logger.info(f"Executed {len(tasks)} tasks")
            return state
        except Exception as e:
            logger.error(f"Error in task_fetching_unit: {str(e)}")
            state["error"] = str(e)
            return state

    def joiner(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = state["question"]
            documents = state.get("documents", [])
            generation = state.get("generation", "")

            joiner_prompt = PromptTemplate(
                template="""Given the question: {question}
                And the current answer: {generation}
                Decide if we need to:
                1. Retrieve more documents
                2. Grade the existing documents
                3. Generate a new answer
                4. Perform a web search
                5. End and return the current answer

                Your decision should be one of: retrieve, grade, generate, search, end
                Decision:
                """,
                input_variables=["question", "generation"],
            )
            decision = self.llm.invoke(joiner_prompt.format(question=question, generation=generation))
            state["joiner_decision"] = decision.strip().lower()
            state["steps"].append("joiner_decision")
            logger.info(f"Joiner decision for question '{question}': {decision}")
            return state
        except Exception as e:
            logger.error(f"Error in joiner: {str(e)}")
            state["error"] = str(e)
            return state

    def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = time.time()
            question = state["question"]
            documents = self.retrieve_documents(question)
            state["documents"] = documents
            state["steps"].append("retrieve_documents")
            logger.info(f"Retrieved {len(documents)} documents for question: {question}")
            self.metrics_tracker.track('retrieve_time', time.time() - start_time)
            self.metrics_tracker.track('num_documents_retrieved', len(documents))
            return state
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            state["error"] = str(e)
            return state

    def grade_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = state["question"]
            documents = state["documents"]
            state["steps"].append("grade_document_retrieval")
            grader = self._create_retrieval_grader()
            filtered_docs = []
            for doc in documents:
                score = grader.invoke({"question": question, "documents": doc.page_content})
                if score.binary_score == "yes":
                    filtered_docs.append(doc)
            state["documents"] = filtered_docs
            logger.info(f"Graded documents: {len(filtered_docs)} relevant out of {len(documents)}")
            return state
        except Exception as e:
            logger.error(f"Error in grade_documents: {str(e)}")
            state["error"] = str(e)
            return state

    def generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = state["question"]
            documents = state["documents"]
            rag_chain = self._create_rag_chain()
            generation = rag_chain.invoke({"documents": documents, "question": question})
            state["generation"] = generation
            state["steps"].append("generate_answer")
            logger.info(f"Generated answer for question: {question}")
            return state
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            state["error"] = str(e)
            return state

    def web_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = state["question"]
            documents = state.get("documents", [])
            state["steps"].append("web_search")
            web_results = self.web_search_tool.invoke(
                {"query": question, "max_results": self.config.get("max_web_searches", 3)}
            )
            documents.extend(
                [
                    Document(page_content=d["content"], metadata={"url": d["url"]})
                    for d in web_results
                ]
            )
            state["documents"] = documents
            logger.info(f"Performed web search for question: {question}")
            return state
        except Exception as e:
            logger.error(f"Error in web_search: {str(e)}")
            state["error"] = str(e)
            return state

    def decide_next_action(self, state: Dict[str, Any]) -> str:
        return state.get("joiner_decision", "end")

    def run(self, question: str, config: RunnableConfig) -> Dict[str, Any]:
        try:
            logger.info(f"Starting LLMCompilerCRAGAgent for question: {question}")
            result = self.graph.invoke(
                {"question": question, "steps": []},
                config
            )
            logger.info(f"LLMCompilerCRAGAgent completed for question: {question}")
            result['metrics'] = self.metrics_tracker.get_metrics()
            return result
        except Exception as e:
            logger.error(f"Error in LLMCompilerCRAGAgent run: {str(e)}")
            return {"error": str(e), "question": question}

# Usage
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# retriever = vectorstore.as_retriever(k=4)
# config = {"max_web_searches": 2, "relevance_threshold": 0.6}
# agent = LLMCompilerCRAGAgent(llm, retriever, config)
# response = agent.run("What are the types of agent memory?", {"configurable": {"thread_id": str(uuid.uuid4())}})
