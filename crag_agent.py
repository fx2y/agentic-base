from typing import List, TypedDict

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END


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


class CRAGAgent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.web_search_tool = TavilySearchResults()
        self.rag_chain = self._create_rag_chain()
        self.retrieval_grader = self._create_retrieval_grader()
        self.graph = self._create_graph()

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
        question = state["question"]
        documents = self.retriever.invoke(question)
        steps = state["steps"]
        steps.append("retrieve_documents")
        return {"documents": documents, "question": question, "steps": steps}

    def generate(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"documents": documents, "question": question})
        steps = state["steps"]
        steps.append("generate_answer")
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }

    def grade_documents(self, state: GraphState) -> GraphState:
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
            if grade == "yes":
                filtered_docs.append(d)
            else:
                search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }

    def web_search(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state.get("documents", [])
        steps = state["steps"]
        steps.append("web_search")
        web_results = self.web_search_tool.invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        return {"documents": documents, "question": question, "steps": steps}

    @staticmethod
    def decide_to_generate(state: GraphState) -> str:
        search = state["search"]
        return "search" if search == "Yes" else "generate"

    def run(self, question: str, config: RunnableConfig) -> dict:
        return self.graph.invoke(
            {"question": question, "steps": []},
            config
        )

# Usage
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# retriever = vectorstore.as_retriever(k=4)
# crag_agent = CRAGAgent(llm, retriever)
# response = crag_agent.run("What are the types of agent memory?", {"configurable": {"thread_id": str(uuid.uuid4())}})
