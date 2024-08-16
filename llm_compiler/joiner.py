from typing import List, Union

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class FinalResponse(BaseModel):
    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


def create_joiner(llm: ChatOpenAI):
    joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
        examples=""
    )
    runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)

    def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
        response = [AIMessage(content=f"Thought: {decision.thought}")]
        if isinstance(decision.action, Replan):
            return response + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}"
                )
            ]
        else:
            return response + [AIMessage(content=decision.action.response)]

    def select_recent_messages(state) -> dict:
        messages = state["messages"]
        selected = []
        for msg in messages[::-1]:
            selected.append(msg)
            if isinstance(msg, HumanMessage):
                break
        return {"messages": selected[::-1]}

    return select_recent_messages | runnable | _parse_joiner_output
