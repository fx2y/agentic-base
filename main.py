import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union, Sequence
from functools import lru_cache

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, chain as as_runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, START

from math_tools import get_math_tool
from output_parser import LLMCompilerPlanParser, Task

class FinalResponse(BaseModel):
    """The final response/answer."""
    response: str

class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )

class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""
    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]

class LLMCompiler:
    def __init__(self, llm: BaseChatModel, tools: Sequence[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.planner = self._create_planner()
        self.joiner = self._create_joiner()
        self.graph = self._create_graph()

    def _create_planner(self):
        base_prompt = hub.pull("wfh/llm-compiler")
        tool_descriptions = "\n".join(
            f"{i+1}. {tool.description}\n"
            for i, tool in enumerate(self.tools)
        )
        planner_prompt = base_prompt.partial(
            replan="",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )
        replanner_prompt = base_prompt.partial(
            replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
            " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )

        def should_replan(state: list):
            return isinstance(state[-1], SystemMessage)

        def wrap_messages(state: list):
            return {"messages": state}

        def wrap_and_get_last_index(state: list):
            next_task = 0
            for message in state[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = message.additional_kwargs["idx"] + 1
                    break
            state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
            return {"messages": state}

        return (
            RunnableBranch(
                (should_replan, wrap_and_get_last_index | replanner_prompt),
                wrap_messages | planner_prompt,
            )
            | self.llm
            | LLMCompilerPlanParser(tools=self.tools)
        )

    def _create_joiner(self):
        joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(examples="")
        runnable = create_structured_output_runnable(JoinOutputs, self.llm, joiner_prompt)

        def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
            response = [AIMessage(content=f"Thought: {decision.thought}")]
            if isinstance(decision.action, Replan):
                return response + [
                    SystemMessage(
                        content=f"Context from last attempt: {decision.action.feedback}"
                    )
                ]
            else:
                return {"messages": response + [AIMessage(content=decision.action.response)]}

        def select_recent_messages(state) -> dict:
            messages = state["messages"]
            selected = []
            for msg in messages[::-1]:
                selected.append(msg)
                if isinstance(msg, HumanMessage):
                    break
            return {"messages": selected[::-1]}

        return select_recent_messages | runnable | _parse_joiner_output

    @staticmethod
    @lru_cache(maxsize=100)
    def _retrieve_documents(question: str):
        # Implement document retrieval logic here
        pass

    @staticmethod
    def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
        results = {}
        for message in messages[::-1]:
            if isinstance(message, FunctionMessage):
                results[int(message.additional_kwargs["idx"])] = message.content
        return results

    @staticmethod
    def _execute_task(task, observations, config):
        tool_to_use = task["tool"]
        if isinstance(tool_to_use, str):
            return tool_to_use
        args = task["args"]
        try:
            if isinstance(args, str):
                resolved_args = LLMCompiler._resolve_arg(args, observations)
            elif isinstance(args, dict):
                resolved_args = {
                    key: LLMCompiler._resolve_arg(val, observations) for key, val in args.items()
                }
            else:
                resolved_args = args
        except Exception as e:
            return (
                f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
                f" Args could not be resolved. Error: {repr(e)}"
            )
        try:
            return tool_to_use.invoke(resolved_args, config)
        except Exception as e:
            return (
                f"ERROR(Failed to call {tool_to_use.name} with args {args}."
                + f" Args resolved to {resolved_args}. Error: {repr(e)})"
            )

    @staticmethod
    def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
        ID_PATTERN = r"\$\{?(\d+)\}?"

        def replace_match(match):
            idx = int(match.group(1))
            return str(observations.get(idx, match.group(0)))

        if isinstance(arg, str):
            return re.sub(ID_PATTERN, replace_match, arg)
        elif isinstance(arg, list):
            return [LLMCompiler._resolve_arg(a, observations) for a in arg]
        else:
            return str(arg)

    @as_runnable
    def _schedule_task(task_inputs, config):
        task: Task = task_inputs["task"]
        observations: Dict[int, Any] = task_inputs["observations"]
        try:
            observation = LLMCompiler._execute_task(task, observations, config)
        except Exception:
            import traceback
            observation = traceback.format_exception()
        observations[task["idx"]] = observation

    @staticmethod
    def _schedule_pending_task(task: Task, observations: Dict[int, Any], retry_after: float = 0.2):
        while True:
            deps = task["dependencies"]
            if deps and (any([dep not in observations for dep in deps])):
                time.sleep(retry_after)
                continue
            LLMCompiler._schedule_task.invoke({"task": task, "observations": observations})
            break

    @as_runnable
    def _schedule_tasks(scheduler_input: Dict):
        tasks = scheduler_input["tasks"]
        args_for_tasks = {}
        messages = scheduler_input["messages"]
        observations = LLMCompiler._get_observations(messages)
        task_names = {}
        originals = set(observations)
        futures = []
        retry_after = 0.25
        with ThreadPoolExecutor() as executor:
            for task in tasks:
                deps = task["dependencies"]
                task_names[task["idx"]] = (
                    task["tool"] if isinstance(task["tool"], str) else task["tool"].name
                )
                args_for_tasks[task["idx"]] = task["args"]
                if deps and (any([dep not in observations for dep in deps])):
                    futures.append(
                        executor.submit(
                            LLMCompiler._schedule_pending_task, task, observations, retry_after
                        )
                    )
                else:
                    LLMCompiler._schedule_task.invoke(dict(task=task, observations=observations))
            wait(futures)
        new_observations = {
            k: (task_names[k], args_for_tasks[k], observations[k])
            for k in sorted(observations.keys() - originals)
        }
        tool_messages = [
            FunctionMessage(
                name=name, content=str(obs), additional_kwargs={"idx": k, "args": task_args}
            )
            for k, (name, task_args, obs) in new_observations.items()
        ]
        return tool_messages

    def _plan_and_schedule(self, state):
        messages = state["messages"]
        tasks = self.planner.stream(messages)
        scheduled_tasks = self._schedule_tasks.invoke(
            {
                "messages": messages,
                "tasks": tasks,
            }
        )
        return {"messages": [scheduled_tasks]}

    def _create_graph(self):
        graph_builder = StateGraph(dict)
        graph_builder.add_node("plan_and_schedule", self._plan_and_schedule)
        graph_builder.add_node("join", self.joiner)
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
        return graph_builder.compile()

    def run(self, question: str, config: Dict = None):
        if config is None:
            config = {}
        initial_state = {"messages": [HumanMessage(content=question)]}
        return self.graph.invoke(initial_state, config)

    def stream(self, question: str, config: Dict = None):
        if config is None:
            config = {}
        initial_state = {"messages": [HumanMessage(content=question)]}
        return self.graph.stream(initial_state, config)

# Usage example
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    calculate = get_math_tool(llm)
    search = TavilySearchResults(max_results=1)
    tools = [search, calculate]

    compiler = LLMCompiler(llm, tools)
    
    question = "What's the GDP of New York raised to the power of 2?"
    for step in compiler.stream(question):
        print(step)
        print("---")

    final_answer = step[END][-1].content
    print("Final answer:", final_answer)