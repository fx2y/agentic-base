from typing import List, Dict, Any, Union
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

class Task(BaseModel):
    idx: int
    tool: Union[BaseTool, str]
    args: Dict[str, Any]
    dependencies: List[int] = Field(default_factory=list)

class LLMCompilerPlanParser:
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
        self.tools["join"] = "join"

    def parse(self, text: str) -> List[Task]:
        tasks = []
        current_task = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("Thought:", "Human:")):
                continue
            if line.startswith(f"{len(tasks) + 1}."):
                if current_task:
                    tasks.append(current_task)
                current_task = self._parse_task(line)
            elif current_task and line:
                current_task.args["input"] += f" {line}"
        if current_task:
            tasks.append(current_task)
        return tasks

    def _parse_task(self, line: str) -> Task:
        parts = line.split("(", 1)
        tool_name = parts[0].split(".", 1)[-1].strip()
        args_str = parts[1].rsplit(")", 1)[0] if len(parts) > 1 else ""
        args = self._parse_args(args_str)
        return Task(
            idx=len(tasks) + 1,
            tool=self.tools[tool_name],
            args=args,
            dependencies=self._extract_dependencies(args),
        )

    def _parse_args(self, args_str: str) -> Dict[str, Any]:
        args = {}
        for arg in args_str.split(","):
            if "=" in arg:
                key, value = arg.split("=", 1)
                args[key.strip()] = value.strip().strip('"')
            else:
                args["input"] = arg.strip().strip('"')
        return args

    def _extract_dependencies(self, args: Dict[str, Any]) -> List[int]:
        dependencies = []
        for value in args.values():
            if isinstance(value, str):
                dependencies.extend([int(m[2:-1]) for m in re.findall(r'\$\{(\d+)\}', value)])
        return dependencies

    def __call__(self, text: str) -> List[Task]:
        return self.parse(text)