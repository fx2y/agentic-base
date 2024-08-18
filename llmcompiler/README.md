# LLMCompiler

LLMCompiler is a Python implementation of the LLMCompiler architecture, designed to speed up the execution of agentic tasks by eagerly executing tasks within a DAG. It reduces costs on redundant token usage by minimizing calls to the LLM.

## Features

- Efficient task planning and execution
- Multi-threading for parallel task execution
- Adaptive replanning capabilities
- Modular design for easy extension and customization

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llmcompiler.git
   cd llmcompiler
   ```

2. Install the required dependencies:
   ```
   pip install langchain langchain_openai langgraph
   ```

3. Set up your API keys as environment variables:
   ```
   export OPENAI_API_KEY=your_openai_api_key
   export TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

Here's a basic example of how to use LLMCompiler:

```python
from llmcompiler.llm_compiler import LLMCompiler

# Initialize the LLM and tools
llm = ChatOpenAI(model="gpt-4-turbo-preview")
calculate = get_math_tool(llm)
search = TavilySearchResults(max_results=1)
tools = [search, calculate]

# Create an LLMCompiler instance
compiler = LLMCompiler(llm, tools)

# Run the LLMCompiler with a question
question = "What's the GDP of New York raised to the power of 2?"
for step in compiler.stream(question):
    print(step)
    print("---")

final_answer = step[END][-1].content
print("Final answer:", final_answer)
