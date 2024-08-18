from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain.chains import LLMMathChain

def get_math_tool(llm: BaseChatModel) -> BaseTool:
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
    return BaseTool(
        name="Calculator",
        func=llm_math.run,
        description="Useful for when you need to answer questions about math",
    )