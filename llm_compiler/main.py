from llm_compiler.agents.sql_agent import SQLAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def main():
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    sql_agent = SQLAgent()
    app = sql_agent.create(llm)

    example_question = "What's the temperature in SF raised to the 3rd power?"
    for step in app.stream({"messages": [HumanMessage(content=example_question)]}):
        print(step)
        print("---")

if __name__ == "__main__":
    main()
