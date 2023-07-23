import os

import langchain
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

from generate_response.prime_number_test import PrimeTest

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class PrimeTestNoTool(PrimeTest):
    def __init__(self,model_name):
        super().__init__()
        self.module = "prime test with tool"
        self.model_name=model_name
        llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_name,
            openai_api_key=OPENAI_API_KEY
        )

        self.llm = llm

    def generate_repsonse(self, csv):
        queries = self.get_prompts_in_list(csv)
        responses = []
        tools = load_tools(["pal-math"], llm=self.llm)

        for query in queries:
            agent = initialize_agent(tools,
                                     self.llm,
                                     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                     verbose=True)
            resp = agent.run(query)
            responses.append(resp)

            # try:
            #     resp = agent.run(query)
            #     responses.append(resp)
            # except langchain.schema.output_parser.OutputParserException as e:
            #     resp = agent.run(query)
            #     responses.append(resp)
            #     print("ignoring the exception" )

        return responses


if __name__ == "__main__":
    csv = "/Users/joyeed/LLMDriftRetest/LLMDriftRetest/data/PRIME_EVAL_TEST_10.csv"
    prime_test_no_tool = PrimeTestNoTool(model_name="gpt-3.5-turbo-0301")
    responses = prime_test_no_tool.generate_repsonse(csv)
    for response in responses:
        print(response)




