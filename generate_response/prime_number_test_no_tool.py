import os

from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
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

        # prompt_template = PromptTemplate(
        #     template="You are an expert math teacher. Answer the question below."
        #              "\n\nQuestion: {question}\n\nAnswer:",
        #     input_variables=["question"]
        # )
        prompt_template = PromptTemplate(
            template="\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["question"]
        )

        for query in queries:
            prompt = prompt_template.format_prompt(question=query).to_string()
            print(prompt)

            llm_chain = LLMChain(
                llm=prime_test_no_tool.llm,
                prompt=PromptTemplate.from_template(prompt)
            )

            resp = llm_chain({"question":query})
            responses.append(resp)

        return responses


if __name__ == "__main__":
    csv = "/Users/joyeed/LLMDriftRetest/LLMDriftRetest/data/PRIME_EVAL_TEST_10.csv"
    prime_test_no_tool = PrimeTestNoTool(model_name="gpt-3.5-turbo-0301")
    responses = prime_test_no_tool.generate_repsonse(csv)
    for response in responses:
        print(response)
        #print(response["text"])




