import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class PrimeTest(ABC):
    def __init__(self):
        self.module = "prime test"

    def get_prompts_in_list(self, csv):
        queries=[]
        loader = CSVLoader(file_path=csv)
        data = loader.load()
        for document in data:
            query = document.page_content.split("Question:")[1]
            queries.append(query)

        return queries

    @abstractmethod
    def generate_repsonse(self,csv):
        pass

if __name__ == "__main__":
    pt = PrimeTest()
    queries = pt.get_prompts_in_list("gg")
    print(queries)