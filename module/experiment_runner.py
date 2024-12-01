import json
import logging
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_upstage import (
    UpstageEmbeddings,
    ChatUpstage
    )
from docs_loaders import DocumentLoader
from text_splitters import TextSplitter
from vector_stores import VectorStore
from retrievers import Retriever
from prompts import Prompts
import html_parsing
from dotenv import load_dotenv

import os

load_dotenv()

class ExperimentLoader:
    def __init__(self,
                 kb_path='../database/ewha.pdf',
                 config_path='./ex_options.json',
                 prompt_options_path="./prompt_options.json", 
                 prompt_folder_path="./templates/"):
        self.kb_path = kb_path
        self.config_path = config_path
        self.prompt_options_path = prompt_options_path
        self.conditions = self._load_conditions()
        self.prompt_options = self._load_prompt_options()
        self.prompts = Prompts(prompt_folder_path)
        logging.info("ExperimentLoader initialized.")

    def _load_conditions(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            logging.info(f"Loading experiment conditions from {self.config_path}.")
            return json.load(f)

    def _load_prompt_options(self):
        with open(self.prompt_options_path, "r", encoding="utf-8") as f:
            logging.info(f"Loading prompt options from {self.prompt_options_path}.")
            return json.load(f)

    def get_scenario(self, scenario_name):
        """Retrieve scenario information."""
        if scenario_name not in self.conditions:
            raise ValueError(f"Scenario '{scenario_name}' not found in the experiment conditions.")
        return self.conditions[scenario_name]

    def get_prompt(self, prompt_option_name):
        """Generate prompt based on the prompt option."""
        return self.prompts.generate_prompt(experiment_options=self.prompt_options_path, option=prompt_option_name)

    def execute_scenario(self, scenario_name, question_data):
        scenario = self.get_scenario(scenario_name)
        splitter_config = scenario["splitter"]
        retriever_config = scenario["retriever"]
        prompt_option_names = scenario["prompt_option"]

        results = []

        if "local_db" in scenario:
            local_db_path = scenario["local_db"]
            print(f"Using local vector database: {local_db_path}")
            vec_store = VectorStore()
            vec_store.load_db_local(local_db_path)
            print(f"Vector Store loaded from: {local_db_path}")


        else:
            # Load Documents
            document_loader = DocumentLoader(self.kb_path)
            if "loader" in scenario:
                docs=html_parsing.extract_text_or_table(self.kb_path)
                print(docs)
            else:    
                docs = document_loader.pdfplumber_loader() # PDF PLUMBER사용해서 로딩

            # Splitter
            splitter_type = splitter_config["type"]
            splitter = TextSplitter()
            splitter.setup_document(docs)
        
            if splitter_type == "semantic":
                splits = splitter.semantic_chunker()
            elif splitter_type == "structure_based":
                splits = splitter.split_into_sections() 
            elif splitter_type =="default":
                splits = splitter.recursive_character_splitter()          
            else:
                raise ValueError(f"Unsupported splitter type: {splitter_type}")

            # Vector store
            vec_store = VectorStore()
            vec_store.setup_FAISS(splits) 
            if "db_save_path" in scenario:
                vec_store.save_db_local(scenario["db_save_path"])
                print(f"DB saved: {scenario['db_save_path']}")

        # Initialize and run the retriever
        print("setup_retriever...")
        retriever_type = retriever_config["type"]
        retriever_instance= Retriever(vec_store.db, vec_store.embeddings) #retriever 인스턴스 생성
        retriever = retriever_instance.vectorstore_retriever() #RETRIEVER 생성
        if retriever_type == "faiss":
            print("db retriever")
            retriever = retriever_instance.vectorstore_retriever()
        elif retriever_type == "ensemble":
            print("db retriever")
            retriever_params = retriever_config["params"]
            retriever = retriever_instance.ensemble_retriever(
                top_k=retriever_params["k"]
            )
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
        print("setup_prompt...")
        prompt_option_dfs = {}

        for prompt_option_name in prompt_option_names:
            print(f"prompt_option_name:{prompt_option_name}")
            # Generate prompt using Prompts class
            prompt = self.prompts.generate_prompt(prompt_options=self.prompt_options_path, option=prompt_option_name)
            prompt_template = PromptTemplate.from_template(prompt)
            logging.info(f"Generated Prompt: {prompt}")

            def process_qa(row):
                question = row["prompts"]
                contexts = retriever.invoke(question)
                context = "\n\n".join([ctx.page_content for ctx in contexts])

                chain = (
                    {"question": RunnablePassthrough(), "context":RunnablePassthrough()}
                    | prompt_template
                    | ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY"))
                    | StrOutputParser()
                )
                return chain.invoke({"question": question, "context": context})
             
            prompt_data = question_data.copy()

            prompt_data["Response"] = prompt_data.apply(process_qa, axis=1)
            print(f"complete invoke {prompt_option_name}")
            prompt_option_dfs[prompt_option_name] = prompt_data

        return prompt_option_dfs