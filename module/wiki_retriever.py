import wikipediaapi
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import re
import numpy as np

load_dotenv()

class WikipediaRetriever:
    def __init__(self, language='en', top_k = 5):
        self.wiki = wikipediaapi.Wikipedia('NLP_RAG(kateking001130@ewhain.net)',
                                           language,
                                           extract_format=wikipediaapi.ExtractFormat.WIKI)
        self.model = ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY"))
        self.embedding_model = UpstageEmbeddings(api_key=os.getenv("UPSTAGE_API_KEY"),
                                                 model="solar-embedding-1-large")
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="interquartile",
            breakpoint_threshold_amount=0.7
        )
        self.vector_store = None
        self.top_k = top_k

    def extract_keywords_with_model(self, question):
        prompt_template = PromptTemplate.from_template(
            """ 
            You are an AI model designed to extract up to 3 key English keywords from questions.
            Question: "{question}"
            Response: Provide only the key English keywords, separated by commas.
            """
        )
        llm = self.model
        chain= (
            {"question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        keywords = response.strip().split(",")  
        return [keyword.strip() for keyword in keywords]  

    def search_wikipedia(self, keywords):
        documents = []
        for keyword in keywords:
            page = self.wiki.page(keyword)
            if page.exists():
                documents.append(Document(
                    page_content=page.text,
                    metadata={"Title": page.title}
                ))
        return documents

    def create_vector_store(self, documents):
        chunked_documents = self.semantic_chunker.split_documents(documents)
        self.vector_store = FAISS.from_documents(chunked_documents, self.embedding_model)

    def retrieve_context(self, question):
        if not self.vector_store:
            return "Vector store is not initialized. Please create it first."

        query_embedding = self.embedding_model.embed_query(question)
        results = self.vector_store.similarity_search_by_vector(query_embedding, k=self.top_k)
        return "\n\n".join([doc.page_content for doc in results])

    def retrieve(self, query,**kwargs):
        keywords = self.extract_keywords_with_model(query)
        if not keywords:
            return []
        
        documents = self.search_wikipedia(keywords)
        if not documents:
            return []

        self.create_vector_store(documents)
        retrieved_context = self.retrieve_context(query)


        results = []
        for doc in documents:
            results.append(Document(
                page_content=doc.page_content,
                metadata={
                    "source": "Wikipedia",
                    "Title": doc.metadata.get("Title", ""),
                }
            ))

        return results

    
class EnsembleRetriever:
    def __init__(self, static_vector_store, top_k=5):
        """
        Combine a static FAISS vector store with a dynamic Wikipedia retriever.
        """
        self.static_vector_store = static_vector_store
        self.wikipedia_retriever = WikipediaRetriever()
        self.top_k = top_k

    def invoke(self, query):
        """
        Retrieve combined results from both sources.
        """
        combined_results = []
        # FAISS retrieval
        static_results = self.static_vector_store.invoke(query)
        combined_results = static_results
        if len(static_results)==0:
        # Wikipedia retrieval
            dynamic_results = self.wikipedia_retriever.retrieve(query)
            combined_results = dynamic_results
        # Weight and combine results
        return combined_results[:self.top_k]