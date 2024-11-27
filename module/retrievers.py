from langchain.chains import RetrievalQA
from langchain_community.retrievers import WikipediaRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageEmbeddings,
    ChatUpstage
    )
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class Retriever:
    # Retriever를 만든다.
    def __init__(self, vectorstore: FAISS, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.retriever = None
        self.retriever_type = None

    def vectorstore_retriever(self, **kwargs):
        self.retriever = self.vectorstore.as_retriever(**kwargs)
        self.retriever_type = "vectorstore"
        return self.retriever

    def wikipedia_retriever(self, **kwargs):
        self.retriever = WikipediaRetriever(**kwargs)
        self.retriever_type = "wiki"
        return self.retriever

    def ensemble_retriever(self,weights=[0.5,0.5],faiss_kwargs=None, wikipedia_kwargs=None):
        # Create individual retrievers
        faiss_retriever = self.vectorstore_retriever(**faiss_kwargs if faiss_kwargs else {})
        wikipedia_retriever = self.wikipedia_retriever(**wikipedia_kwargs if wikipedia_kwargs else {})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, wikipedia_retriever],
            weights=weights,
        )
        return ensemble_retriever

    #여기 수정 중
    #def multi_query_retriever(self, **kwargs):
    #     llm = ChatUpstage(api_key=os.getenv("UPSTAGE_API_KEY"))
    #     self.retriever = MultiQueryRetriever(
    #         retriever=self.vectorstore_retriever(**kwargs),
    #         llm = llm
    #     )
    #     self.retriever_type = "multiquery"
    #     return self.retriever