from wiki_retriever import WikipediaRetriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

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

    def ensemble_retriever(self,faiss_kwargs=None, top_k=5):
        faiss_retriever = self.vectorstore_retriever(**faiss_kwargs if faiss_kwargs else {})
        ensemble_retriever = EnsembleRetriever(faiss_retriever,top_k=5)
        return ensemble_retriever
