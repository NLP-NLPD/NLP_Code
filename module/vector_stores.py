from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os

load_dotenv()

class VectorStore:
    #VectorStore를 셋업한다.
    def __init__(self,splits):
        self.splits = splits
        self.db = None
        self.id = None
        self.embeddings = None

    def setup_embeddings(self):
        self.embeddings = UpstageEmbeddings(api_key=os.getenv("UPSTAGE_API_KEY"), model="solar-embedding-1-large")
        return self.embeddings

    def setup_FAISS(self):
        if self.embeddings is None:
            print("Embeddings not initialized. Setting up embeddings...")
            self.setup_embeddings()

        if not self.embeddings:
            raise ValueError("Embeddings must be initialized before setting up FAISS.")

        try:
            sample_text = "sample text for dimension check"
            sample_embedding = self.embeddings.embed_documents([sample_text])
            dimension_size = len(sample_embedding[0])
        except Exception as e:
            raise ValueError(f"Failed to determine embedding dimension: {e}")

        self.db = FAISS.from_documents(
            self.splits,
            self.embeddings,
        )
        self.id = self.db.index_to_docstore_id
        print(f"{self.id}")
        print("FAISS database successfully set up.")
        return self.db

    def save_db_local(self, local_db_path):
        if not os.path.exists(local_db_path):
            os.makedirs(local_db_path)
            print(f"Created directory at {local_db_path}.")
        try:
            if self.db is None:
                raise ValueError("No FAISS database found to save.")
            self.db.save_local(folder_path=local_db_path)
            print(f"Database saved locally at {local_db_path}.")
        except Exception as e:
            print(f"Error saving database: {e}")

    def load_db_local(self, local_db_path):
        """Load FAISS database from a local directory."""
        try:
            self.db = FAISS.load_local(folder_path=local_db_path, 
                                       embeddings=self.embeddings,
                                       allow_dangerous_deserialization=True)
            self.id = self.db.index_to_docstore_id
        except Exception as e:
            print(f"Error loading database: {e}")
        return self.db