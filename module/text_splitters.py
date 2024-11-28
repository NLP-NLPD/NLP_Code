import re
import os
from tokenizers import Tokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv

load_dotenv()

class TextSplitter:
    # 읽은 데이터를 split한다.
    def __init__(self):
        """
        load document 를 받아서 split한다.
        """
        self.documents = None
        self.content = None
        self.token_enc = Tokenizer.from_pretrained("upstage/solar-1-mini-tokenizer")


    def setup_document(self,documents):
        self.documents = documents


    # Function to split text into chunks based on token limits
    def chunk_text(self, text, max_tokens=2000):
        """
        Splits the text into chunks, ensuring no chunk exceeds the max token limit.
        """
        # Tokenize the text
        encoding = self.token_enc.encode(text)
        num_tokens = len(encoding.ids)

        # If the number of tokens exceeds max_tokens, split into chunks
        chunks = []
        start = 0
        while start < num_tokens:
            end = min(start + max_tokens, num_tokens)
            chunk_ids = encoding.ids[start:end]  # Select a chunk of token IDs
            chunk = self.token_enc.decode(chunk_ids)  # Decode using the tokenizer instance
            chunks.append(chunk)
            start = end

        return chunks

    # 장, 조 기준 분할
    def split_into_sections(self):
        """
        PDF의 텍스트를 "제n장" 및 "제n조"를 기준으로 나눔
        :param documents: 페이지별 Document 객체 리스트
        :return: 섹션별로 나뉜 결과 딕셔너리
        """
        full_text = "\n".join(doc.page_content for doc in self.documents)

        # "제n장" 기준으로 텍스트 분할
        # chapter_pattern =r"(제\d+장\s*(?:.(?!제\d+장))*)(.*?)(?=제\d+장|$)"
        chapter_pattern = r"(제\d+장[^\n]*)"
        chapters = re.split(chapter_pattern, full_text)

        document_sections = []

        # 각 장별로 처리
        for i in range(1, len(chapters), 2):
            chapter_title = chapters[i].strip()
            chapter_content = chapters[i + 1].strip()

            chapter_doc = Document(
                page_content=chapter_content,
                metadata={"title": chapter_title, "section": "chapter"}
            )

            # 각 장 내에서 "제n조" 기준으로 나눔
            article_pattern = r"(제\d+조(?:의\d+)?)"
            articles = re.split(article_pattern, chapter_content)
            articles_dict = {}
            for j in range(1, len(articles), 2):
                article_title = articles[j].strip()
                article_content = articles[j + 1].strip()

                article_tokens = len(self.token_enc.encode(article_content).ids)
                article_chunks = []

                if article_tokens > 2000:
                    article_chunks = self.chunk_text(article_content, 2000)
                else:
                    # If tokens are within the limit, keep it as a single chunk
                    article_chunks = [article_content]

                for chunk in article_chunks:
                    article_doc = Document(
                        page_content=chunk,
                        metadata={"title": article_title, "section": "article", "chapter": chapter_title}
                    )
                    document_sections.append(article_doc)

        return document_sections

    # RecursiveCharavter
    def recursive_character_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
        return text_splitter.split_documents(self.documents)


    # SemanticChunker
    def semantic_chunker(self):
        upstage_embeddings = UpstageEmbeddings(api_key=os.getenv("UPSTAGE_API_KEY"), 
                                               model="solar-embedding-1-large")
        text_splitter = SemanticChunker(upstage_embeddings,
                                        breakpoint_threshold_type="interquartile",
                                        breakpoint_threshold_amount=0.5,)
        return text_splitter.split_documents(self.documents)