import fitz  # PyMuPDF
import pdfplumber
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
import os
from dotenv import load_dotenv


load_dotenv()

class DocumentLoader:
    #document를 읽어온다.
    def __init__(self, file_path: str):
        """
        문서 경로를 받아 초기화한다.
        :param file_path: 처리할 문서 파일 경로
        """
        self.file_path = file_path
        self.content = None
        
    def pymupdf_loader(self):
        """ 
        PyMuPDF 
        """
        loader = PyMuPDFLoader(self.file_path)
        return loader.load() # docs 반환

    def pdfplumber_loader(self):
        """PDF Plumber"""
        documents = []

        with pdfplumber.open(self.file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. 표 추출
                tables = page.extract_tables()
                if tables:
                    # 여러 테이블이 있는 경우 병합
                    table_content = "\n\n".join(
                        f"Table {j + 1}:\n{table}" for j, table in enumerate(tables)
                    )
                    content = f"표가 감지된 페이지 {i + 1}:\n{table_content}"
                else:
                    # 2. 텍스트 추출 (PyMuPDF 사용)
                    with fitz.open(self.file_path) as doc:
                        text = doc[i].get_text("text")
                        content = f"텍스트 페이지 {i + 1}:\n{text.strip() if text else '텍스트 없음'}"

                # Document 객체 생성
                documents.append(Document(page_content=content, metadata={"page": i + 1}))
        return documents