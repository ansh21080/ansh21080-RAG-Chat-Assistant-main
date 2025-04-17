from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        # Determine file type and use appropriate loader
        file_extension = os.path.splitext(self.file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(self.file_path)
        elif file_extension == '.txt':
            loader = TextLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        documents = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=1500,  # Increased from 1000
            chunk_overlap=300,  # Increased from 200
            separator="\n",
            length_function=len
        )
        texts = splitter.split_documents(documents)
        return texts

if __name__ == "__main__":
    processor = DocumentProcessor("data/sample.pdf")
    texts = processor.load_and_split()
    print(f"Processed {len(texts)} text chunks")