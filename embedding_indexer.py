from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingIndexer:
    def __init__(self):
        # Using sentence-transformers model for better performance
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_vectorstore(self, texts):
        if not texts:
            raise ValueError("No texts provided to create vector store")
        
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore
    
    def save_vectorstore(self, vectorstore, path="vectorstore"):
        vectorstore.save_local(path)
    
    def load_vectorstore(self, path="vectorstore"):
        return FAISS.load_local(path, self.embeddings)

if __name__ == "__main__":
    from document_processor import DocumentProcessor

    processor = DocumentProcessor("data/sample.pdf")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    print("Vector store created successfully")