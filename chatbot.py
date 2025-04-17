class Chatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.conversation_history = []

    def get_response(self, query):
        try:
            # Get response from the chain
            result = self.qa_chain({"query": query})
            
            # Extract answer and source documents
            answer = result.get("result", "I couldn't find an answer to your question.")
            source_docs = result.get("source_documents", [])
            
            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "answer": answer,
                "sources": source_docs
            })
            
            # Return only the answer without sources
            return answer
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_history(self):
        return self.conversation_history