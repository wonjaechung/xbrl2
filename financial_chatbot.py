# financial_chatbot.py

import os
import pickle
import traceback
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

FAISS_INDEX_PATH = "faiss_index_ko"

def create_or_load_vector_store(embeddings):
    """
    Loads the vector store from disk if it exists, otherwise creates it.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        print("--- Loading Existing Vector Store ---")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print("--- Creating New Vector Store ---")
    
    print("--- 1. Loading Documents ---")
    try:
        loader = DirectoryLoader('output/concept_details/', glob="**/*.md", show_progress=True)
        docs = loader.load()
        if not docs:
            print("Error: No documents found in 'output/concept_details/'.")
            return None
        print(f"Loaded {len(docs)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

    print("\n--- 2. Splitting Documents into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} chunks.")
    
    print("\n--- 3. Creating and Saving Vector Store with Korean Model (this may take a while...) ---")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS vector store created and saved to '{FAISS_INDEX_PATH}'.")
    
    return vectorstore

def main():
    """
    Main function to run the RAG-based financial chatbot.
    """
    print("--- Setting up Models ---")
    try:
        # Use a specialized Korean model for embeddings
        model_name = "dragonkue/BGE-m3-ko"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Keep using Exaone for generation
        llm = ChatOllama(model="exaone3.5:7.8b")
        
        print(f"Embedding Model: {model_name}")
        print(f"LLM for Generation: exaone3.5:7.8b")

    except Exception as e:
        print(f"Error initializing models: {e}")
        traceback.print_exc()
        return

    vectorstore = create_or_load_vector_store(embeddings)
    if vectorstore is None:
        return

    print("\n--- 5. Creating RAG Chain ---")
    retriever = vectorstore.as_retriever()

    template = """
You are a helpful financial expert assistant. 
Answer the user's question based only on the following context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {input}
"""
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain created successfully.")
    
    print("\n--- 6. Running Chatbot ---")
    
    questions = [
        "회사의 영업이익은 얼마인가요?",
        "바이오의약품 부문의 매출액과 영업이익을 알려주세요.",
        "가장 수익성이 높은 부문은 어디인가요?"
    ]
    
    for question in questions:
        print(f"\n[Question] {question}")
        try:
            response = retrieval_chain.invoke({"input": question})
            print(f"[Answer] {response['answer']}")
        except Exception as e:
            print(f"An error occurred while processing the question:")
            traceback.print_exc()

if __name__ == '__main__':
    main()
