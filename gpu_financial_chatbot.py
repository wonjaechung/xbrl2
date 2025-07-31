# gpu_financial_chatbot.py

import os
import pickle
import traceback
import warnings
import logging
from typing import List, Optional, Dict, Any
import numpy as np

# Import core libraries
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# GPU acceleration imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU acceleration disabled.")

try:
    import faiss
    FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    FAISS_GPU_AVAILABLE = False
    warnings.warn("FAISS-GPU not available. Using CPU version.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "faiss_index_gpu"
FAISS_INDEX_PATH_CPU = "faiss_index_ko"

class GPUAcceleratedEmbeddings(HuggingFaceEmbeddings):
    """Enhanced HuggingFace embeddings with GPU acceleration and batch processing."""
    
    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop('batch_size', 32)
        self.use_gpu = self._detect_gpu_availability()
        
        if self.use_gpu:
            kwargs['model_kwargs'] = kwargs.get('model_kwargs', {})
            kwargs['model_kwargs']['device'] = 'cuda'
            logger.info("Using GPU for embeddings")
        else:
            kwargs['model_kwargs'] = kwargs.get('model_kwargs', {})
            kwargs['model_kwargs']['device'] = 'cpu'
            logger.info("Using CPU for embeddings")
            
        super().__init__(*args, **kwargs)
    
    def _detect_gpu_availability(self) -> bool:
        """Detect if GPU is available for embeddings."""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with batch processing for GPU efficiency."""
        if not texts:
            return []
        
        # Process in batches for better GPU utilization
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1) // self.batch_size}")
            
            try:
                batch_embeddings = super().embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size + 1}: {e}")
                # Fallback to individual processing
                for text in batch:
                    try:
                        embedding = super().embed_documents([text])
                        all_embeddings.extend(embedding)
                    except Exception as single_error:
                        logger.error(f"Error processing single text: {single_error}")
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * self.client.get_sentence_embedding_dimension())
        
        return all_embeddings

class GPUVectorStore:
    """Enhanced vector store with GPU acceleration for FAISS operations."""
    
    def __init__(self, embedding_function, use_gpu: bool = None):
        self.embedding_function = embedding_function
        self.use_gpu = use_gpu if use_gpu is not None else self._detect_faiss_gpu()
        self.vector_store = None
        self.gpu_resource = None
        
        if self.use_gpu:
            logger.info("Initializing GPU-accelerated FAISS")
            self._setup_gpu_resources()
        else:
            logger.info("Using CPU FAISS")
    
    def _detect_faiss_gpu(self) -> bool:
        """Detect if FAISS-GPU is available."""
        return FAISS_GPU_AVAILABLE and TORCH_AVAILABLE and torch.cuda.is_available()
    
    def _setup_gpu_resources(self):
        """Setup GPU resources for FAISS."""
        if self.use_gpu and FAISS_GPU_AVAILABLE:
            try:
                self.gpu_resource = faiss.StandardGpuResources()
                logger.info("GPU resources initialized for FAISS")
            except Exception as e:
                logger.warning(f"Failed to setup GPU resources: {e}. Falling back to CPU.")
                self.use_gpu = False
    
    def create_from_documents(self, documents, embeddings):
        """Create vector store from documents with GPU optimization."""
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Create base FAISS index
        self.vector_store = FAISS.from_documents(
            documents=documents, 
            embedding=embeddings
        )
        
        # Optimize for GPU if available
        if self.use_gpu and self.gpu_resource:
            try:
                self._optimize_for_gpu()
            except Exception as e:
                logger.warning(f"GPU optimization failed: {e}. Using CPU version.")
        
        return self.vector_store
    
    def _optimize_for_gpu(self):
        """Optimize the FAISS index for GPU operations."""
        if not (self.use_gpu and self.gpu_resource and self.vector_store):
            return
        
        try:
            # Get the underlying FAISS index
            cpu_index = self.vector_store.index
            
            # Create GPU index
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, cpu_index)
            
            # Replace the CPU index with GPU index
            self.vector_store.index = gpu_index
            logger.info("Successfully moved FAISS index to GPU")
            
        except Exception as e:
            logger.error(f"Failed to move index to GPU: {e}")
            raise
    
    def save_local(self, path: str):
        """Save the vector store locally."""
        if self.vector_store:
            # Convert back to CPU for saving if needed
            if self.use_gpu:
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.vector_store.index)
                    temp_store = FAISS(
                        embedding_function=self.vector_store.embedding_function,
                        index=cpu_index,
                        docstore=self.vector_store.docstore,
                        index_to_docstore_id=self.vector_store.index_to_docstore_id
                    )
                    temp_store.save_local(path)
                    logger.info(f"GPU vector store saved to {path}")
                except Exception as e:
                    logger.error(f"Failed to save GPU vector store: {e}")
                    self.vector_store.save_local(path)
            else:
                self.vector_store.save_local(path)
    
    @classmethod
    def load_local(cls, path: str, embeddings, allow_dangerous_deserialization: bool = True):
        """Load vector store and optionally move to GPU."""
        instance = cls(embeddings)
        
        try:
            instance.vector_store = FAISS.load_local(
                path, 
                embeddings, 
                allow_dangerous_deserialization=allow_dangerous_deserialization
            )
            
            # Move to GPU if available
            if instance.use_gpu:
                instance._optimize_for_gpu()
                
            logger.info(f"Vector store loaded from {path}")
            return instance.vector_store
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None

def create_or_load_vector_store(embeddings):
    """
    Loads the GPU-optimized vector store from disk if it exists, otherwise creates it.
    """
    gpu_store = GPUVectorStore(embeddings)
    
    # Try to load GPU-optimized version first
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("--- Loading Existing GPU-Optimized Vector Store ---")
        vector_store = gpu_store.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        if vector_store:
            return vector_store
    
    # Fallback to CPU version
    if os.path.exists(FAISS_INDEX_PATH_CPU):
        logger.info("--- Loading Existing CPU Vector Store ---")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH_CPU, embeddings, allow_dangerous_deserialization=True)
            
            # Try to optimize for GPU
            if gpu_store.use_gpu:
                logger.info("--- Optimizing loaded vector store for GPU ---")
                gpu_store.vector_store = vector_store
                gpu_store._optimize_for_gpu()
                return gpu_store.vector_store
            
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load CPU vector store: {e}")
    
    logger.info("--- Creating New GPU-Optimized Vector Store ---")
    
    logger.info("--- 1. Loading Documents ---")
    try:
        loader = DirectoryLoader('output/concept_details/', glob="**/*.md", show_progress=True)
        docs = loader.load()
        if not docs:
            logger.error("No documents found in 'output/concept_details/'.")
            return None
        logger.info(f"Loaded {len(docs)} documents.")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return None

    logger.info("\n--- 2. Splitting Documents into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split documents into {len(splits)} chunks.")
    
    logger.info(f"\n--- 3. Creating GPU-Optimized Vector Store (this may take a while...) ---")
    vectorstore = gpu_store.create_from_documents(documents=splits, embeddings=embeddings)
    
    # Save both GPU and CPU versions
    try:
        gpu_store.save_local(FAISS_INDEX_PATH)
        logger.info(f"GPU-optimized vector store saved to '{FAISS_INDEX_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to save GPU vector store: {e}")
        # Fallback to CPU save
        vectorstore.save_local(FAISS_INDEX_PATH_CPU)
    
    return vectorstore

def setup_gpu_optimized_embeddings():
    """Setup GPU-optimized embeddings with fallback options."""
    logger.info("--- Setting up GPU-Optimized Embeddings ---")
    
    # Try different Korean models in order of preference
    model_options = [
        "dragonkue/BGE-m3-ko",  # Current model
        "jhgan/ko-sroberta-multitask",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]
    
    for model_name in model_options:
        try:
            logger.info(f"Trying model: {model_name}")
            
            embeddings = GPUAcceleratedEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                batch_size=32  # Larger batch size for GPU efficiency
            )
            
            # Test the model with a simple embedding
            test_embedding = embeddings.embed_query("테스트 문장")
            logger.info(f"Successfully loaded model: {model_name}")
            logger.info(f"Embedding dimension: {len(test_embedding)}")
            return embeddings
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue
    
    # Final fallback to basic model
    logger.warning("All specialized models failed. Using basic fallback.")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_gpu_info():
    """Get information about available GPU resources."""
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "cuda_devices": 0,
        "faiss_gpu_available": FAISS_GPU_AVAILABLE,
        "device_names": []
    }
    
    if TORCH_AVAILABLE:
        try:
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_devices"] = torch.cuda.device_count()
            info["device_names"] = [torch.cuda.get_device_name(i) for i in range(info["cuda_devices"])]
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")
    
    return info

def main():
    """
    Main function to run the GPU-accelerated RAG-based financial chatbot.
    """
    logger.info("=== GPU-Accelerated Financial Chatbot ===")
    
    # Display GPU information
    gpu_info = get_gpu_info()
    logger.info("--- GPU Information ---")
    for key, value in gpu_info.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\n--- Setting up Models ---")
    try:
        # Setup GPU-optimized embeddings
        embeddings = setup_gpu_optimized_embeddings()
        
        # Setup LLM (keep using Exaone for generation)
        llm = ChatOllama(model="exaone3.5:7.8b")
        
        logger.info("Models initialized successfully.")

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        traceback.print_exc()
        return

    # Create or load GPU-optimized vector store
    vectorstore = create_or_load_vector_store(embeddings)
    if vectorstore is None:
        logger.error("Failed to create or load vector store.")
        return

    logger.info("\n--- Creating RAG Chain ---")
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Retrieve more documents for better context
    )

    template = """
You are a helpful financial expert assistant. 
Answer the user's question based only on the following context.
If you don't know the answer, just say that you don't know.
Provide detailed and accurate information when available.

Context:
{context}

Question: {input}
"""
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    logger.info("GPU-accelerated RAG chain created successfully.")
    
    logger.info("\n--- Running GPU-Accelerated Chatbot ---")
    
    questions = [
        "회사의 영업이익은 얼마인가요?",
        "바이오의약품 부문의 매출액과 영업이익을 알려주세요.",
        "가장 수익성이 높은 부문은 어디인가요?",
        "회사의 전체 매출액 대비 각 부문의 비중은 어떻게 되나요?"
    ]
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[Question {i}] {question}")
        try:
            response = retrieval_chain.invoke({"input": question})
            logger.info(f"[Answer] {response['answer']}")
            
            # Show relevant documents for transparency
            if 'context' in response:
                relevant_docs = response['context'][:2]  # Show top 2 relevant docs
                logger.info(f"[Sources] Based on {len(response.get('context', []))} relevant documents")
                
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            traceback.print_exc()

    logger.info("\n=== GPU-Accelerated Processing Complete ===")

if __name__ == '__main__':
    main()