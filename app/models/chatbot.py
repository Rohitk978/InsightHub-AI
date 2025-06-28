# from flask import current_app
# import numpy as np
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_community.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate
# import faiss
# import logging
# import pdfplumber
# from PIL import Image
# import pytesseract
# import os
# import torch
# from transformers import pipeline
# import traceback

# # Setup logging
# logger = logging.getLogger(__name__)
# logging.getLogger('langchain_community.vectorstores.faiss').setLevel(logging.WARNING)

# class CustomHuggingFacePipeline(HuggingFacePipeline):
#     """Custom pipeline to handle question-answering input formatting."""
#     def _call(self, prompt, **kwargs):
#         try:
#             logger.debug(f"Received prompt: {prompt}")
#             if isinstance(prompt, dict) and "question" in prompt and "context" in prompt:
#                 result = self.pipeline({"question": prompt["question"], "context": prompt["context"]}, **kwargs)
#             else:
#                 # Parse string prompt with clear delimiters
#                 if "Context:" not in prompt or "Question:" not in prompt:
#                     raise ValueError("Invalid prompt format: 'Context:' or 'Question:' not found")
#                 parts = prompt.split("Question:", 1)
#                 if len(parts) != 2:
#                     raise ValueError("Invalid prompt format: cannot split on 'Question:'")
#                 context = parts[0].replace("Context:", "").strip()
#                 question = parts[1].split("Answer:", 1)[0].strip()
#                 result = self.pipeline({"question": question, "context": context}, **kwargs)
#             return result["answer"]
#         except Exception as e:
#             logger.error(f"Pipeline error: {str(e)}\n{traceback.format_exc()}")
#             raise ValueError(f"Pipeline processing failed: {str(e)}")

# class DocumentQA:
#     """Class for question-answering on documents using embeddings and QA pipeline."""
#     def __init__(self, document, is_file=False, chunk_size=500, chunk_overlap=50):
#         # Document processing
#         if is_file:
#             if not os.path.exists(document):
#                 logger.error(f"PDF file not found: {document}")
#                 raise ValueError("PDF file not found.")
#             self.text = self._extract_text_from_pdf(document)
#         else:
#             if not document or not isinstance(document, str) or not document.strip():
#                 logger.error("Invalid or empty text document provided")
#                 raise ValueError("Text document is empty or invalid.")
#             self.text = document

#         if not self.text.strip():
#             logger.error("No text extracted from input")
#             raise ValueError("No text could be extracted from the input.")

#         # Text splitting
#         self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

#         # Embeddings initialization
#         model_name = "sentence-transformers/all-mpnet-base-v2"
#         model_kwargs = {
#             'device': 'cuda' if current_app.config.get('CUDA_AVAILABLE', False) and torch.cuda.is_available() else 'cpu'
#         }
#         encode_kwargs = {'normalize_embeddings': False}
#         try:
#             self.hf_embeddings = HuggingFaceEmbeddings(
#                 model_name=model_name,
#                 model_kwargs=model_kwargs,
#                 encode_kwargs=encode_kwargs
#             )
#         except Exception as e:
#             logger.error(f"Embedding initialization failed: {str(e)}\n{traceback.format_exc()}")
#             raise ValueError(f"Embedding initialization failed: {str(e)}")

#         # FAISS index initialization
#         sample_embedding = np.array(self.hf_embeddings.embed_query("sample text"))
#         dimension = sample_embedding.shape[0]
#         if current_app.config.get('CUDA_AVAILABLE', False) and torch.cuda.is_available():
#             try:
#                 res = faiss.StandardGpuResources()
#                 index = faiss.GpuIndexFlatL2(res, dimension)
#                 logger.info("Initialized FAISS index with GPU support")
#             except Exception as e:
#                 logger.warning(f"GPU FAISS failed: {str(e)}\n{traceback.format_exc()}")
#                 index = faiss.IndexFlatL2(dimension)
#                 logger.info("Initialized FAISS index with CPU support")
#         else:
#             index = faiss.IndexFlatL2(dimension)
#             logger.info("Initialized FAISS index with CPU support")

#         # Vector store setup
#         try:
#             self.vectorstore = FAISS(
#                 embedding_function=self.hf_embeddings,
#                 index=index,
#                 docstore=InMemoryDocstore(),
#                 index_to_docstore_id={}
#             )
#             text_splits = self.text_splitter.split_text(self.text)
#             if not text_splits:
#                 logger.error("No text splits generated from document")
#                 raise ValueError("No text could be split from the document.")
#             self.vectorstore.add_texts(text_splits)
#             logger.info("Text split and added to vector store")
#         except Exception as e:
#             logger.error(f"Vector store setup failed: {str(e)}\n{traceback.format_exc()}")
#             raise ValueError(f"Failed to process document: {str(e)}")

#         # QA pipeline initialization
#         try:
#             device = 0 if current_app.config.get('CUDA_AVAILABLE', False) and torch.cuda.is_available() else -1
#             qa_pipeline = pipeline(
#                 "question-answering",
#                 model="deepset/minilm-uncased-squad2",
#                 tokenizer="deepset/minilm-uncased-squad2",
#                 device=device,
#                 max_length=512,
#                 handle_impossible_answer=True
#             )
#             self.llm = CustomHuggingFacePipeline(pipeline=qa_pipeline)
#             logger.info("Initialized QA pipeline with deepset/minilm-uncased-squad2")
#         except Exception as e:
#             logger.error(f"QA pipeline initialization failed: {str(e)}\n{traceback.format_exc()}")
#             raise ValueError(f"QA pipeline initialization failed: {str(e)}")

#         # RetrievalQA setup
#         try:
#             prompt_template = """Context:
# {context}

# Question:
# {question}

# Answer:"""
#             prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#             self.qa = RetrievalQA.from_chain_type(
#                 llm=self.llm,
#                 chain_type="stuff",
#                 retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": prompt}
#             )
#             logger.info("Initialized RetrievalQA chain")
#         except Exception as e:
#             logger.error(f"RetrievalQA setup failed: {str(e)}\n{traceback.format_exc()}")
#             raise ValueError(f"RetrievalQA initialization failed: {str(e)}")

#     def _extract_text_from_pdf(self, pdf_path):
#         """Extract text from PDF using pdfplumber, with OCR fallback."""
#         text = ""
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 for page in pdf.pages:
#                     page_text = page.extract_text() or ""
#                     text += page_text + "\n"
#             logger.info(f"Extracted text length from PDF: {len(text)} characters")
#         except Exception as e:
#             logger.error(f"PDF text extraction failed: {str(e)}\n{traceback.format_exc()}")

#         if not text.strip():
#             logger.info("Attempting OCR for image-based PDF")
#             try:
#                 with pdfplumber.open(pdf_path) as pdf:
#                     for page_num, page in enumerate(pdf.pages, 1):
#                         img = page.to_image(resolution=300).original
#                         img = img.convert("L")  # Convert to grayscale
#                         page_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
#                         if page_text.strip():
#                             text += page_text + "\n"
#                         logger.debug(f"OCR page {page_num}: {len(page_text)} characters extracted")
#             except Exception as e:
#                 logger.error(f"OCR extraction failed: {str(e)}\n{traceback.format_exc()}")
#         return text

#     def answer_query(self, query):
#         """Answer a query using the RetrievalQA chain."""
#         if not query or not isinstance(query, str) or not query.strip():
#             logger.error("Query cannot be empty")
#             raise ValueError("Query cannot be empty")
#         try:
#             result = self.qa.invoke({"query": query})
#             logger.info(f"Generated response for query: {query[:50]}...")
#             return result
#         except Exception as e:
#             logger.error(f"Query error: {str(e)}\n{traceback.format_exc()}")
#             raise ValueError(f"Query failed: {str(e)}")













import os
import numpy as np
import faiss
import pdfplumber
import logging
import torch
import traceback
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain_community.vectorstores.faiss").setLevel(logging.WARNING)

class DocumentQA:
    def __init__(self, document, is_pdf=False):
        """Initialize the QA system with a document (text or PDF content)."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Set custom cache directory
        cache_dir = "D://huggingface_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': False}
        try:
            self.hf_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        except Exception as e:
            logger.error(f"Embedding initialization failed: {str(e)}")
            raise RuntimeError(f"Embedding initialization failed: {str(e)}")

        # Process document
        try:
            if is_pdf:
                # Assume document is a file-like object or path
                with pdfplumber.open(document) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                text = document
            if not text.strip():
                raise ValueError("No text could be extracted from the document.")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise RuntimeError(f"Error processing document: {str(e)}")

        # Split text into chunks
        try:
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self.text_split = text_splitter.split_text(text)
            if not self.text_split:
                raise ValueError("Text splitting failed: no documents generated.")
        except Exception as e:
            logger.error(f"Text splitter initialization failed: {str(e)}")
            raise ValueError(f"Text splitter initialization failed: {str(e)}")

        # Create FAISS vectorstore
        try:
            sample_embedding = np.array(self.hf_embeddings.embed_query("sample text"))
            dimension = sample_embedding.shape[0]
            index = faiss.IndexFlatL2(dimension)
            self.vectorstore = FAISS(
                embedding_function=self.hf_embeddings.embed_query,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            self.vectorstore.add_texts(texts=self.text_split)
        except Exception as e:
            logger.error(f"Vector store setup failed: {str(e)}")
            raise RuntimeError(f"Vector store setup failed: {str(e)}")

        # Initialize LLM via Hugging Face Endpoint
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        try:
            self.llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=token,
                temperature=0.6,
                provider="hf-inference"
            )
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            raise RuntimeError(f"LLM initialization failed: {str(e)}. Ensure your token has access to 'meta-llama/Llama-3.1-8B'.")

        # Define prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, say so, but don't make up an answer.

{context}

Question: {question}
Answer: """
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = RunnableParallel(
            {
                "result": (
                    {
                        "context": self.vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs,
                        "question": RunnablePassthrough()
                    }
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
                ),
                "source_documents": self.vectorstore.as_retriever(search_kwargs={"k": 3})
            }
        )

    def answer_query(self, question):
        """Answer a query based on the document."""
        try:
            if not question.strip():
                raise ValueError("Query cannot be empty.")
            result = self.qa_chain.invoke(question)
            return result
        except Exception as e:
            logger.error(f"Query error: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Error processing query: {str(e)}")