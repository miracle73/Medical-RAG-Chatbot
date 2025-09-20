from langchain.chains import RetrievalQA
# The RetrievalQA class from LangChain, which creates a chain that retrieves relevant documents 
# and uses them to answer questions.
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID,HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """ Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=["context" , "question"])

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded")
# Creates the main QA chain by using the loaded LLM and setting the chain type to "stuff" which combines all retrieved documents into one prompt. 
# It creates a retriever from the vector store that returns only the single most relevant document (k=1), configures it to not return source documents in the final output, 
# and applies the custom prompt template to structure how the LLM processes the question and context.
# The retriever with k=1 will only return the single most relevant chunk from across all those documents.
# So it searches through all document chunks in the vector store and picks just the one most similar chunk to answer the question, regardless of which original document that chunk came from.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 1}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        logger.info("Successfully created the QA chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        # 🚨 Explicitly return None on failure
        return None