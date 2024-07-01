from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import os
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
import qdrant_client
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.prompts import PromptTemplate

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")


# Load data
def data_ingestion():
    pdf_loader = PyPDFDirectoryLoader("data")
    # Load data
    documents = pdf_loader.load()
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                   chunk_overlap=200)
    # Split docs
    docs = text_splitter.split_documents(documents)

    return docs


def get_embeddings_model():
    huggingface_embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HUGGINGFACEHUB_API_TOKEN,
                                                               model_name="sentence-transformers/all-MiniLM-l6-v2")
    return huggingface_embeddings


def qdrant_client_connection():
    client_connection = qdrant_client.QdrantClient(
        qdrant_host,
        api_key=qdrant_api_key
    )
    return client_connection


def qdrant_add_documents(embeddings, docs):
    # Client connection
    client_connection = qdrant_client_connection()

    text = "How are you?"
    embeddings_text = embeddings.embed_query(text)
    len_embeddings = len(embeddings_text)
    print("len_embeddings")
    print(len_embeddings)
    # Configure collection
    collection_config = qdrant_client.http.models.VectorParams(
        size=len_embeddings,
        distance=qdrant_client.http.models.Distance.COSINE
    )

    # Create the collection
    client_connection.recreate_collection(
        collection_name=qdrant_collection_name,
        vectors_config=collection_config
    )

    # Vector store
    vector_store = Qdrant(
        client=client_connection,
        collection_name=qdrant_collection_name,
        embeddings=embeddings
    )

    # Add documents to vector store
    vector_store.add_documents(docs)


def qdrant_get_documents(embeddings):
    # Client connection
    client_connection = qdrant_client_connection()

    # Vector store
    vector_store = Qdrant(
        client=client_connection,
        collection_name=qdrant_collection_name,
        embeddings=embeddings
    )
    return vector_store


def get_llm():
    chat_groq = ChatGroq(temperature=0, model_name="llama3-70b-8192",
                         groq_api_key=groq_api_key)
    return chat_groq


def get_answers(llm, vector_store, query):
    template = """

        Human: Use the following pieces of context to provide a 
        concise answer to the question at the end but use at least summarize with 
        250 words with detailed explanations. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context

        Question: {question}

        Assistant:"""
    # Define prompt template
    prompt = PromptTemplate(template=template,
                            input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    answer = qa({"query": query})

    return answer['result']
