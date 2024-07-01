import streamlit as st

from common.utils import data_ingestion, qdrant_add_documents, get_embeddings_model, qdrant_get_documents, get_llm, \
    get_answers


def main():
    # Page config
    st.set_page_config("Chat PDF 🔍")

    # Header
    st.header("Chat with all your PDFs 🤖")

    # Question - filling box
    user_question = st.text_input("Ask a Question from the PDF Files🚀")

    # Hugging face embeddings
    huggingface_embeddings = get_embeddings_model()
    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Qdrant Vectors Update"):
            with st.spinner("Processing..."):
                # Load PDFs
                docs = data_ingestion()

                qdrant_add_documents(embeddings=huggingface_embeddings, docs=docs)

                st.success("Done")
    if st.button("Generate Response"):
        with st.spinner("Thinking..."):
            # Connect to db and get documents
            qdrant_vector_store = qdrant_get_documents(embeddings=huggingface_embeddings)

            # Load llm
            llm = get_llm()

            # Chain
            answer = get_answers(llm=llm, vector_store=qdrant_vector_store, query=user_question)

            st.write(answer)

            st.success("Done")


if __name__ == "__main__":
    main()
