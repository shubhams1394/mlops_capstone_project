from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def rag_pipeline(retriever):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Use ONLY the provided CONTEXT to answer the question clearly.
    If the answer cannot be found in the context, say:
    "I don't have enough information in the provided documents."

    CONTEXT:
    {context}

    QUESTION:
    {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain