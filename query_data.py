import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from get_embedding import get_embedding

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based only the data provided.:

{context}

---

Answer the question based on the above context: {question}
"""

PROMPT_TEMPLATE1 = """
        Use the following pieces of context to answer the question at the end. 
        Check context very carefully and reference and try to make sense of that before responding.
        If you don't know the answer, just say you don't know. 
        Don't try to make up an answer.
        Answer must be to the point.
        Think step-by-step.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding())

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = OllamaLLM(model="llama3.2:1b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()