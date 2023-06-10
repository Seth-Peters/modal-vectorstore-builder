import modal
from modal import Image

image = Image.debian_slim().pip_install(
    "langchain",
    "pypdf",
    "faiss-cpu",
)

stub = modal.Stub(
    name="etl-pdfs",
    image=image,
)


@stub.local_entrypoint()
def main(json_path="data/llm-papers.json"):
    """
    Calls the ETL pipeline using a JSON file with PDF metadata.
    """
    import json

    with open(json_path) as f:
        pdf_infos = json.load(f)

    pdf_urls = [pdf["url"] for pdf in pdf_infos]

    documents = []
    for result in load_pdf_url.map(pdf_urls, return_exceptions=True):
        documents.append(result)

    documents = sum(documents, [])

    split_documents = split_docs(documents)

    create_vectorstore_of_text(split_documents)

    # return vectorstore


@stub.function(image=image)
def load_pdf_url(pdf_url):
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_url)
    doc = loader.load_and_split()
    return doc


@stub.function(image=image)
def split_docs(docs):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter()

    docs = text_splitter.split_documents(docs)

    return docs


@stub.function(image=image)
def create_vectorstore_of_text(docs):
    from langchain.embeddings import FakeEmbeddings
    from langchain.vectorstores.faiss import FAISS

    faiss_index = FAISS.from_documents(docs, embedding=FakeEmbeddings(size=200))

    faiss_index.save_local("vectorstore/llm-papers-faiss")
