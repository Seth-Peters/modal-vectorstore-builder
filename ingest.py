from modal import Image, Stub


def download_model():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_name_or_path="all-mpnet-base-v2", cache_folder="/models")


transformer_image = (
    Image.debian_slim().pip_install("sentence_transformers", "langchain").run_function(download_model)
)

image = Image.debian_slim().pip_install(
    "langchain",
    "pypdf",
    "faiss-cpu",
    "sentence_transformers",
)

stub = Stub(
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

    # pdf_urls = pdf_urls[:50]

    documents = []
    for result in load_pdf_url.map(pdf_urls, return_exceptions=True):
        documents.append(result)

    # print("documents: ", len(documents))
    # print("documents[0]: ", documents[0])

    documents = sum(documents, [])

    split_documents = split_docs.call(documents)

    embeddings_list = []
    for results in embed_single_document.map(split_documents, return_exceptions=True):
        embeddings_list.append(results)

    upsert_to_vectorstore = []
    for text, embedding in zip(split_documents, embeddings_list):
        upsert_to_vectorstore.append((text.page_content, embedding))

    # print("split_documents: ", len(split_documents))
    # print("embeddings_list: ", len(embeddings_list))
    # print("upsert_to_vectorstore: ", len(upsert_to_vectorstore))

    create_vectorstore_of_text(upsert_to_vectorstore)

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


@stub.function(image=transformer_image)
def embed_single_document(doc: str) -> list:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", cache_folder="/models")
    text = doc.page_content
    embeddings = model.encode(text)

    return embeddings


@stub.function(image=image, retries=3)
def create_vectorstore_of_text(docs):
    from langchain.vectorstores.faiss import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

    faiss_index = FAISS.from_embeddings(docs, embedding=HuggingFaceEmbeddings())

    faiss_index.save_local("vectorstore/llm-papers-faiss-testing")
