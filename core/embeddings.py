import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def init_vector_store():
    """Инициализирует векторное хранилище"""
    persist_dir = "chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Если база уже существует - загружаем
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("♻️ Загрузка существующей векторной базы")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        ).as_retriever(search_kwargs={"k": 3})

    print("🆕 Создание новой векторной базы")

    # Загрузка документов
    loader = DirectoryLoader(
        "knowledge_base/",
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()

    # Разделение на чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(docs)

    # Создание и сохранение векторной базы
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb.as_retriever(search_kwargs={"k": 3})