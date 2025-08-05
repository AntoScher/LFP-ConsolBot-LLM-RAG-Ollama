import os
import hashlib
import numpy as np
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


class LocalEmbeddings:
    """Полностью локальный embeddings без внешних зависимостей"""
    
    def __init__(self, dimension=384):
        self.dimension = dimension
        print(f"✅ Создан локальный embeddings с размерностью {dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для документов"""
        embeddings = []
        for text in texts:
            # Используем SHA-256 для лучшего распределения
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Создаем вектор нужной размерности
            vector = []
            for i in range(self.dimension):
                if i < len(hash_bytes):
                    # Нормализуем значения от 0 до 1
                    vector.append(float(hash_bytes[i % len(hash_bytes)]) / 255.0)
                else:
                    vector.append(0.0)
            
            # Нормализуем вектор
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = [v / norm for v in vector]
            
            embeddings.append(vector)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Создает embeddings для запроса"""
        return self.embed_documents([text])[0]


class LocalVectorStore:
    """Простое локальное векторное хранилище"""
    
    def __init__(self, documents: List[Dict], embeddings: LocalEmbeddings):
        self.documents = documents
        self.embeddings = embeddings
        # Извлекаем содержимое документов
        doc_contents = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                doc_contents.append(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                doc_contents.append(doc['page_content'])
            else:
                doc_contents.append(str(doc))
        
        self.doc_embeddings = embeddings.embed_documents(doc_contents)
        print(f"✅ Создано локальное векторное хранилище с {len(documents)} документами")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Поиск похожих документов"""
        query_embedding = self.embeddings.embed_query(query)
        
        # Вычисляем косинусное сходство
        similarities = []
        for i, doc_embedding in enumerate(self.doc_embeddings):
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Сортируем по убыванию сходства
        similarities.sort(reverse=True)
        
        # Возвращаем топ-k документов
        results = []
        for similarity, idx in similarities[:k]:
            results.append(self.documents[idx])
        
        return results


def init_local_vector_store():
    """Инициализирует локальное векторное хранилище"""
    print("🔄 Инициализация локального векторного хранилища...")
    
    # Создаем локальный embeddings
    embeddings = LocalEmbeddings()
    
    # Загружаем документы с правильной кодировкой
    try:
        loader = DirectoryLoader(
            "knowledge_base/",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=False  # Отключаем многопоточность для стабильности
        )
        docs = loader.load()
        print(f"✅ Загружено {len(docs)} документов")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки документов: {e}")
        # Создаем тестовые документы
        from langchain.schema import Document
        docs = [
            Document(page_content="Тестовый документ о продуктах компании", metadata={"source": "test"}),
            Document(page_content="Информация о доставке и гарантии", metadata={"source": "test"}),
            Document(page_content="Цены и скидки на товары", metadata={"source": "test"})
        ]
    
    # Разделяем на чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Создано {len(chunks)} чанков")
    
    # Создаем локальное хранилище
    vector_store = LocalVectorStore(chunks, embeddings)
    
    # Создаем retriever-интерфейс совместимый с LangChain
    class LocalRetriever:
        def __init__(self, vector_store):
            self.vector_store = vector_store
        
        def get_relevant_documents(self, query: str) -> List[Dict]:
            return self.vector_store.similarity_search(query, k=3)
        
        def __call__(self, query: str) -> List[Dict]:
            return self.get_relevant_documents(query)
        
        @property
        def vectorstore(self):
            return self.vector_store
    
    retriever = LocalRetriever(vector_store)
    # Добавляем атрибуты для совместимости с LangChain
    retriever.search_type = "similarity"
    retriever.search_kwargs = {"k": 3}
    
    return retriever 