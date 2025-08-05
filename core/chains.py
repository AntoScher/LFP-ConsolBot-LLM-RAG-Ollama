import os
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from typing import List, Dict, Any


def load_prompt():
    """Загружает системный промпт из файла"""
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Ошибка загрузки промпта: {e}")
        return "Ты — AI-ассистент. Отвечай на вопросы профессионально."


class CustomQAChat:
    """Собственная QA цепочка для работы с локальным retriever"""
    
    def __init__(self, retriever, model_name):
        self.retriever = retriever
        self.system_prompt = load_prompt()
        
        # Инициализация модели Ollama
        self.llm = Ollama(
            model=model_name,
            temperature=0.3,
            num_predict=256,
            num_gpu=int(os.getenv("OLLAMA_NUM_GPU", 1)),
            num_thread=int(os.getenv("OLLAMA_NUM_THREADS", 4))
        )
        
        print(f"✅ QA цепочка инициализирована с моделью {model_name}")
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает запрос и возвращает ответ"""
        query = inputs.get("query", "")
        
        # Получаем релевантные документы
        docs = self.retriever.get_relevant_documents(query)
        
        # Формируем контекст из документов
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Создаем промпт
        prompt = f"""{self.system_prompt}

Контекст: {context}
Вопрос: {query}
Ответ:"""
        
        # Получаем ответ от модели
        try:
            response = self.llm.invoke(prompt)
            return {"result": response}
        except Exception as e:
            print(f"⚠️ Ошибка получения ответа от модели: {e}")
            return {"result": "Извините, произошла ошибка при обработке запроса."}


def init_qa_chain(retriever, model_name):
    """Инициализирует цепочку вопрос-ответ"""
    return CustomQAChat(retriever, model_name)