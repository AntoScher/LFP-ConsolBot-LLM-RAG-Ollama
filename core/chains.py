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
        self.cache = {}  # Простой кэш для ответов
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Инициализация модели Ollama
        self.llm = Ollama(
            model=model_name,
            temperature=0.3,
            num_predict=256,
            num_gpu=int(os.getenv("OLLAMA_NUM_GPU", 1)),
            num_thread=int(os.getenv("OLLAMA_NUM_THREADS", 6))  # Оптимальное значение для стабильности
        )
        
        print(f"✅ QA цепочка инициализирована с моделью {model_name}")
    
    def _is_similar_query(self, query1: str, query2: str, threshold: float = 0.5) -> bool:  # Снижен порог с 0.7 до 0.5
        """Проверяет, похожи ли два запроса"""
        # Простая проверка по ключевым словам
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 or not words2:
            return False
        
        # Вычисляем сходство Жаккара
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return False
        
        similarity = intersection / union
        return similarity >= threshold
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает запрос и возвращает ответ"""
        query = inputs.get("query", "").strip().lower()
        
        # Проверяем кэш (точное совпадение)
        if query in self.cache:
            self.cache_hits += 1
            print(f"⚡ Кэш-хит! Ответ из кэша (всего хит: {self.cache_hits})")
            return {"result": self.cache[query]}
        
        # Проверяем похожие запросы в кэше
        for cached_query in self.cache.keys():
            if self._is_similar_query(query, cached_query):
                self.cache_hits += 1
                print(f"⚡ Кэш-хит! Похожий запрос найден (всего хит: {self.cache_hits})")
                return {"result": self.cache[cached_query]}
        
        self.cache_misses += 1
        print(f"🔄 Кэш-мисс! Обработка запроса (всего мисс: {self.cache_misses})")
        
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
            
            # Обработка кодировки ответа
            if isinstance(response, bytes):
                try:
                    response = response.decode('utf-8')
                except UnicodeDecodeError:
                    response = response.decode('utf-8', errors='ignore')
            elif not isinstance(response, str):
                response = str(response)
            
            # Сохраняем в кэш (ограничиваем размер кэша)
            if len(self.cache) < 50:  # Максимум 50 записей
                self.cache[query] = response
            
            return {"result": response}
        except Exception as e:
            print(f"⚠️ Ошибка получения ответа от модели: {e}")
            return {"result": "Извините, произошла ошибка при обработке запроса."}


def init_qa_chain(retriever, model_name):
    """Инициализирует цепочку вопрос-ответ"""
    return CustomQAChat(retriever, model_name)