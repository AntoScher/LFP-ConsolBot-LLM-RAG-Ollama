import os
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_prompt():
    """Загружает системный промпт из файла"""
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Ошибка загрузки промпта: {e}")
        return "Ты — AI-ассистент. Отвечай на вопросы профессионально."


def init_qa_chain(retriever, model_name):
    """Инициализирует цепочку вопрос-ответ"""
    # Загрузка системного промпта
    system_prompt = load_prompt()

    # Шаблон промпта
    prompt_template = f"""{system_prompt}

    Контекст: {{context}}
    Вопрос: {{question}}
    Ответ:"""

    # Инициализация модели Ollama
    llm = Ollama(
        model=model_name,
        temperature=0.3,
        num_predict=256,
        num_gpu=int(os.getenv("OLLAMA_NUM_GPU", 1)),
        num_thread=int(os.getenv("OLLAMA_NUM_THREADS", 4))
    )

    # Создание цепочки
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        },
        return_source_documents=False
    )