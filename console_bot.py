from core.database import init_db, log_query
from core.local_embeddings import init_local_vector_store
from core.chains import init_qa_chain
import os
import logging

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO")
)
logger = logging.getLogger("AI-Assistant")


def main():
    logger.info("🔄 Инициализация системы...")

    # Инициализация БД
    init_db()

    # Инициализация локального векторного хранилища
    retriever = init_local_vector_store()

    # Инициализация цепочки QA
    model_name = os.getenv("OLLAMA_MODEL", "phi3:3.8b-mini-128k-instruct-q4_K_M")
    qa_chain = init_qa_chain(retriever, model_name)

    logger.info("✅ Система готова к работе")
    print("\nВведите ваш запрос (exit для выхода):")

    user_id = 1  # Для консольной версии

    while True:
        try:
            query = input("> ").strip()
            if query.lower() in ["exit", "quit"]:
                logger.info("Завершение работы...")
                break

            if len(query) < 3:
                print("❌ Слишком короткий запрос. Минимум 3 символа.")
                continue

            # Обработка запроса
            logger.info(f"Обработка запроса: {query[:30]}...")
            result = qa_chain({"query": query})
            response = result["result"]

            # Логирование
            log_query(user_id, query, response)

            # Вывод ответа
            print("\n💬 Ответ:", response, "\n")
            logger.info("Ответ отправлен")

        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}")
            print("⚠️ Произошла ошибка. Попробуйте другой запрос")


if __name__ == "__main__":
    main()