import os
import random


def generate_knowledge():
    """Генерирует тестовую базу знаний"""
    # Создаем структуру каталогов
    os.makedirs("knowledge_base/products", exist_ok=True)
    os.makedirs("knowledge_base/pricing", exist_ok=True)
    os.makedirs("knowledge_base/policies", exist_ok=True)

    # Продукты
    products = [
        "Ноутбук ExpertBook B5\nХарактеристики:\n- Процессор: Intel Core i7-1255U\n- Память: 16 ГБ DDR5\n- Дисплей: 14\" FHD IPS\nЦена: 89 990 руб",
        "Планшет TabMaster X3\nХарактеристики:\n- Экран: 10.5\" AMOLED\n- Память: 128 ГБ\nЦена: 34 990 руб",
        "Смартфон SmartX Pro\nХарактеристики:\n- Экран: 6.7\" OLED\n- Память: 256 ГБ\nЦена: 59 990 руб",
        "Монитор UltraView 27\"\nХарактеристики:\n- Разрешение: 4K UHD\n- Частота: 144 Гц\n- Тип матрицы: IPS\nЦена: 32 990 руб",
        "Игровая мышь Gamer Pro\nХарактеристики:\n- DPI: 16000\n- Кнопки: 6 программируемых\nЦена: 4 990 руб"
    ]

    for i, content in enumerate(products, 1):
        with open(f"knowledge_base/products/product_{i}.txt", "w", encoding="utf-8") as f:
            f.write(content)

    # Цены и скидки
    discounts = [
        "Акции июля 2024:\n- Скидка 10% на все ноутбуки\n- Бесплатная доставка при заказе от 5000 руб",
        "Специальные предложения:\n- При покупке планшета - чехол в подарок\n- Рассрочка 0% на 6 месяцев"
    ]

    for i, content in enumerate(discounts, 1):
        with open(f"knowledge_base/pricing/discount_{i}.txt", "w", encoding="utf-8") as f:
            f.write(content)

    # Политики
    policies = {
        "returns.txt": "Условия возврата:\n- Возврат в течение 14 дней\n- Товар должен быть в оригинальной упаковке\n- Не принимаются товары с механическими повреждениями",
        "warranty.txt": "Гарантийные условия:\n- Гарантия 12 месяцев на все товары\n- Расширенная гарантия доступна за дополнительную плату",
        "delivery.txt": "Условия доставки:\n- Бесплатная доставка по городу при заказе от 5000 руб\n- Курьерская доставка: 300 руб\n- Самовывоз из магазинов"
    }

    for filename, content in policies.items():
        with open(f"knowledge_base/policies/{filename}", "w", encoding="utf-8") as f:
            f.write(content)

    print("✅ Сгенерировано 10 документов в knowledge_base/")


if __name__ == "__main__":
    generate_knowledge()