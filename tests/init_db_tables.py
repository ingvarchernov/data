#!/usr/bin/env python3
"""Скрипт для ініціалізації таблиць БД зі схеми"""

import os
import sys
import logging
from dotenv import load_dotenv

# Завантажуємо .env якщо існує
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Якщо змінних немає - просто використаємо db_manager який візьме з .env автоматично
from optimized.database import db_manager

# Читаємо SQL схему
with open('db.sql', 'r', encoding='utf-8') as f:
    sql_schema = f.read()

# Розбиваємо на окремі команди
sql_commands = sql_schema.split(';')

# Виконуємо кожну команду
from sqlalchemy import text
for command in sql_commands:
    command = command.strip()
    if command:  # Пропускаємо порожні
        with db_manager.sync_engine.connect() as conn:
            try:
                conn.execute(text(command))
                conn.commit()
                # Виводимо тільки перші 50 символів для лаконічності
                cmd_preview = ' '.join(command.split()[:7])
                logger.info(f"✅ Виконано: {cmd_preview}...")
            except Exception as e:
                conn.rollback()  # Відкатуємо транзакцію після помилки
                # Ігноруємо помилки "вже існує"
                if "already exists" in str(e).lower():
                    cmd_preview = ' '.join(command.split()[:7])
                    logger.info(f"⏭️ Пропускаємо (вже існує): {cmd_preview}...")
                else:
                    logger.error(f"❌ Помилка: {e}")
                    cmd_preview = ' '.join(command.split()[:7])
                    logger.error(f"Команда: {cmd_preview}...")

logger.info("✅ Схема БД успішно ініціалізована!")
