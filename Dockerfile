# ═══════════════════════════════════════════════════
#  ScooterParts — Dockerfile для Render.com
# ═══════════════════════════════════════════════════
#
# Деплой на Render:
# 1. Создайте новый Web Service на render.com
# 2. Подключите репозиторий (GitHub / GitLab)
# 3. Runtime: Docker
# 4. Dockerfile Path: ./Dockerfile
# 5. Добавьте переменные окружения (Environment Variables):
#       DATABASE_URL  = postgresql://... (из Render PostgreSQL)
#       SECRET_KEY    = <случайная строка 32+ символа>
#       ADMIN_PASSWORD = <ваш пароль>
#       PAYMENT_API_KEY = (после подключения эквайринга)
#       PAYMENT_SHOP_ID = (после подключения эквайринга)
#       PAYMENT_SECRET_KEY = (после подключения эквайринга)
#       PAYMENT_CALLBACK_URL = https://your-app.onrender.com/api/payment/callback
# ═══════════════════════════════════════════════════

FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение
COPY . .

# Создаём нужные папки с правильными правами
RUN mkdir -p /app/static/uploads \
             /app/static/images \
             /app/static/favicon \
             /app/data && \
    chmod -R 777 /app/static/uploads && \
    chmod -R 755 /app/static/images && \
    chmod -R 755 /app/static/favicon

# Порт
EXPOSE 8000

# Запуск (Render передаёт PORT через переменную окружения)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
