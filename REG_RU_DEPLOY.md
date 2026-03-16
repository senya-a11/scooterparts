# Деплой ScooterParts на Reg.ru

## ЧТО ПОКУПАТЬ

### 1. Домен (fmtun.ru) — 129₽/год

При оформлении заказа домена:

| Дополнение | Брать? | Цена |
|---|---|---|
| Пакет «Старт» | ✅ Да | 259₽/год |
| DomainSSL на 6 месяцев | ✅ Да | Бесплатно |
| Почта на домене | ❌ Нет (опционально) | +198₽ |
| Переадресация домена | ❌ Нет | +122₽ |
| Электронное свидетельство о регистрации | ❌ Нет | +203₽ |

> **Электронное свидетельство** — юридический документ для Роспатента/госорганов. Для работы сайта не нужно.

### 2. VPS сервер — обязательно

Шаред-хостинг не подойдёт — Python/ASGI не поддерживается.

| План | Цена | Характеристики | Когда выбрать |
|---|---|---|---|
| Std C1-M1-D10 | 390₽/мес | 1 vCPU, 1 GB RAM, 10 GB SSD | Только для теста |
| **Std C2-M2-D40** | **980₽/мес** | **2 vCPU, 2 GB RAM, 40 GB SSD** | **Рекомендуется** |
| Std C4-M4-D80 | 1960₽/мес | 4 vCPU, 4 GB RAM, 80 GB SSD | При высокой нагрузке |

**Итого: ~1109₽/мес** (VPS 980₽ + домен ~32₽/мес)

---

## ПОШАГОВАЯ ИНСТРУКЦИЯ

### Шаг 1: Купить VPS

1. reg.ru → **Серверы и VPS** → Заказать VPS
2. Выбрать план **Std C2-M2-D40**
3. Настройки при создании:
   - ОС: **Ubuntu 22.04 LTS**
   - Приложения: поставить галочку **Docker**
   - Дата-центр: Москва
4. Записать IP-адрес из письма (вида `185.x.x.x`)

### Шаг 2: Привязать домен к VPS (DNS)

Личный кабинет reg.ru → Домены → fmtun.ru → **Управление DNS**

Добавить A-записи:
```
Тип: A    Имя: @      Значение: <IP вашего VPS>    TTL: 3600
Тип: A    Имя: www    Значение: <IP вашего VPS>    TTL: 3600
```

> DNS обновляется 1-24 часа.

### Шаг 3: Подключиться по SSH

```bash
ssh root@<IP вашего VPS>
# Пароль — из письма от reg.ru
```

### Шаг 4: Проверить/установить Docker

```bash
apt update && apt upgrade -y

# Проверить — если уже установлен при создании VPS:
docker --version
docker compose version

# Если не установлен:
curl -fsSL https://get.docker.com | sh
apt install -y docker-compose-plugin
```

### Шаг 5: Установить Nginx

```bash
apt install -y nginx certbot python3-certbot-nginx
```

### Шаг 6: Загрузить код на сервер

```bash
# На сервере:
apt install -y git
git clone https://github.com/<ваш-аккаунт>/scooterparts.git /app/scooterparts
```

Или с локального компьютера через SCP:
```bash
scp -r /путь/к/scooterparts root@<IP>:/app/scooterparts
```

### Шаг 7: Настроить переменные окружения

```bash
cd /app/scooterparts
nano .env
```

```env
# База данных (PostgreSQL в Docker)
DATABASE_URL=postgresql://scooter_user:SECURE_PASS@postgres:5432/scooter_shop
POSTGRES_USER=scooter_user
POSTGRES_PASSWORD=SECURE_PASS
POSTGRES_DB=scooter_shop

# Безопасность (сгенерировать: python3 -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=<32+ символа случайного текста>
ADMIN_PASSWORD=<надёжный пароль>

# Домен
ALLOWED_ORIGINS=https://fmtun.ru,https://www.fmtun.ru
BASE_URL=https://fmtun.ru
PORT=8000

# Лимиты
AUTH_RATE_LIMIT=10
GLOBAL_RATE_LIMIT=120
PAYMENT_CALLBACK_URL=https://fmtun.ru/api/payment/callback
```

### Шаг 8: Запустить приложение

```bash
cd /app/scooterparts
docker compose up -d --build

# Проверить статус
docker compose ps

# Смотреть логи
docker compose logs -f app

# Тест (должен вернуть JSON)
curl http://localhost:8000/api/csrf-token
```

### Шаг 9: Настроить Nginx

```bash
nano /etc/nginx/sites-available/fmtun.ru
```

```nginx
server {
    listen 80;
    server_name fmtun.ru www.fmtun.ru;

    client_max_body_size 20M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /app/scooterparts/static/;
        expires 30d;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/fmtun.ru /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

### Шаг 10: Получить SSL сертификат

> Используем бесплатный Let's Encrypt (автопродление) вместо DomainSSL от reg.ru.

```bash
# Дождаться обновления DNS — сначала проверить:
curl http://fmtun.ru  # должен ответить сайт

# Получить сертификат (certbot сам обновит nginx)
certbot --nginx -d fmtun.ru -d www.fmtun.ru
# Ввести email, согласиться с условиями

# Проверить автопродление
certbot renew --dry-run
```

### Шаг 11: Автозапуск при перезагрузке

```bash
nano /etc/systemd/system/scooterparts.service
```

```ini
[Unit]
Description=ScooterParts
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/app/scooterparts
ExecStart=docker compose up
ExecStop=docker compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable scooterparts
systemctl start scooterparts
```

---

## БАЗА ДАННЫХ

Ничего докупать не нужно. Варианты:

### Вариант 1 — PostgreSQL в Docker на VPS (рекомендуется)

Уже настроен в `docker-compose.yml`. При `docker compose up` база поднимается автоматически.
- Данные хранятся в Docker volume — не теряются при перезапуске
- Таблицы создаются сами при первом запуске (`db.init_database()`)
- В `.env` указать: `DATABASE_URL=postgresql://scooter_user:PASS@postgres:5432/scooter_shop`

### Вариант 2 — Оставить базу на Render.com

Скопировать `DATABASE_URL` из Render Dashboard → Environment и вставить в `.env` на VPS.

> ⚠️ Бесплатный PostgreSQL на Render живёт 90 дней, потом ~700₽/мес.

### Вариант 3 — PostgreSQL напрямую на VPS (без Docker)

```bash
apt install -y postgresql-15
sudo -u postgres createuser scooter_user
sudo -u postgres createdb scooter_shop
sudo -u postgres psql -c "ALTER USER scooter_user WITH PASSWORD 'SECURE_PASS';"
```

В `.env`: `DATABASE_URL=postgresql://scooter_user:SECURE_PASS@localhost:5432/scooter_shop`

---

## ИТОГОВЫЕ ЗАТРАТЫ

| Услуга | Цена |
|---|---|
| Домен fmtun.ru | 129₽/год |
| Пакет «Старт» | 259₽/год |
| DomainSSL | Бесплатно (6 мес) |
| VPS Std C2-M2-D40 | 980₽/мес |
| **Первый месяц** | **~1368₽** |
| **Ежемесячно** | **~980₽** |

---

## ПРОВЕРКА ПОСЛЕ ДЕПЛОЯ

```bash
# Сайт работает
curl https://fmtun.ru/api/csrf-token

# Логи приложения
docker compose logs -f

# Таблицы базы данных
docker compose exec postgres psql -U scooter_user -d scooter_shop -c "\dt"

# Статус контейнеров
docker compose ps
```
