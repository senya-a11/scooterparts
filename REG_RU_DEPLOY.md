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

### 2. VPS сервер — обязательно (без ISPmanager!)

> **Почему не ISPmanager?**
> ISPmanager — это панель для PHP-сайтов (WordPress, Битрикс). Наш проект (FastAPI + PostgreSQL + Docker) с ней несовместим:
> - Python/FastAPI — не поддерживается
> - База данных — только MySQL, а нам нужен PostgreSQL
> - Docker — не поддерживается
>
> Вместо ISPmanager используем **Portainer** — браузерный интерфейс для Docker. Он даёт те же удобства (кнопки, логи, мониторинг) но работает с нашим стеком. Устанавливается один раз за 2 команды.

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

### Шаг 3: Подключиться к серверу по SSH

**Что такое SSH?**
SSH — это способ управлять сервером удалённо с твоего компьютера через текстовые команды. Представь, что это как "пульт управления" сервером: ты пишешь команды у себя на компьютере, они выполняются на сервере в Москве.

**Откуда взять IP и пароль?**
После покупки VPS reg.ru пришлёт письмо на email. В письме будет:
- **IP-адрес** сервера — вида `185.123.45.67`
- **Пароль** для подключения

Также IP можно найти в личном кабинете: reg.ru → **Серверы и VPS** → кликнуть на свой сервер.

**Как подключиться (Windows 10/11):**

1. Нажать `Win + R`, ввести `powershell`, нажать Enter — откроется синее окно
2. Ввести команду (заменить IP на свой):

```
ssh root@185.123.45.67
```

3. Первый раз появится вопрос:
```
Are you sure you want to continue connecting (yes/no)?
```
Написать `yes` и нажать Enter.

4. Ввести пароль из письма (при вводе пароля символы не отображаются — это нормально).

5. Если подключение успешно, появится строка вида:
```
root@vps123456:~#
```
Это значит — ты внутри сервера. Теперь все команды выполняются на сервере.

> Чтобы выйти из SSH — написать `exit` и нажать Enter.

---

### Шаг 4: Обновить систему и проверить Docker

**Что такое Docker?**
Docker — программа которая запускает приложения в изолированных "контейнерах". Представь контейнер как коробку: внутри лежит всё необходимое для работы сайта (Python, библиотеки, настройки). Коробка работает одинаково на любом сервере. В нашем проекте Docker уже настроен — файл `docker-compose.yml` описывает какие контейнеры запускать (сайт + база данных).

**Команды (вводить в SSH-сессии, одну за другой):**

```bash
# Обновить список пакетов и установить обновления системы.
# Занимает 1-3 минуты. Нажать Y если спросит подтверждение.
apt update && apt upgrade -y
```

```bash
# Проверить что Docker установлен (ставили при создании VPS)
docker --version
```

Ожидаемый ответ:
```
Docker version 27.x.x, build ...
```

```bash
# Проверить docker compose
docker compose version
```

Ожидаемый ответ:
```
Docker Compose version v2.x.x
```

Если команды не найдены (ошибка `command not found`) — установить вручную:
```bash
curl -fsSL https://get.docker.com | sh
apt install -y docker-compose-plugin
```

---

### Шаг 5: Установить Nginx и Certbot

**Что такое Nginx?**
Nginx — это веб-сервер, который стоит "перед" твоим приложением. Когда пользователь заходит на fmtun.ru, запрос сначала попадает в Nginx, а он уже передаёт его твоему FastAPI-приложению. Это нужно потому что:
- Nginx умеет работать с HTTPS (шифрованием)
- Nginx эффективно отдаёт статические файлы (картинки, CSS)
- Один сервер может хостить несколько сайтов

**Certbot** — программа которая автоматически получает и обновляет бесплатные SSL-сертификаты (Let's Encrypt).

```bash
# Установить Nginx, Certbot и плагин для автонастройки
apt install -y nginx certbot python3-certbot-nginx
```

Ожидаемый результат: установка пройдёт без ошибок, в конце вернётся строка приглашения `root@...#`.

---

### Шаг 6: Скачать код сайта на сервер

**Откуда берётся код?**
Код находится на GitHub. Команда `git clone` скачивает его прямо на сервер из репозитория.

```bash
# Создать папку для приложения
mkdir -p /app

# Скачать код (заменить URL на свой репозиторий GitHub)
git clone https://github.com/<твой-аккаунт>/scooterparts.git /app/scooterparts
```

Например:
```bash
git clone https://github.com/ivan123/scooterparts.git /app/scooterparts
```

Ожидаемый результат:
```
Cloning into '/app/scooterparts'...
remote: Enumerating objects: 150, done.
...
```

```bash
# Перейти в папку проекта
cd /app/scooterparts

# Проверить что файлы скачались
ls
```

Должны появиться: `backend/`, `templates/`, `static/`, `docker-compose.yml`, `Dockerfile` и другие файлы.

---

### Шаг 7: Настроить переменные окружения (.env)

**Что такое .env файл?**
`.env` — это файл с настройками и паролями приложения. Он не хранится в GitHub (по соображениям безопасности), поэтому нужно создать его вручную на сервере. Каждая строка — это переменная: `ИМЯ=значение`.

**Как редактировать файлы в терминале — редактор nano:**
`nano` — простой текстовый редактор прямо в терминале.
- Стрелки клавиатуры — передвигаться по тексту
- `Ctrl + O` — **сохранить** (затем Enter для подтверждения имени файла)
- `Ctrl + X` — **выйти**

```bash
# Открыть файл .env для редактирования
nano /app/scooterparts/.env
```

Удалить всё что там есть и вставить следующее (заменить значения в угловых скобках):

```env
# === БАЗА ДАННЫХ ===
# Адрес подключения к PostgreSQL внутри Docker
# "postgres" — имя контейнера с базой (не менять)
DATABASE_URL=postgresql://scooter_user:МОЙ_ПАРОЛЬ_БД@postgres:5432/scooter_shop

# Логин/пароль для создания базы (придумать самому)
POSTGRES_USER=scooter_user
POSTGRES_PASSWORD=МОЙ_ПАРОЛЬ_БД
POSTGRES_DB=scooter_shop

# === БЕЗОПАСНОСТЬ ===
# Секретный ключ для JWT-токенов (сгенерировать ниже)
SECRET_KEY=<СГЕНЕРИРОВАТЬ>
# Пароль для входа в админку сайта
ADMIN_PASSWORD=<ПРИДУМАТЬ НАДЁЖНЫЙ ПАРОЛЬ>

# === ДОМЕН ===
# Список разрешённых источников запросов (точно как указано)
ALLOWED_ORIGINS=https://fmtun.ru,https://www.fmtun.ru
BASE_URL=https://fmtun.ru
PORT=8000

# === ПРОЧЕЕ ===
AUTH_RATE_LIMIT=10
GLOBAL_RATE_LIMIT=120
PAYMENT_CALLBACK_URL=https://fmtun.ru/api/payment/callback
```

**Как сгенерировать SECRET_KEY:**

Не закрывая nano, открыть второе окно PowerShell и подключиться снова по SSH, затем:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```
Скопировать выведенный текст (64 символа) и вставить вместо `<СГЕНЕРИРОВАТЬ>` в .env.

После заполнения: `Ctrl + O` → Enter → `Ctrl + X`

---

### Шаг 8: Запустить приложение

**Что происходит при запуске?**
Команда `docker compose up` читает файл `docker-compose.yml` и запускает два контейнера:
1. **postgres** — база данных PostgreSQL
2. **app** — твой FastAPI-сайт

При первом запуске Docker скачивает образы и собирает контейнеры — это занимает 3-10 минут.

```bash
cd /app/scooterparts

# Запустить в фоне (-d = detached, --build = пересобрать образ)
docker compose up -d --build
```

Ожидаемый результат в конце:
```
✔ Container scooterparts-postgres-1  Started
✔ Container scooterparts-app-1       Started
```

```bash
# Проверить что оба контейнера работают (STATUS должен быть "Up")
docker compose ps
```

```bash
# Смотреть логи приложения в реальном времени
# Выйти из логов: Ctrl + C
docker compose logs -f app
```

В логах должно быть:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

```bash
# Финальный тест — должен вернуть JSON с токеном
curl http://localhost:8000/api/csrf-token
```

Ожидаемый ответ: `{"csrf_token":"..."}` — сайт работает!

**Если что-то пошло не так:**
```bash
# Смотреть подробные логи ошибок
docker compose logs app --tail=50

# Перезапустить
docker compose down
docker compose up -d --build
```

---

### Шаг 8б: Установить Portainer (веб-интерфейс для управления Docker)

**Что такое Portainer?**
Portainer — это сайт который открывается в твоём браузере и позволяет управлять Docker-контейнерами без командной строки. После этого шага ты сможешь:
- Видеть статус сайта и базы (запущены / упали)
- Смотреть логи прямо в браузере
- Перезапускать контейнеры кнопкой
- Обновлять сайт (pull новой версии) через интерфейс

Это делается **один раз** — дальше работа через браузер.

```bash
# Создать хранилище данных Portainer
docker volume create portainer_data

# Запустить Portainer (будет доступен на порту 9000)
docker run -d \
  --name portainer \
  --restart=always \
  -p 9000:9000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest
```

После запуска открыть в браузере: `http://<IP твоего VPS>:9000`

При первом открытии:
1. Придумать логин и пароль администратора Portainer
2. Нажать **"Get Started"** → выбрать **"local"**
3. Ты увидишь свои контейнеры (postgres и app)

> **Важно:** Portainer доступен по IP:9000, не по домену. Это нормально — он только для тебя.

**Что делать в Portainer вместо командной строки:**

| Задача | В командной строке | В Portainer |
|---|---|---|
| Посмотреть логи | `docker compose logs -f app` | Containers → app → Logs |
| Перезапустить | `docker compose restart app` | Containers → app → Restart (кнопка) |
| Остановить | `docker compose down` | Containers → Stop (кнопка) |
| Зайти в базу | `docker compose exec postgres psql ...` | Containers → postgres → Console |
| Обновить сайт | `git pull && docker compose up -d --build` | Через Stacks → Update |

---

### Шаг 9: Настроить Nginx (связать домен с приложением)

**Зачем это нужно?**
Сейчас сайт работает на порту 8000 (http://IP:8000). Но пользователи заходят на fmtun.ru (порт 80/443). Nginx будет принимать запросы на стандартных портах и "проксировать" их на порт 8000 — это называется reverse proxy.

```bash
# Создать конфиг для нашего сайта
nano /etc/nginx/sites-available/fmtun.ru
```

Вставить следующее содержимое (не менять — всё уже правильно настроено):

```nginx
server {
    listen 80;
    server_name fmtun.ru www.fmtun.ru;

    # Максимальный размер загружаемых файлов (для фото товаров)
    client_max_body_size 20M;

    # Все запросы передавать в FastAPI на порт 8000
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Статические файлы (картинки, CSS) отдавать напрямую без FastAPI
    location /static/ {
        alias /app/scooterparts/static/;
        expires 30d;
    }
}
```

Сохранить: `Ctrl + O` → Enter → `Ctrl + X`

```bash
# Подключить конфиг (создать символическую ссылку)
ln -s /etc/nginx/sites-available/fmtun.ru /etc/nginx/sites-enabled/

# Проверить что конфиг без ошибок
nginx -t
```

Ожидаемый ответ:
```
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

```bash
# Применить конфиг
systemctl reload nginx
```

Теперь сайт должен открываться по http://fmtun.ru (но без HTTPS пока).

---

### Шаг 10: Получить SSL-сертификат (включить HTTPS)

**Что такое SSL/HTTPS?**
HTTPS — зашифрованное соединение между браузером и сайтом. Без него браузер показывает "Небезопасно" и многие функции не работают. SSL-сертификат — это файл который подтверждает подлинность сайта.

Мы используем **Let's Encrypt** — бесплатный сервис сертификатов, которым пользуются миллионы сайтов. Программа `certbot` сама получает сертификат и обновляет его каждые 90 дней автоматически.

> **Важно:** Сначала убедиться что DNS обновился и домен ведёт на сервер. Проверить: открыть браузер и зайти на http://fmtun.ru — должен открыться сайт (пусть без HTTPS). Если не открывается — подождать ещё 1-2 часа.

```bash
# Получить сертификат для домена и www.домена
# certbot сам обновит конфиг nginx для HTTPS
certbot --nginx -d fmtun.ru -d www.fmtun.ru
```

Certbot задаст вопросы:
1. `Enter email address` — ввести свой email (для уведомлений о продлении)
2. `(A)gree/(C)ancel` — написать `A`
3. `(Y)es/(N)o` — можно написать `N` (отказаться от рассылки)

После успешного завершения появится:
```
Successfully deployed certificate for fmtun.ru
```

Certbot сам отредактирует nginx-конфиг и добавит HTTPS.

```bash
# Убедиться что автопродление работает
certbot renew --dry-run
```

Ожидаемый ответ: `Congratulations, all simulated renewals succeeded`

Теперь сайт доступен по **https://fmtun.ru**.

---

### Шаг 11: Настроить автозапуск при перезагрузке

**Зачем?**
Если сервер перезагрузится (плановое обслуживание, обновление ядра), Docker и приложение не запустятся сами — сайт упадёт. Systemd — это менеджер служб Linux, который автоматически запускает нужные программы при старте.

```bash
# Создать файл службы
nano /etc/systemd/system/scooterparts.service
```

Вставить:

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

Сохранить: `Ctrl + O` → Enter → `Ctrl + X`

```bash
# Зарегистрировать службу и запустить
systemctl enable scooterparts
systemctl start scooterparts

# Проверить что служба запущена
systemctl status scooterparts
```

Ожидаемый ответ: `Active: active (running)`

Всё — сайт будет автоматически запускаться при каждой перезагрузке сервера.

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
