# 🔒 ФИНАЛЬНЫЙ АУДИТ v11 — Все исправления

## ✅ КРИТИЧЕСКИЕ ОШИБКИ (ИСПРАВЛЕНЫ)

### 🔴 Bug-1: delete_my_account — крэш при удалении аккаунта
**Файл:** `backend/main.py` (строки ~1228-1237)  
**Проблема:** `int(user_id)` на UUID-строке → `ValueError: invalid literal for int()`  
Пользователь не мог удалить свой аккаунт — эндпоинт всегда падал с 500.  
**Исправление:** Заменено `int(user_id)` → `user_id` (asyncpg передаёт UUID как строку через `$1::uuid`)

---

### 🔴 Bug-2: Корзина — дубликаты при specification_id IS NULL
**Файл:** `backend/main.py` (init_database + add_to_cart)  
**Проблема:** `ON CONFLICT (user_id, product_id, specification_id)` не работает с NULL в PostgreSQL —  
уникальный индекс не видит NULL = NULL, поэтому один товар добавлялся многократно.  
**Исправление:** Заменено на два **partial unique index**:
- `cart_items_uniq_no_spec` WHERE specification_id IS NULL
- `cart_items_uniq_with_spec` WHERE specification_id IS NOT NULL  
И соответствующие отдельные `ON CONFLICT` ветки в `add_to_cart`.

---

## ✅ ОШИБКИ БЕЗОПАСНОСТИ (ИСПРАВЛЕНЫ)

### 🟠 Sec-1: Logout не очищал CSRF-cookie
**Файл:** `backend/main.py` → `/api/logout`  
**Проблема:** После выхода `csrf_token` оставался в браузере — потенциально мог быть переиспользован.  
**Исправление:** `/api/logout` теперь удаляет оба cookie: `access_token` + `csrf_token`

---

### 🟠 Sec-2: Незащищённый endpoint `/api/admin/add-product`
**Файл:** `backend/main.py`  
**Проблема:** Дублирующий роут возвращал страницу добавления товара без проверки is_admin.  
**Исправление:** Роут удалён. Страница доступна только через `/admin/add-product` (с серверной проверкой)

---

### 🟡 Sec-3: python-jose конфликтует с PyJWT
**Файл:** `requirements.txt`  
**Проблема:** Обе библиотеки занимают namespace `jwt.*` — в зависимости от порядка установки  
поведение `import jwt` непредсказуемо. Код использует PyJWT.  
**Исправление:** `python-jose[cryptography]` закомментирован в requirements.txt

---

## ✅ КАЧЕСТВО КОДА (ИСПРАВЛЕНО)

### 🟡 Code-1: Повторный import logging / logger внутри payment_callback
**Файл:** `backend/main.py`  
**Проблема:** `import logging` + `logger = logging.getLogger()` продублированы внутри функции.  
Logger уже определён на уровне модуля (строка ~39).  
**Исправление:** Удалены дублирующие строки.

### 🟡 Code-2: import base64 внутри функций (5 мест)
**Файл:** `backend/main.py`  
**Проблема:** `import base64` повторялось внутри каждой функции работающей с изображениями.  
**Исправление:** Перенесено в секцию top-level imports (строка 22).  
Аналогично исправлен `import json as _json` внутри `get_user_orders`.

---

## ⚠️ ТРЕБУЕТ РУЧНОГО ДЕЙСТВИЯ (ЗАДОКУМЕНТИРОВАНО)

### 📋 Manual-1: SRI-хеши для CDN скриптов (A-9)
**Файл:** все `.html` шаблоны — 15 `<script src="https://cdn...">` тегов  
**Проблема:** Нет атрибутов `integrity="sha384-..."` — GSAP/Lenis могут быть подменены CDN.  
**Действие:** Запустить `python3 generate_sri.py` (требует интернет):
```bash
pip install requests
python3 generate_sri.py
# Скрипт скачает все 8 CDN-файлов и выведет готовые теги с integrity=
```

### 📋 Manual-2: Плейсхолдеры в legal.html (B-5)
**Файл:** `templates/legal.html` строки 698, 701  
Заполнить до деплоя:  
- `[ИНН]` → ИНН организации  
- `[НОМЕР УВЕДОМЛЕНИЯ РКН]` → номер из реестра pd.rkn.gov.ru  
- `[ДАТА]` → дата регистрации в РКН

### 📋 Manual-3: Переменные окружения
Заполнить `.env` на основе `.env.example`:
```
SECRET_KEY=<python -c "import secrets; print(secrets.token_hex(32))">
ADMIN_PASSWORD=<сильный пароль>
DATABASE_URL=postgresql://...
ALLOWED_ORIGINS=https://your-domain.com
ENVIRONMENT=production
```

---

## ✅ СТАТУС ВСЕХ ПУНКТОВ АУДИТА

| # | Пункт | Статус |
|---|-------|--------|
| A-1 | CSRF токен | ✅ Работает |
| A-2 | CSP nonce | ✅ Работает (все шаблоны) |
| A-3 | Дублирующийся маршрут | ✅ Исправлен |
| B-1 | Логирование согласий | ✅ Работает |
| A-4 | Admin login pbkdf2-хеш | ✅ Работает |
| A-5 | CSP для CDN/шрифтов | ✅ Работает |
| A-6 | Публичные эндпоинты | ✅ Закрыты |
| A-7 | update_specification диск→base64 | ✅ Работает |
| B-2/B-3 | Cookie баннер v2.0 | ✅ Работает |
| A-8 | Лимит тела запроса | ✅ BodySizeLimitMiddleware |
| A-9 | SRI для CDN | ⚠️ Нужен generate_sri.py |
| A-10 | samesite strict | ✅ Работает |
| A-11 | Refresh token | ✅ Работает |
| B-4 | Очистка согласий | ✅ Работает |
| B-5 | Плейсхолдеры РКН | ⚠️ Заполнить вручную |
| B-6 | Удаление аккаунта | ✅ **ИСПРАВЛЕН (UUID bug)** |
| B-7 | session_id в localStorage | ✅ Работает |
| B-8 | Баннер на всех страницах | ✅ Все 9 шаблонов |
| — | Logout CSRF bug | ✅ **ИСПРАВЛЕН** |
| — | Cart NULL unique | ✅ **ИСПРАВЛЕН** |
| — | python-jose conflict | ✅ **ИСПРАВЛЕН** |
| — | Duplicate logger | ✅ **ИСПРАВЛЕН** |
| — | Inline imports | ✅ **ИСПРАВЛЕН** |
