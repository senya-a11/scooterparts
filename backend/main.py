# backend/main.py  ·  ScooterParts v5.0
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import os
from pathlib import Path
import jwt
import hashlib
import uuid
import hmac
import json
from uuid import uuid4
import secrets
import shutil
import aiofiles

import asyncpg
from asyncpg.pool import Pool
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi.templating import Jinja2Templates
from fastapi import Request

# ========== НАСТРОЙКА ПУТЕЙ ==========
BASE_DIR = Path(__file__).parent.parent

templates_path = BASE_DIR / "templates"
if not templates_path.exists():
    templates_path = Path("/app/templates")
    if not templates_path.exists():
        templates_path = BASE_DIR / "templates"
        templates_path.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_path))

STATIC_DIR = BASE_DIR / "static"
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "images").mkdir(exist_ok=True)
(STATIC_DIR / "favicon").mkdir(exist_ok=True)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_USERNAME = "admin"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/scooter_shop")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Конфиг платёжной системы (заполните переменные окружения)
PAYMENT_API_KEY    = os.getenv("PAYMENT_API_KEY", "")
PAYMENT_SHOP_ID    = os.getenv("PAYMENT_SHOP_ID", "")
PAYMENT_SECRET_KEY = os.getenv("PAYMENT_SECRET_KEY", "")
PAYMENT_CALLBACK_URL = os.getenv("PAYMENT_CALLBACK_URL", "https://your-domain.com/api/payment/callback")


# ========== МОДЕЛИ ==========
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None
    privacy_accepted: bool = False

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError('Имя пользователя должно содержать минимум 3 символа')
        if len(v) > 50:
            raise ValueError('Имя пользователя должно содержать не более 50 символов')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Пароль должен содержать минимум 6 символов')
        return v

    @field_validator('privacy_accepted')
    @classmethod
    def validate_privacy(cls, v: bool) -> bool:
        if not v:
            raise ValueError('Необходимо принять политику конфиденциальности')
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class CartItem(BaseModel):
    product_id: int
    quantity: int


class CartUpdate(BaseModel):
    product_id: int
    quantity: int


class AdminLogin(BaseModel):
    username: str
    password: str


class ProductCreate(BaseModel):
    name: str
    category: str
    price: float
    description: str
    stock: int = 0
    featured: bool = False


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    stock: Optional[int] = None
    featured: Optional[bool] = None


class CategoryCreate(BaseModel):
    slug: str
    name: str
    emoji: str = "📦"
    description: Optional[str] = ""


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    emoji: Optional[str] = None
    description: Optional[str] = None


class OrderCreate(BaseModel):
    delivery_address: Optional[str] = None
    comment: Optional[str] = None


class PaymentCreate(BaseModel):
    order_id: int
    amount: float
    currency: str = "RUB"


# ========== АУТЕНТИФИКАЦИЯ ==========
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM  = "HS256"
security   = HTTPBearer()


class PasswordHasher:
    @staticmethod
    def get_password_hash(password: str) -> str:
        salt       = secrets.token_hex(16)
        iterations = 100000
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), iterations)
        return f"pbkdf2_sha256:{iterations}:{salt}:{key.hex()}"

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        try:
            parts = hashed_password.split(':')
            if len(parts) != 4:
                return False
            algorithm, iterations_str, salt, stored_hash = parts
            if algorithm != 'pbkdf2_sha256':
                return False
            key = hashlib.pbkdf2_hmac(
                'sha256', plain_password.encode(), salt.encode(), int(iterations_str)
            )
            return hmac.compare_digest(key.hex(), stored_hash)
        except Exception:
            return False


hasher = PasswordHasher()


def create_access_token(data: dict):
    to_encode = data.copy()
    for k, v in to_encode.items():
        if isinstance(v, (uuid.UUID, datetime)):
            to_encode[k] = str(v)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("user_id")
    except Exception:
        return None


def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("is_admin"):
            raise HTTPException(status_code=403, detail="Доступ запрещён")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Не авторизован")


# ========== БАЗА ДАННЫХ ==========
DEFAULT_CATEGORIES = [
    ("batteries",  "Аккумуляторы", "🔋", "Литий-ионные аккумуляторы и зарядные устройства"),
    ("motors",     "Моторы",       "⚙️", "Мотор-колёса, контроллеры и двигатели"),
    ("electronics","Электроника",  "📱", "Дисплеи, контроллеры, электроника"),
    ("brakes",     "Тормоза",      "🛑", "Тормозные диски, колодки, тросы"),
    ("tires",      "Колёса",       "🛞", "Покрышки, камеры, обода"),
    ("accessories","Аксессуары",   "🔧", "Ручки, подножки, зеркала и прочее"),
]


class Database:
    def __init__(self):
        self.pool: Optional[Pool] = None

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(
                DATABASE_URL, min_size=1, max_size=10, command_timeout=60
            )
            await self.init_database()
            print("✅ База данных подключена")
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    async def init_database(self):
        async with self.pool.acquire() as conn:
            # Пользователи
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    full_name VARCHAR(100) NOT NULL,
                    phone VARCHAR(20),
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_admin BOOLEAN DEFAULT FALSE,
                    privacy_accepted BOOLEAN DEFAULT FALSE,
                    privacy_accepted_at TIMESTAMP
                )
            ''')

            # Категории товаров (CRUD)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    slug VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    emoji VARCHAR(10) DEFAULT '📦',
                    description TEXT DEFAULT '',
                    sort_order INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Товары
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    description TEXT NOT NULL,
                    image_url VARCHAR(500) NOT NULL,
                    stock INTEGER DEFAULT 0,
                    featured BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Корзина
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS cart_items (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL,
                    product_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, product_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
                )
            ''')

            # Заказы
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL,
                    status VARCHAR(30) DEFAULT 'pending',
                    total_amount DECIMAL(12,2) NOT NULL,
                    delivery_address TEXT,
                    comment TEXT,
                    payment_status VARCHAR(30) DEFAULT 'pending',
                    payment_id VARCHAR(200),
                    payment_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            ''')

            # Позиции заказа
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS order_items (
                    id SERIAL PRIMARY KEY,
                    order_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    product_name VARCHAR(200) NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    quantity INTEGER NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
                )
            ''')

            # --- Начальные категории ---
            for slug, name, emoji, desc in DEFAULT_CATEGORIES:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM categories WHERE slug=$1)", slug
                )
                if not exists:
                    await conn.execute(
                        "INSERT INTO categories (slug, name, emoji, description) VALUES ($1,$2,$3,$4)",
                        slug, name, emoji, desc
                    )

            # --- Демо-пользователь ---
            if not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM users WHERE username='demo')"):
                await conn.execute(
                    "INSERT INTO users (id,username,email,full_name,phone,password_hash,privacy_accepted) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                    str(uuid4()), 'demo', 'demo@scooterparts.ru', 'Демо Пользователь',
                    '+79991234567', hasher.get_password_hash("demo123"), True
                )

            # --- Админ ---
            if not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM users WHERE username='admin')"):
                await conn.execute(
                    "INSERT INTO users (id,username,email,full_name,password_hash,is_admin,privacy_accepted) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                    str(uuid4()), 'admin', 'admin@scooterparts.ru', 'Администратор',
                    hasher.get_password_hash(ADMIN_PASSWORD), True, True
                )

            # --- Демо-товары ---
            if not await conn.fetchval("SELECT COUNT(*) FROM products"):
                demo = [
                    ("Аккумулятор Premium 36V 15Ah","batteries",16500.00,"Высокоёмкий литий-ионный аккумулятор с BMS. Гарантия 24 мес.","/static/images/battery.jpg",8,True),
                    ("Мотор-колесо Ultra 500W","motors",12500.00,"Бесщёточный мотор с прямым приводом. Макс. скорость 45 км/ч.","/static/images/motor.jpg",5,True),
                    ("Контроллер Smart 36V","electronics",4900.00,"Интеллектуальный контроллер с Bluetooth и мобильным приложением.","/static/images/controller.jpg",15,False),
                    ("Дисплей Color LCD","electronics",3200.00,"Цветной LCD дисплей с подсветкой и индикацией всех параметров.","/static/images/display.jpg",12,True),
                    ("Тормозные диски Premium","brakes",2200.00,"Вентилируемые тормозные диски из нержавеющей стали.","/static/images/brakes.jpg",25,False),
                    ("Колесо 10\" All-Terrain","tires",1800.00,"Пневматическое колесо для бездорожья с усиленными стенками.","/static/images/wheel.jpg",20,False),
                    ("Тормозные колодки Premium","brakes",1200.00,"Керамические тормозные колодки для дисковых тормозов.","/static/images/brake-pads.jpg",30,True),
                    ("Руль алюминиевый","accessories",2500.00,"Алюминиевый руль с резиновыми накладками.","/static/images/handlebar.jpg",15,False),
                ]
                for p in demo:
                    await conn.execute(
                        "INSERT INTO products (name,category,price,description,image_url,stock,featured) VALUES ($1,$2,$3,$4,$5,$6,$7)",
                        *p
                    )

            print("✅ БД инициализирована")


db = Database()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


# ========== ПРИЛОЖЕНИЕ ==========
app = FastAPI(title="ScooterParts API v5", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==========================================
# ========== СТРАНИЦЫ ==========
# ==========================================

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/products")
async def products_page(request: Request):
    return templates.TemplateResponse("products.html", {"request": request})

@app.get("/admin")
async def admin_panel(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/admin/add-product")
async def admin_add_product_page(request: Request):
    return templates.TemplateResponse("add_product.html", {"request": request})

@app.get("/privacy-policy")
async def privacy_policy_page(request: Request):
    return templates.TemplateResponse("legal.html", {"request": request, "doc_type": "privacy"})

@app.get("/terms")
async def terms_page(request: Request):
    return templates.TemplateResponse("legal.html", {"request": request, "doc_type": "terms"})

@app.get("/offer")
async def offer_page(request: Request):
    return templates.TemplateResponse("legal.html", {"request": request, "doc_type": "offer"})

@app.get("/returns")
async def returns_page(request: Request):
    return templates.TemplateResponse("legal.html", {"request": request, "doc_type": "returns"})


# ==========================================
# ========== AUTH API ==========
# ==========================================

@app.post("/api/register")
async def register(user_data: UserRegister):
    if not user_data.privacy_accepted:
        raise HTTPException(status_code=400, detail="Необходимо принять политику конфиденциальности")
    try:
        async with db.pool.acquire() as conn:
            if await conn.fetchval("SELECT EXISTS(SELECT 1 FROM users WHERE username=$1)", user_data.username):
                raise HTTPException(status_code=400, detail="Имя пользователя уже занято")
            if await conn.fetchval("SELECT EXISTS(SELECT 1 FROM users WHERE email=$1)", user_data.email):
                raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

            user_id = str(uuid4())
            password_hash = hasher.get_password_hash(user_data.password)
            await conn.execute(
                "INSERT INTO users (id,username,email,full_name,phone,password_hash,privacy_accepted,privacy_accepted_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                user_id, user_data.username, user_data.email, user_data.full_name,
                user_data.phone, password_hash, True, datetime.utcnow()
            )
            return {"message": "Аккаунт создан успешно", "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/login")
async def login(login_data: UserLogin):
    try:
        async with db.pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT id,username,email,full_name,password_hash,is_admin FROM users WHERE username=$1",
                login_data.username
            )
            if not user or not hasher.verify_password(login_data.password, user['password_hash']):
                raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")

            user_id = str(user['id'])
            token = create_access_token({"user_id": user_id, "is_admin": user['is_admin']})
            return {
                "access_token": token,
                "token_type": "bearer",
                "user": {"id": user_id, "username": user['username'], "email": user['email'],
                         "full_name": user['full_name'], "is_admin": user['is_admin']}
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/me")
async def get_me(user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT id,username,email,full_name,phone,is_admin,created_at FROM users WHERE id=$1",
                user_id
            )
            if not user:
                raise HTTPException(status_code=404, detail="Пользователь не найден")
            d = dict(user)
            d['id'] = str(d['id'])
            if isinstance(d.get('created_at'), datetime):
                d['created_at'] = d['created_at'].isoformat()
            return d
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ========== CATEGORIES API ==========
# ==========================================

@app.get("/api/categories")
async def get_categories():
    """Получить все категории с количеством товаров"""
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT c.slug, c.name, c.emoji, c.description, c.sort_order,
                       COUNT(p.id) as count
                FROM categories c
                LEFT JOIN products p ON p.category = c.slug
                GROUP BY c.slug, c.name, c.emoji, c.description, c.sort_order
                ORDER BY c.sort_order, c.name
            ''')
            return {"categories": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/categories")
async def create_category(cat: CategoryCreate, admin=Depends(verify_admin)):
    """Создать новую категорию"""
    try:
        async with db.pool.acquire() as conn:
            if await conn.fetchval("SELECT EXISTS(SELECT 1 FROM categories WHERE slug=$1)", cat.slug):
                raise HTTPException(status_code=400, detail="Категория с таким slug уже существует")
            max_order = await conn.fetchval("SELECT COALESCE(MAX(sort_order),0) FROM categories")
            await conn.execute(
                "INSERT INTO categories (slug,name,emoji,description,sort_order) VALUES ($1,$2,$3,$4,$5)",
                cat.slug, cat.name, cat.emoji, cat.description or "", max_order + 1
            )
            return {"message": "Категория создана", "slug": cat.slug}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/categories/{slug}")
async def update_category(slug: str, cat: CategoryUpdate, admin=Depends(verify_admin)):
    """Обновить категорию"""
    try:
        async with db.pool.acquire() as conn:
            existing = await conn.fetchrow("SELECT * FROM categories WHERE slug=$1", slug)
            if not existing:
                raise HTTPException(status_code=404, detail="Категория не найдена")
            new_name  = cat.name  if cat.name  is not None else existing['name']
            new_emoji = cat.emoji if cat.emoji is not None else existing['emoji']
            new_desc  = cat.description if cat.description is not None else existing['description']
            await conn.execute(
                "UPDATE categories SET name=$1, emoji=$2, description=$3 WHERE slug=$4",
                new_name, new_emoji, new_desc, slug
            )
            return {"message": "Категория обновлена"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/categories/{slug}")
async def delete_category(slug: str, admin=Depends(verify_admin)):
    """Удалить категорию (только если нет товаров)"""
    try:
        async with db.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM products WHERE category=$1", slug)
            if count > 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Нельзя удалить категорию: в ней {count} товар(ов). Сначала переместите или удалите товары."
                )
            result = await conn.execute("DELETE FROM categories WHERE slug=$1", slug)
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Категория не найдена")
            return {"message": "Категория удалена"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ========== PRODUCTS API ==========
# ==========================================

@app.get("/api/products")
async def get_products(category: Optional[str] = None, featured: Optional[bool] = None,
                       search: Optional[str] = None, limit: Optional[int] = None):
    try:
        async with db.pool.acquire() as conn:
            query  = "SELECT * FROM products WHERE 1=1"
            params = []
            i = 1
            if category:
                query += f" AND category = ${i}"; params.append(category); i += 1
            if featured is not None:
                query += f" AND featured = ${i}"; params.append(featured); i += 1
            if search:
                query += f" AND (LOWER(name) LIKE ${i} OR LOWER(description) LIKE ${i})"; params.append(f"%{search.lower()}%"); i += 1
            query += " ORDER BY id"
            if limit:
                query += f" LIMIT ${i}"; params.append(limit)

            rows = await conn.fetch(query, *params)
            result = []
            for r in rows:
                d = dict(r)
                d['price'] = float(d['price'])
                if isinstance(d.get('created_at'), datetime):
                    d['created_at'] = d['created_at'].isoformat()
                result.append(d)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/products/{product_id}")
async def get_product(product_id: int):
    try:
        async with db.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM products WHERE id=$1", product_id)
            if not row:
                raise HTTPException(status_code=404, detail="Товар не найден")
            d = dict(row)
            d['price'] = float(d['price'])
            return d
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ========== CART API ==========
# ==========================================

@app.get("/api/cart")
async def get_cart(user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            items = await conn.fetch('''
                SELECT ci.product_id, ci.quantity,
                       p.name, p.category, p.price, p.description, p.image_url, p.stock
                FROM cart_items ci JOIN products p ON ci.product_id = p.id
                WHERE ci.user_id = $1 ORDER BY ci.added_at DESC
            ''', user_id)
            total = 0
            result = []
            for item in items:
                item_total = float(item['price']) * item['quantity']
                total += item_total
                result.append({
                    "product_id": item['product_id'],
                    "quantity": item['quantity'],
                    "product": {
                        "id": item['product_id'], "name": item['name'],
                        "category": item['category'], "price": float(item['price']),
                        "description": item['description'], "image_url": item['image_url'],
                        "stock": item['stock']
                    },
                    "item_total": item_total
                })
            return {"items": result, "total": total, "items_count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cart")
async def add_to_cart(cart_item: CartUpdate, user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    if cart_item.quantity <= 0:
        raise HTTPException(status_code=400, detail="Количество должно быть больше 0")
    try:
        async with db.pool.acquire() as conn:
            product = await conn.fetchrow("SELECT id, stock FROM products WHERE id=$1", cart_item.product_id)
            if not product:
                raise HTTPException(status_code=404, detail="Товар не найден")
            if product['stock'] < cart_item.quantity:
                raise HTTPException(status_code=400, detail="Недостаточно товара на складе")
            await conn.execute('''
                INSERT INTO cart_items (user_id, product_id, quantity)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, product_id) DO UPDATE SET quantity = EXCLUDED.quantity
            ''', user_id, cart_item.product_id, cart_item.quantity)
            return {"message": "Товар добавлен в корзину"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cart/{product_id}")
async def remove_from_cart(product_id: int, user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            r = await conn.execute(
                "DELETE FROM cart_items WHERE user_id=$1 AND product_id=$2", user_id, product_id
            )
            if r == "DELETE 0":
                raise HTTPException(status_code=404, detail="Товар не найден в корзине")
            return {"message": "Товар удалён из корзины"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cart")
async def clear_cart(user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM cart_items WHERE user_id=$1", user_id)
            return {"message": "Корзина очищена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ========== ORDERS API ==========
# ==========================================

@app.post("/api/orders")
async def create_order(order_data: OrderCreate, user_id: str = Depends(get_current_user)):
    """Создать заказ из корзины"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            cart_items = await conn.fetch('''
                SELECT ci.product_id, ci.quantity, p.name, p.price, p.stock
                FROM cart_items ci JOIN products p ON ci.product_id = p.id
                WHERE ci.user_id = $1
            ''', user_id)

            if not cart_items:
                raise HTTPException(status_code=400, detail="Корзина пуста")

            total = sum(float(i['price']) * i['quantity'] for i in cart_items)

            order_id = await conn.fetchval('''
                INSERT INTO orders (user_id, total_amount, delivery_address, comment, status, payment_status)
                VALUES ($1, $2, $3, $4, 'pending', 'pending')
                RETURNING id
            ''', user_id, total, order_data.delivery_address, order_data.comment)

            for item in cart_items:
                await conn.execute('''
                    INSERT INTO order_items (order_id, product_id, product_name, price, quantity)
                    VALUES ($1, $2, $3, $4, $5)
                ''', order_id, item['product_id'], item['name'], float(item['price']), item['quantity'])
                await conn.execute(
                    "UPDATE products SET stock = stock - $1 WHERE id = $2",
                    item['quantity'], item['product_id']
                )

            await conn.execute("DELETE FROM cart_items WHERE user_id=$1", user_id)

            return {
                "message": "Заказ создан",
                "order_id": order_id,
                "total": total,
                "status": "pending"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders")
async def get_user_orders(user_id: str = Depends(get_current_user)):
    """Получить заказы пользователя"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            orders = await conn.fetch('''
                SELECT id, status, total_amount, payment_status, created_at, delivery_address
                FROM orders WHERE user_id=$1 ORDER BY created_at DESC
            ''', user_id)
            result = []
            for o in orders:
                d = dict(o)
                d['total_amount'] = float(d['total_amount'])
                if isinstance(d.get('created_at'), datetime):
                    d['created_at'] = d['created_at'].isoformat()
                result.append(d)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders/active-count")
async def get_active_orders_count():
    """Публичный счётчик активных заказов для плашки в каталоге"""
    try:
        async with db.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM orders WHERE status NOT IN ('delivered','cancelled')"
            )
            return {"count": count or 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ========== PAYMENT API (заглушка) ==========
# ==========================================

@app.post("/api/payment/create")
async def create_payment(payment: PaymentCreate, user_id: str = Depends(get_current_user)):
    """
    Заглушка платёжной интеграции.
    Для подключения реального эквайринга замените этот блок:
    - ЮKassa: https://yookassa.ru/developers/api
    - Тинькофф: https://www.tinkoff.ru/kassa/develop/api/
    - Stripe: https://stripe.com/docs/api

    Переменные окружения:
      PAYMENT_API_KEY=...
      PAYMENT_SHOP_ID=...
      PAYMENT_SECRET_KEY=...
      PAYMENT_CALLBACK_URL=https://your-domain.com/api/payment/callback
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            order = await conn.fetchrow(
                "SELECT id, total_amount, payment_status FROM orders WHERE id=$1 AND user_id=$2",
                payment.order_id, user_id
            )
            if not order:
                raise HTTPException(status_code=404, detail="Заказ не найден")
            if order['payment_status'] == 'paid':
                raise HTTPException(status_code=400, detail="Заказ уже оплачен")

            # ── ЗДЕСЬ ИНТЕГРИРУЙТЕ РЕАЛЬНЫЙ ЭКВАЙРИНГ ──
            # Пример для ЮKassa:
            # import yookassa
            # yookassa.Configuration.account_id = PAYMENT_SHOP_ID
            # yookassa.Configuration.secret_key = PAYMENT_SECRET_KEY
            # payment_obj = yookassa.Payment.create({
            #     "amount": {"value": str(payment.amount), "currency": "RUB"},
            #     "confirmation": {"type": "redirect", "return_url": PAYMENT_CALLBACK_URL},
            #     "capture": True,
            #     "description": f"Заказ #{payment.order_id}",
            # })
            # payment_url = payment_obj.confirmation.confirmation_url
            # payment_id  = payment_obj.id

            # ЗАГЛУШКА — возвращаем тестовые данные
            payment_id  = f"stub_{uuid4().hex[:16]}"
            payment_url = f"/payment-stub?order_id={payment.order_id}&amount={payment.amount}"

            await conn.execute(
                "UPDATE orders SET payment_id=$1, payment_url=$2, payment_status='waiting' WHERE id=$3",
                payment_id, payment_url, payment.order_id
            )

            return {
                "payment_id":  payment_id,
                "payment_url": payment_url,
                "amount":      payment.amount,
                "currency":    payment.currency,
                "status":      "waiting",
                "note":        "⚠️ Это заглушка. Подключите реальный эквайринг в backend/main.py"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/payment/callback")
async def payment_callback(request: Request):
    """
    Webhook / колбэк от платёжной системы.
    Реализуйте проверку подписи и обновление статуса заказа.
    """
    try:
        body = await request.json()
        print(f"💳 Payment callback: {body}")

        # TODO: проверьте подпись от платёжной системы
        # TODO: извлеките order_id и status из body
        # TODO: обновите orders.payment_status и orders.status

        return {"status": "ok"}
    except Exception as e:
        print(f"❌ Payment callback error: {e}")
        return {"status": "error", "detail": str(e)}


@app.get("/payment-stub")
async def payment_stub_page(request: Request, order_id: int = 0, amount: float = 0):
    """Страница-заглушка оплаты (удалить после подключения реального эквайринга)"""
    return templates.TemplateResponse("payment_stub.html", {
        "request": request, "order_id": order_id, "amount": amount
    })


# ==========================================
# ========== ADMIN API ==========
# ==========================================

@app.post("/api/admin/login")
async def admin_login(login_data: AdminLogin):
    if login_data.username != ADMIN_USERNAME or login_data.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Неверные данные для входа")
    try:
        async with db.pool.acquire() as conn:
            admin = await conn.fetchrow("SELECT id FROM users WHERE username=$1", ADMIN_USERNAME)
            if not admin:
                raise HTTPException(status_code=401, detail="Администратор не найден")
            token = create_access_token({"user_id": str(admin['id']), "username": ADMIN_USERNAME, "is_admin": True})
            return {"access_token": token, "token_type": "bearer", "user": {"username": ADMIN_USERNAME, "is_admin": True}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/stats")
async def get_admin_stats(admin=Depends(verify_admin)):
    try:
        async with db.pool.acquire() as conn:
            return {
                "users": {
                    "total": await conn.fetchval("SELECT COUNT(*) FROM users"),
                    "with_carts": await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM cart_items"),
                },
                "products": {
                    "total": await conn.fetchval("SELECT COUNT(*) FROM products"),
                    "in_stock": await conn.fetchval("SELECT COUNT(*) FROM products WHERE stock>0"),
                    "out_of_stock": await conn.fetchval("SELECT COUNT(*) FROM products WHERE stock=0"),
                    "featured": await conn.fetchval("SELECT COUNT(*) FROM products WHERE featured=true"),
                },
                "orders": {
                    "total": await conn.fetchval("SELECT COUNT(*) FROM orders"),
                    "active": await conn.fetchval("SELECT COUNT(*) FROM orders WHERE status NOT IN ('delivered','cancelled')"),
                    "pending_payment": await conn.fetchval("SELECT COUNT(*) FROM orders WHERE payment_status='pending'"),
                    "revenue": float(await conn.fetchval("SELECT COALESCE(SUM(total_amount),0) FROM orders WHERE payment_status='paid'") or 0),
                },
                "categories": {
                    "total": await conn.fetchval("SELECT COUNT(*) FROM categories"),
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/orders")
async def get_admin_orders(admin=Depends(verify_admin), status: Optional[str] = None, limit: int = 50):
    try:
        async with db.pool.acquire() as conn:
            q = "SELECT o.*, u.username, u.email FROM orders o LEFT JOIN users u ON o.user_id=u.id WHERE 1=1"
            params = []
            if status:
                q += f" AND o.status = ${len(params)+1}"; params.append(status)
            q += f" ORDER BY o.created_at DESC LIMIT ${len(params)+1}"; params.append(limit)
            rows = await conn.fetch(q, *params)
            result = []
            for r in rows:
                d = dict(r)
                d['total_amount'] = float(d['total_amount'])
                d['user_id'] = str(d['user_id']) if d.get('user_id') else None
                if isinstance(d.get('created_at'), datetime): d['created_at'] = d['created_at'].isoformat()
                if isinstance(d.get('updated_at'), datetime): d['updated_at'] = d['updated_at'].isoformat()
                result.append(d)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/orders/{order_id}/status")
async def update_order_status(order_id: int, body: dict, admin=Depends(verify_admin)):
    new_status = body.get("status")
    valid = ['pending','confirmed','processing','shipped','delivered','cancelled']
    if new_status not in valid:
        raise HTTPException(status_code=400, detail=f"Недопустимый статус. Допустимые: {valid}")
    try:
        async with db.pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE orders SET status=$1, updated_at=NOW() WHERE id=$2", new_status, order_id
            )
            if r == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Заказ не найден")
            return {"message": "Статус обновлён", "status": new_status}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/products")
async def create_product(request: Request, admin=Depends(verify_admin)):
    try:
        form = await request.form()
        name     = str(form.get("name","")).strip()
        category = str(form.get("category","")).strip()
        price    = float(form.get("price",0))
        desc     = str(form.get("description","")).strip()
        stock    = int(form.get("stock",0))
        featured = str(form.get("featured","false")).lower() == "true"
        image_url = str(form.get("image_url","")).strip()
        image_file = form.get("image_file")

        if not name or len(name) < 3:
            raise HTTPException(status_code=400, detail="Название слишком короткое (мин. 3 символа)")
        if not category:
            raise HTTPException(status_code=400, detail="Категория обязательна")
        if price <= 0:
            raise HTTPException(status_code=400, detail="Цена должна быть больше 0")
        if not desc or len(desc) < 10:
            raise HTTPException(status_code=400, detail="Описание слишком короткое (мин. 10 символов)")

        final_image = "/static/images/product_default.jpg"

        if image_file and isinstance(image_file, UploadFile) and image_file.filename:
            ext = Path(image_file.filename).suffix.lower()
            if ext not in ['.jpg','.jpeg','.png','.gif','.webp']:
                raise HTTPException(status_code=400, detail="Недопустимый формат файла")
            fname = f"{uuid4().hex}{ext}"
            fpath = UPLOAD_DIR / fname
            async with aiofiles.open(fpath, 'wb') as buf:
                await buf.write(await image_file.read())
            final_image = f"/static/uploads/{fname}"
        elif image_url:
            final_image = image_url

        async with db.pool.acquire() as conn:
            row = await conn.fetchrow('''
                INSERT INTO products (name,category,price,description,image_url,stock,featured)
                VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *
            ''', name, category, price, desc, final_image, stock, featured)
            d = dict(row)
            d['price'] = float(d['price'])
            return {"success": True, "message": "Товар добавлен!", "product": d}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/products/{product_id}")
async def update_product(product_id: int, request: Request, admin=Depends(verify_admin)):
    try:
        form = await request.form()
        async with db.pool.acquire() as conn:
            existing = await conn.fetchrow("SELECT * FROM products WHERE id=$1", product_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Товар не найден")

            name     = str(form.get("name", existing['name'])).strip()
            category = str(form.get("category", existing['category'])).strip()
            price    = float(form.get("price", existing['price']))
            desc     = str(form.get("description", existing['description'])).strip()
            stock    = int(form.get("stock", existing['stock']))
            featured = str(form.get("featured", str(existing['featured']))).lower() == "true"
            image_url = str(form.get("image_url","")).strip()
            image_file = form.get("image_file")

            final_image = existing['image_url']
            if image_file and isinstance(image_file, UploadFile) and image_file.filename:
                ext = Path(image_file.filename).suffix.lower()
                if ext not in ['.jpg','.jpeg','.png','.gif','.webp']:
                    raise HTTPException(status_code=400, detail="Недопустимый формат файла")
                fname = f"{uuid4().hex}{ext}"
                async with aiofiles.open(UPLOAD_DIR / fname, 'wb') as buf:
                    await buf.write(await image_file.read())
                final_image = f"/static/uploads/{fname}"
            elif image_url:
                final_image = image_url

            await conn.execute('''
                UPDATE products SET name=$1,category=$2,price=$3,description=$4,
                image_url=$5,stock=$6,featured=$7 WHERE id=$8
            ''', name, category, price, desc, final_image, stock, featured, product_id)

            return {"success": True, "message": "Товар обновлён"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/products/{product_id}")
async def delete_product(product_id: int, admin=Depends(verify_admin)):
    try:
        async with db.pool.acquire() as conn:
            r = await conn.execute("DELETE FROM products WHERE id=$1", product_id)
            if r == "DELETE 0":
                raise HTTPException(status_code=404, detail="Товар не найден")
            return {"message": "Товар удалён"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/add-product")
async def admin_add_product_page_redirect(request: Request):
    return templates.TemplateResponse("add_product.html", {"request": request})


@app.get("/api/stats")
async def get_public_stats():
    try:
        async with db.pool.acquire() as conn:
            return {
                "total_products": await conn.fetchval("SELECT COUNT(*) FROM products") or 0,
                "total_orders": await conn.fetchval("SELECT COUNT(*) FROM orders") or 0,
                "active_orders": await conn.fetchval("SELECT COUNT(*) FROM orders WHERE status NOT IN ('delivered','cancelled')") or 0,
                "categories": await conn.fetchval("SELECT COUNT(*) FROM categories") or 0,
                "total_stock": await conn.fetchval("SELECT COALESCE(SUM(stock),0) FROM products") or 0,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-auth")
async def test_auth():
    try:
        async with db.pool.acquire() as conn:
            return {
                "status": "ok",
                "users_count": await conn.fetchval("SELECT COUNT(*) FROM users"),
                "products_count": await conn.fetchval("SELECT COUNT(*) FROM products"),
                "categories_count": await conn.fetchval("SELECT COUNT(*) FROM categories"),
                "orders_count": await conn.fetchval("SELECT COUNT(*) FROM orders"),
                "database_connected": True
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ========== ЗАПУСК ==========
if __name__ == "__main__":
    print("=" * 70)
    print("🛴 ScooterParts v5.0")
    print("=" * 70)
    print("   http://localhost:8000              — Главная")
    print("   http://localhost:8000/products     — Каталог")
    print("   http://localhost:8000/admin        — Админка")
    print("   http://localhost:8000/privacy-policy — Политика конфиденциальности")
    print("=" * 70)
    print("⚠️  Заполните .env: ADMIN_PASSWORD, SECRET_KEY, DATABASE_URL")
    print("💳  Для эквайринга: PAYMENT_API_KEY, PAYMENT_SHOP_ID, PAYMENT_SECRET_KEY")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
