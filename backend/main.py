# backend/main.py  ·  IMPORT v8.3_FINAL
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, UploadFile, File, Form, Query
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
from PIL import Image
import io

import asyncpg
from asyncpg.pool import Pool
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from backend.security import CSRFProtection, CookieAuth, CookieConsent

from fastapi.templating import Jinja2Templates
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import time

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

# Создаем директории с правильными правами
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "images").mkdir(exist_ok=True)
(STATIC_DIR / "favicon").mkdir(exist_ok=True)

# Устанавливаем права на запись для uploads
try:
    os.chmod(UPLOAD_DIR, 0o777)
    print(f"✅ Права на папку uploads установлены: {UPLOAD_DIR}")
except Exception as e:
    print(f"⚠️ Не удалось установить права на {UPLOAD_DIR}: {e}")


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
    specification_id: Optional[int] = None


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
    in_stock: bool = False  # В наличии
    preorder: bool = False  # Доступен по предзаказу
    cost_price: Optional[float] = None  # Себестоимость (только для админа)


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    stock: Optional[int] = None
    featured: Optional[bool] = None
    in_stock: Optional[bool] = None  # В наличии
    preorder: Optional[bool] = None  # Доступен по предзаказу
    cost_price: Optional[float] = None  # Себестоимость


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

# Инициализация модулей безопасности
csrf_protection = CSRFProtection(SECRET_KEY)
cookie_auth = CookieAuth()


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
                    image_url TEXT NOT NULL,
                    stock INTEGER DEFAULT 0,
                    featured BOOLEAN DEFAULT FALSE,
                    in_stock BOOLEAN DEFAULT FALSE,
                    preorder BOOLEAN DEFAULT FALSE,
                    cost_price DECIMAL(10,2),
                    has_specifications BOOLEAN DEFAULT FALSE,
                    specifications_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add columns if they don't exist (migration)
            try:
                await conn.execute('''
                    ALTER TABLE products ADD COLUMN IF NOT EXISTS preorder BOOLEAN DEFAULT FALSE
                ''')
                await conn.execute('''
                    ALTER TABLE products ADD COLUMN IF NOT EXISTS in_stock BOOLEAN DEFAULT FALSE
                ''')
                await conn.execute('''
                    ALTER TABLE products ADD COLUMN IF NOT EXISTS image_data TEXT
                ''')
                await conn.execute('''
                    ALTER TABLE products ADD COLUMN IF NOT EXISTS cost_price DECIMAL(10,2)
                ''')
                await conn.execute('''
                    ALTER TABLE products ADD COLUMN IF NOT EXISTS has_specifications BOOLEAN DEFAULT FALSE
                ''')
                await conn.execute('''
                    ALTER TABLE products ADD COLUMN IF NOT EXISTS specifications_data TEXT
                ''')
                # КРИТИЧЕСКИЕ МИГРАЦИИ: изменение типа image_url для поддержки base64
                await conn.execute('''
                    ALTER TABLE products ALTER COLUMN image_url TYPE TEXT
                ''')
                await conn.execute('''
                    ALTER TABLE product_specifications ALTER COLUMN image_url TYPE TEXT
                ''')
                await conn.execute('''
                    ALTER TABLE product_images ALTER COLUMN image_url TYPE TEXT
                ''')
                print("✅ Миграция: все image_url изменены на TEXT для поддержки base64")
                
                # КРИТИЧЕСКАЯ МИГРАЦИЯ: обновление in_stock для всех существующих товаров
                await conn.execute('''
                    UPDATE products SET in_stock = (stock > 0) WHERE in_stock != (stock > 0)
                ''')
                await conn.execute('''
                    UPDATE product_specifications SET in_stock = (stock > 0) WHERE in_stock != (stock > 0)
                ''')
                print("✅ Миграция: поле in_stock обновлено для всех товаров на основе stock")
            except Exception as e:
                print(f"⚠️ Миграция: {e}")
                pass  # Columns already exist
            
            # Спецификации товаров (версии/поколения)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS product_specifications (
                    id SERIAL PRIMARY KEY,
                    product_id INTEGER NOT NULL,
                    name VARCHAR(200) NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    description TEXT,
                    image_url TEXT,
                    stock INTEGER DEFAULT 0,
                    in_stock BOOLEAN DEFAULT FALSE,
                    preorder BOOLEAN DEFAULT FALSE,
                    cost_price DECIMAL(10,2),
                    sort_order INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
                )
            ''')
            
            # Характеристики товара
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS product_characteristics (
                    id SERIAL PRIMARY KEY,
                    product_id INTEGER NOT NULL,
                    specification_id INTEGER,
                    char_name VARCHAR(100) NOT NULL,
                    char_value TEXT NOT NULL,
                    sort_order INTEGER DEFAULT 0,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
                    FOREIGN KEY (specification_id) REFERENCES product_specifications(id) ON DELETE CASCADE
                )
            ''')
            
            # Дополнительные изображения товара
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS product_images (
                    id SERIAL PRIMARY KEY,
                    product_id INTEGER NOT NULL,
                    specification_id INTEGER,
                    image_url TEXT NOT NULL,
                    sort_order INTEGER DEFAULT 0,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
                    FOREIGN KEY (specification_id) REFERENCES product_specifications(id) ON DELETE CASCADE
                )
            ''')

            # Корзина
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS cart_items (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL,
                    product_id INTEGER NOT NULL,
                    specification_id INTEGER,
                    quantity INTEGER NOT NULL CHECK (quantity > 0),
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, product_id, specification_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
                    FOREIGN KEY (specification_id) REFERENCES product_specifications(id) ON DELETE CASCADE
                )
            ''')
            
            # Миграция: добавляем specification_id если его нет
            try:
                await conn.execute('''
                    ALTER TABLE cart_items ADD COLUMN IF NOT EXISTS specification_id INTEGER
                ''')
                # Удаляем старое ограничение уникальности
                await conn.execute('''
                    ALTER TABLE cart_items DROP CONSTRAINT IF EXISTS cart_items_user_id_product_id_key
                ''')
                # Добавляем новое ограничение уникальности
                await conn.execute('''
                    ALTER TABLE cart_items ADD CONSTRAINT cart_items_unique 
                    UNIQUE(user_id, product_id, specification_id)
                ''')
            except:
                pass  # Columns/constraints already exist

            # Миграция: delay_note в orders
            try:
                await conn.execute('''
                    ALTER TABLE orders ADD COLUMN IF NOT EXISTS delay_note TEXT
                ''')
            except:
                pass

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

            # Заметки о клиентах (CRM)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS customer_notes (
                    id SERIAL PRIMARY KEY,
                    user_id UUID NOT NULL,
                    note TEXT NOT NULL,
                    created_by VARCHAR(50) DEFAULT 'admin',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
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
                    # УДАЛЕНО: ("Колесо 10\" All-Terrain","tires",1800.00,"Пневматическое колесо для бездорожья с усиленными стенками.","/static/images/wheel.jpg",20,False),
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
app = FastAPI(title="IMPORT API v8.3 Security Edition", lifespan=lifespan)

# ========== SECURITY MIDDLEWARE ==========
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware для добавления заголовков безопасности ко всем ответам.
    Защита от XSS, clickjacking, MIME sniffing и других атак.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Content Security Policy - защита от XSS
        # Разрешаем скрипты только с того же домена и inline (для совместимости)
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
                "https://fonts.googleapis.com "
                "https://cdn.jsdelivr.net "
                "https://cdnjs.cloudflare.com "
                "https://cdn.prod.website-files.com; "
            "style-src 'self' 'unsafe-inline' "
                "https://fonts.googleapis.com "
                "https://cdn.jsdelivr.net "
                "https://cdnjs.cloudflare.com; "
            "font-src 'self' data: "
                "https://fonts.gstatic.com "
                "https://cdn.jsdelivr.net "
                "https://cdnjs.cloudflare.com "
                "https://cdn.prod.website-files.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )
        response.headers["Content-Security-Policy"] = csp_policy
        
        # X-Frame-Options - защита от clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options - защита от MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection (устаревший, но все еще полезен для старых браузеров)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy - контроль передачи referrer
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy - ограничение доступа к API браузера
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Strict-Transport-Security (HSTS) - только для HTTPS
        # На продакшене (Render.com) будет HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Rate Limiting Middleware - защита от brute-force
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Простой rate limiter для защиты от brute-force атак.
    Ограничивает количество запросов с одного IP.
    """
    def __init__(self, app, max_requests: int = 100, window: int = 60):
        super().__init__(app)
        self.max_requests = max_requests  # Максимум запросов
        self.window = window  # Окно времени в секундах
        self.requests = {}  # {ip: [(timestamp, count)]}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Очищаем старые записи
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (ts, count) for ts, count in self.requests[client_ip]
                if current_time - ts < self.window
            ]
        
        # Подсчитываем запросы
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        total_requests = sum(count for _, count in self.requests[client_ip])
        
        if total_requests >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Слишком много запросов. Попробуйте позже."}
            )
        
        # Добавляем текущий запрос
        self.requests[client_ip].append((current_time, 1))
        
        response = await call_next(request)
        return response

app.add_middleware(RateLimitMiddleware, max_requests=100, window=60)

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

@app.get("/about")
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/tracking")
async def tracking_page(request: Request):
    return templates.TemplateResponse("tracking.html", {"request": request})

@app.get("/auth")
async def auth_page(request: Request, next: str = "/"):
    return templates.TemplateResponse("auth.html", {"request": request, "next_url": next})


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
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: гарантируем правильное значение in_stock
                d['in_stock'] = bool(d.get('stock', 0) > 0)
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
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: гарантируем правильное значение in_stock
            d['in_stock'] = bool(d.get('stock', 0) > 0)
            
            # Если товар имеет спецификации, загружаем их
            if d.get('has_specifications'):
                specs = await conn.fetch('''
                    SELECT id, name, price, description, image_url, stock, in_stock, preorder, cost_price, sort_order
                    FROM product_specifications
                    WHERE product_id = $1
                    ORDER BY sort_order ASC, id ASC
                ''', product_id)
                
                d['specifications'] = []
                for s in specs:
                    spec_dict = dict(s)
                    spec_dict['price'] = float(spec_dict['price'])
                    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: гарантируем правильное значение in_stock для спецификаций
                    spec_dict['in_stock'] = bool(spec_dict.get('stock', 0) > 0)
                    if spec_dict.get('cost_price'):
                        spec_dict['cost_price'] = float(spec_dict['cost_price'])
                    
                    # Загружаем характеристики для каждой спецификации
                    chars = await conn.fetch('''
                        SELECT char_name, char_value
                        FROM product_characteristics
                        WHERE specification_id = $1
                        ORDER BY sort_order ASC, id ASC
                    ''', s['id'])
                    spec_dict['characteristics'] = [dict(c) for c in chars]
                    
                    # Загружаем дополнительные изображения для каждой спецификации
                    images = await conn.fetch('''
                        SELECT image_url
                        FROM product_images
                        WHERE specification_id = $1
                        ORDER BY sort_order ASC, id ASC
                    ''', s['id'])
                    spec_dict['images'] = [img['image_url'] for img in images]
                    
                    d['specifications'].append(spec_dict)
            
            # Загружаем характеристики основного товара (если есть)
            chars = await conn.fetch('''
                SELECT char_name, char_value
                FROM product_characteristics
                WHERE product_id = $1 AND specification_id IS NULL
                ORDER BY sort_order ASC, id ASC
            ''', product_id)
            d['characteristics'] = [dict(c) for c in chars]
            
            # Загружаем дополнительные изображения основного товара
            images = await conn.fetch('''
                SELECT image_url
                FROM product_images
                WHERE product_id = $1 AND specification_id IS NULL
                ORDER BY sort_order ASC, id ASC
            ''', product_id)
            d['images'] = [img['image_url'] for img in images]
            
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
                SELECT ci.product_id, ci.specification_id, ci.quantity,
                       p.name, p.category, p.price, p.description, p.image_url, p.stock,
                       ps.name as spec_name, ps.price as spec_price, ps.stock as spec_stock,
                       ps.image_url as spec_image_url, ps.description as spec_description
                FROM cart_items ci 
                JOIN products p ON ci.product_id = p.id
                LEFT JOIN product_specifications ps ON ci.specification_id = ps.id
                WHERE ci.user_id = $1 ORDER BY ci.added_at DESC
            ''', user_id)
            total = 0
            result = []
            for item in items:
                # Если есть спецификация, используем её данные
                if item['specification_id']:
                    price = float(item['spec_price'])
                    stock = item['spec_stock']
                    image_url = item['spec_image_url'] or item['image_url']
                    name = f"{item['name']} - {item['spec_name']}"
                    description = item['spec_description'] or item['description']
                else:
                    price = float(item['price'])
                    stock = item['stock']
                    image_url = item['image_url']
                    name = item['name']
                    description = item['description']
                
                item_total = price * item['quantity']
                total += item_total
                result.append({
                    "product_id": item['product_id'],
                    "specification_id": item['specification_id'],
                    "quantity": item['quantity'],
                    "product": {
                        "id": item['product_id'], 
                        "specification_id": item['specification_id'],
                        "name": name,
                        "category": item['category'], 
                        "price": price,
                        "description": description, 
                        "image_url": image_url,
                        "stock": stock
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
            # Если указана спецификация, проверяем её
            if cart_item.specification_id:
                spec = await conn.fetchrow(
                    "SELECT id, stock, product_id FROM product_specifications WHERE id=$1", 
                    cart_item.specification_id
                )
                if not spec:
                    raise HTTPException(status_code=404, detail="Спецификация не найдена")
                if spec['product_id'] != cart_item.product_id:
                    raise HTTPException(status_code=400, detail="Спецификация не принадлежит данному товару")
                if spec['stock'] < cart_item.quantity:
                    raise HTTPException(status_code=400, detail="Недостаточно товара на складе")
            else:
                # Проверяем основной товар
                product = await conn.fetchrow("SELECT id, stock FROM products WHERE id=$1", cart_item.product_id)
                if not product:
                    raise HTTPException(status_code=404, detail="Товар не найден")
                if product['stock'] < cart_item.quantity:
                    raise HTTPException(status_code=400, detail="Недостаточно товара на складе")
            
            await conn.execute('''
                INSERT INTO cart_items (user_id, product_id, specification_id, quantity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, product_id, specification_id) 
                DO UPDATE SET quantity = EXCLUDED.quantity
            ''', user_id, cart_item.product_id, cart_item.specification_id, cart_item.quantity)
            return {"message": "Товар добавлен в корзину"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cart/{product_id}")
async def remove_from_cart(product_id: int, specification_id: Optional[int] = Query(None), user_id: str = Depends(get_current_user)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    try:
        async with db.pool.acquire() as conn:
            if specification_id is not None:
                r = await conn.execute(
                    "DELETE FROM cart_items WHERE user_id=$1 AND product_id=$2 AND specification_id=$3", 
                    user_id, product_id, specification_id
                )
            else:
                r = await conn.execute(
                    "DELETE FROM cart_items WHERE user_id=$1 AND product_id=$2 AND specification_id IS NULL", 
                    user_id, product_id
                )
            if r == "DELETE 0":
                raise HTTPException(status_code=404, detail="Товар не найден в корзине")
            return {"message": "Товар удалён из корзины"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/cart/{product_id}")
async def update_cart_quantity(product_id: int, body: dict, user_id: str = Depends(get_current_user)):
    """Обновить количество товара в корзине"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Не авторизован")
    
    quantity = body.get("quantity", 1)
    specification_id = body.get("specification_id")
    
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Количество должно быть больше 0")
    
    try:
        async with db.pool.acquire() as conn:
            # Проверяем наличие товара/спецификации и stock
            if specification_id:
                item = await conn.fetchrow(
                    "SELECT stock FROM product_specifications WHERE id=$1", 
                    specification_id
                )
                if not item:
                    raise HTTPException(status_code=404, detail="Спецификация не найдена")
            else:
                item = await conn.fetchrow(
                    "SELECT stock FROM products WHERE id=$1", 
                    product_id
                )
                if not item:
                    raise HTTPException(status_code=404, detail="Товар не найден")
            
            if item['stock'] < quantity:
                raise HTTPException(status_code=400, detail=f"Недостаточно на складе (доступно: {item['stock']})")
            
            # Обновляем количество
            if specification_id:
                await conn.execute('''
                    UPDATE cart_items SET quantity = $1
                    WHERE user_id = $2 AND product_id = $3 AND specification_id = $4
                ''', quantity, user_id, product_id, specification_id)
            else:
                await conn.execute('''
                    UPDATE cart_items SET quantity = $1
                    WHERE user_id = $2 AND product_id = $3 AND specification_id IS NULL
                ''', quantity, user_id, product_id)
            
            return {"message": "Количество обновлено", "quantity": quantity}
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
                SELECT ci.product_id, ci.specification_id, ci.quantity, 
                       p.name, p.price, p.stock,
                       ps.name as spec_name, ps.price as spec_price, ps.stock as spec_stock
                FROM cart_items ci 
                JOIN products p ON ci.product_id = p.id
                LEFT JOIN product_specifications ps ON ci.specification_id = ps.id
                WHERE ci.user_id = $1
            ''', user_id)

            if not cart_items:
                raise HTTPException(status_code=400, detail="Корзина пуста")

            # Подсчитываем общую сумму с учётом спецификаций
            total = 0
            for i in cart_items:
                price = float(i['spec_price']) if i['specification_id'] else float(i['price'])
                total += price * i['quantity']

            order_id = await conn.fetchval('''
                INSERT INTO orders (user_id, total_amount, delivery_address, comment, status, payment_status)
                VALUES ($1, $2, $3, $4, 'pending', 'pending')
                RETURNING id
            ''', user_id, total, order_data.delivery_address, order_data.comment)

            for item in cart_items:
                # Определяем название и цену товара
                if item['specification_id']:
                    product_name = f"{item['name']} - {item['spec_name']}"
                    price = float(item['spec_price'])
                    stock_column = 'product_specifications'
                    stock_id = item['specification_id']
                    current_stock = item['spec_stock']
                else:
                    product_name = item['name']
                    price = float(item['price'])
                    stock_column = 'products'
                    stock_id = item['product_id']
                    current_stock = item['stock']
                
                # Проверяем наличие
                if current_stock < item['quantity']:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Недостаточно товара '{product_name}' на складе"
                    )
                
                await conn.execute('''
                    INSERT INTO order_items (order_id, product_id, product_name, price, quantity)
                    VALUES ($1, $2, $3, $4, $5)
                ''', order_id, item['product_id'], product_name, price, item['quantity'])
                
                # Обновляем остатки
                if item['specification_id']:
                    await conn.execute(
                        "UPDATE product_specifications SET stock = stock - $1 WHERE id = $2",
                        item['quantity'], item['specification_id']
                    )
                else:
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
                SELECT o.id, o.status, o.total_amount, o.payment_status,
                       o.created_at, o.delivery_address, o.delay_note,
                       COALESCE(
                         json_agg(
                           json_build_object(
                             'product_name', oi.product_name,
                             'quantity', oi.quantity,
                             'price', oi.price
                           ) ORDER BY oi.id
                         ) FILTER (WHERE oi.id IS NOT NULL),
                         '[]'
                       ) AS items
                FROM orders o
                LEFT JOIN order_items oi ON oi.order_id = o.id
                WHERE o.user_id=$1
                GROUP BY o.id
                ORDER BY o.created_at DESC
            ''', user_id)
            result = []
            for o in orders:
                d = dict(o)
                d['total_amount'] = float(d['total_amount'])
                if isinstance(d.get('created_at'), datetime):
                    d['created_at'] = d['created_at'].isoformat()
                # Parse items JSON if returned as string
                import json as _json
                if isinstance(d.get('items'), str):
                    try: d['items'] = _json.loads(d['items'])
                    except: d['items'] = []
                # Convert Decimal prices in items
                if d.get('items'):
                    for item in d['items']:
                        if 'price' in item:
                            item['price'] = float(item['price'])
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


@app.post("/api/admin/migrate-images")
async def migrate_images_to_base64(admin=Depends(verify_admin)):
    """
    Утилита для миграции изображений из файловой системы в base64
    Полезно при переносе с файлового хранения на БД
    """
    try:
        import base64
        migrated = 0
        failed = []
        
        async with db.pool.acquire() as conn:
            # Получаем все товары с изображениями из файлов
            products = await conn.fetch('''
                SELECT id, name, image_url 
                FROM products 
                WHERE image_url LIKE '/static/uploads/%'
            ''')
            
            for product in products:
                try:
                    # Путь к файлу
                    file_path = STATIC_DIR / product['image_url'].lstrip('/')
                    
                    if not file_path.exists():
                        failed.append(f"Product {product['id']}: File not found")
                        continue
                    
                    # Читаем файл
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    # Оптимизируем
                    optimized_data = await optimize_image(file_data)
                    
                    # Конвертируем в base64
                    image_base64 = base64.b64encode(optimized_data).decode('utf-8')
                    image_data_url = f"data:image/jpeg;base64,{image_base64}"
                    
                    # Обновляем в БД
                    await conn.execute('''
                        UPDATE products 
                        SET image_url = $1 
                        WHERE id = $2
                    ''', image_data_url, product['id'])
                    
                    migrated += 1
                    print(f"✅ Migrated: {product['name']}")
                    
                except Exception as e:
                    failed.append(f"Product {product['id']}: {str(e)}")
                    print(f"❌ Failed: {product['name']} - {e}")
        
        return {
            "success": True,
            "migrated": migrated,
            "failed": failed,
            "message": f"Migrated {migrated} images to base64"
        }
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


@app.get("/api/admin/top-customers")
async def get_top_customers(admin=Depends(verify_admin), limit: int = 20):
    """Получить топ активных покупателей для CRM"""
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT 
                    u.id,
                    u.username,
                    u.email,
                    u.full_name,
                    u.phone,
                    COUNT(o.id) as order_count,
                    COALESCE(SUM(o.total_amount), 0) as total_spent,
                    MAX(o.created_at) as last_order_date,
                    COUNT(CASE WHEN o.payment_status = 'paid' THEN 1 END) as paid_orders
                FROM users u
                LEFT JOIN orders o ON u.id = o.user_id
                WHERE u.is_admin = FALSE
                GROUP BY u.id, u.username, u.email, u.full_name, u.phone
                HAVING COUNT(o.id) > 0
                ORDER BY total_spent DESC
                LIMIT $1
            ''', limit)
            
            result = []
            for r in rows:
                d = dict(r)
                d['id'] = str(d['id'])
                d['total_spent'] = float(d['total_spent'])
                if isinstance(d.get('last_order_date'), datetime):
                    d['last_order_date'] = d['last_order_date'].isoformat()
                result.append(d)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/customers")
async def get_all_customers(admin=Depends(verify_admin)):
    """Получить всех покупателей с количеством заметок для CRM"""
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT 
                    u.id,
                    u.username,
                    u.email,
                    u.full_name,
                    u.phone,
                    COUNT(DISTINCT o.id) as order_count,
                    COALESCE(SUM(o.total_amount), 0) as total_spent,
                    MAX(o.created_at) as last_order_date,
                    COUNT(DISTINCT n.id) as notes_count
                FROM users u
                LEFT JOIN orders o ON u.id = o.user_id
                LEFT JOIN customer_notes n ON u.id = n.user_id
                WHERE u.is_admin = FALSE
                GROUP BY u.id, u.username, u.email, u.full_name, u.phone
                HAVING COUNT(DISTINCT o.id) > 0
                ORDER BY total_spent DESC
            ''')
            
            result = []
            for r in rows:
                d = dict(r)
                d['id'] = str(d['id'])
                d['total_spent'] = float(d['total_spent'])
                if isinstance(d.get('last_order_date'), datetime):
                    d['last_order_date'] = d['last_order_date'].isoformat()
                result.append(d)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/customers/{user_id}/notes")
async def get_customer_notes(user_id: str, admin=Depends(verify_admin)):
    """Получить заметки о клиенте"""
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT id, note, created_by, created_at
                FROM customer_notes
                WHERE user_id = $1
                ORDER BY created_at DESC
            ''', user_id)
            
            result = []
            for r in rows:
                d = dict(r)
                if isinstance(d.get('created_at'), datetime):
                    d['created_at'] = d['created_at'].isoformat()
                result.append(d)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/customers/{user_id}/notes")
async def add_customer_note(user_id: str, body: dict, admin=Depends(verify_admin)):
    """Добавить заметку о клиенте"""
    note = body.get("note", "").strip()
    if not note:
        raise HTTPException(status_code=400, detail="Заметка не может быть пустой")
    
    try:
        async with db.pool.acquire() as conn:
            row = await conn.fetchrow('''
                INSERT INTO customer_notes (user_id, note, created_by)
                VALUES ($1, $2, $3)
                RETURNING id, note, created_by, created_at
            ''', user_id, note, "admin")
            
            d = dict(row)
            if isinstance(d.get('created_at'), datetime):
                d['created_at'] = d['created_at'].isoformat()
            return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/customers/notes/{note_id}")
async def delete_customer_note(note_id: int, admin=Depends(verify_admin)):
    """Удалить заметку о клиенте"""
    try:
        async with db.pool.acquire() as conn:
            r = await conn.execute("DELETE FROM customer_notes WHERE id=$1", note_id)
            if r == "DELETE 0":
                raise HTTPException(status_code=404, detail="Заметка не найдена")
            return {"message": "Заметка удалена"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/orders/{order_id}/status")
async def update_order_status(order_id: int, body: dict, admin=Depends(verify_admin)):
    new_status = body.get("status")
    delay_note = body.get("delay_note")  # Optional[str]
    valid = [
        'pending', 'confirmed', 'processing', 'shipped',
        'customs', 'delivered', 'handed_over', 'completed', 'cancelled'
    ]
    if new_status not in valid:
        raise HTTPException(status_code=400, detail=f"Недопустимый статус. Допустимые: {valid}")
    try:
        async with db.pool.acquire() as conn:
            r = await conn.execute(
                "UPDATE orders SET status=$1, delay_note=$2, updated_at=NOW() WHERE id=$3",
                new_status,
                delay_note if delay_note else None,
                order_id
            )
            if r == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Заказ не найден")
            return {"message": "Статус обновлён", "status": new_status, "delay_note": delay_note}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ========== IMAGE OPTIMIZATION ==========
# ==========================================

async def optimize_image(image_data: bytes, max_size: tuple = (800, 800), quality: int = 75) -> bytes:
    """
    Оптимизирует изображение для хранения в базе данных (base64)
    
    Агрессивная оптимизация:
    - Уменьшение до 800x800 (вместо 1200x1200)
    - Качество 75% (вместо 85%)
    - Это уменьшает размер на ~60-70%
    
    Args:
        image_data: Байты исходного изображения
        max_size: Максимальный размер (ширина, высота)
        quality: Качество сжатия JPEG (1-100)
    
    Returns:
        Оптимизированные байты изображения
    """
    try:
        # Открываем изображение
        img = Image.open(io.BytesIO(image_data))
        
        # Логируем исходный размер
        print(f"📸 Original size: {img.size} ({len(image_data)} bytes)")
        
        # Конвертируем в RGB если необходимо (для PNG с прозрачностью)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Создаем белый фон для прозрачных изображений
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Изменяем размер, сохраняя пропорции
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        print(f"📸 Resized to: {img.size}")
        
        # Сохраняем оптимизированное изображение
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        optimized_data = output.getvalue()
        
        print(f"📸 Optimized size: {len(optimized_data)} bytes (compression: {100 - int(len(optimized_data)/len(image_data)*100)}%)")
        
        return optimized_data
    except Exception as e:
        # Если оптимизация не удалась, возвращаем оригинал
        print(f"⚠️ Image optimization failed: {e}, using original")
        return image_data


@app.post("/api/admin/products")
async def create_product(request: Request, admin=Depends(verify_admin)):
    try:
        form = await request.form()
        
        # КРИТИЧЕСКОЕ ЛОГИРОВАНИЕ
        print("\n" + "="*80)
        print("🔍 НАЧАЛО ОБРАБОТКИ ЗАПРОСА НА СОЗДАНИЕ ТОВАРА")
        print("="*80)
        
        name     = str(form.get("name","")).strip()
        category = str(form.get("category","")).strip()
        price    = float(form.get("price",0))
        desc     = str(form.get("description","")).strip()
        stock    = int(form.get("stock",0))
        featured = str(form.get("featured","false")).lower() == "true"
        # ИСПРАВЛЕНИЕ: Автоматически определяем in_stock на основе stock
        in_stock = stock > 0
        preorder = str(form.get("preorder","false")).lower() == "true"
        cost_price_str = str(form.get("cost_price","")).strip()
        cost_price = float(cost_price_str) if cost_price_str else None
        image_url = str(form.get("image_url","")).strip()
        
        # Получаем все файлы изображений (до 5 штук)
        image_files = []
        for i in range(5):
            img_file = form.get(f"image_file_{i}")
            if img_file and hasattr(img_file, 'filename') and img_file.filename:
                image_files.append(img_file)
        
        # Обратная совместимость: если используется старое поле image_file
        old_image_file = form.get("image_file")
        if old_image_file and hasattr(old_image_file, 'filename') and old_image_file.filename:
            if not image_files:  # Только если новые файлы не загружены
                image_files.append(old_image_file)
        
        print(f"📝 Название: {name}")
        print(f"📁 Категория: {category}")
        print(f"💰 Цена: {price}")
        print(f"🖼️  Получено файлов изображений: {len(image_files)}")
        print(f"🌐 image_url: '{image_url}'")
        print("="*80 + "\n")

        if not name or len(name) < 3:
            raise HTTPException(status_code=400, detail="Название слишком короткое (мин. 3 символа)")
        if not category:
            raise HTTPException(status_code=400, detail="Категория обязательна")
        if price <= 0:
            raise HTTPException(status_code=400, detail="Цена должна быть больше 0")
        if not desc or len(desc) < 10:
            raise HTTPException(status_code=400, detail="Описание слишком короткое (мин. 10 символов)")

        # По умолчанию пустая строка (не null), чтобы избежать constraint violation
        final_image = ""
        additional_images = []

        # Обработка всех загруженных изображений
        for idx, image_file in enumerate(image_files):
            print(f"🔍 Обработка изображения {idx + 1}/{len(image_files)}")
            
            # Валидация формата
            ext = Path(image_file.filename).suffix.lower()
            allowed_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            if ext not in allowed_formats:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Недопустимый формат файла {image_file.filename}. Разрешены: {', '.join(allowed_formats)}"
                )
            
            # Чтение файла
            file_content = await image_file.read()
            
            # Валидация размера (макс 10MB)
            max_size_mb = 10
            if len(file_content) > max_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Файл {image_file.filename} слишком большой. Максимум {max_size_mb}MB"
                )
            
            # Оптимизация изображения
            try:
                optimized_content = await optimize_image(file_content)
                
                # Конвертируем в base64 для хранения в БД
                import base64
                image_base64 = base64.b64encode(optimized_content).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{image_base64}"
                
                # Первое изображение - основное
                if idx == 0:
                    final_image = image_data_url
                    print(f"📸 Основное изображение: {len(file_content)} → {len(optimized_content)} bytes")
                else:
                    additional_images.append(image_data_url)
                    print(f"📸 Дополнительное изображение {idx}: {len(file_content)} → {len(optimized_content)} bytes")
                
            except Exception as e:
                print(f"❌ Error processing image {idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения {idx + 1}: {str(e)}")
        
        # Если файлы не загружены, используем URL
        if not image_files and image_url:
            print(f"ℹ️ Используется image_url: {image_url}")
            final_image = image_url

        async with db.pool.acquire() as conn:
            print(f"\n💾 СОХРАНЕНИЕ В БД:")
            print(f"   final_image длина: {len(final_image)}")
            print(f"   Дополнительных изображений: {len(additional_images)}")
            print("="*80 + "\n")
            
            # Создаем товар
            row = await conn.fetchrow('''
                INSERT INTO products (name,category,price,description,image_url,stock,featured,in_stock,preorder,cost_price)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING *
            ''', name, category, price, desc, final_image, stock, featured, in_stock, preorder, cost_price)
            
            product_id = row['id']
            
            # Сохраняем дополнительные изображения
            for idx, img_url in enumerate(additional_images):
                await conn.execute('''
                    INSERT INTO product_images (product_id, image_url, sort_order)
                    VALUES ($1, $2, $3)
                ''', product_id, img_url, idx + 1)
                print(f"✅ Сохранено дополнительное изображение {idx + 1}")
            
            d = dict(row)
            d['price'] = float(d['price'])
            if d.get('cost_price'):
                d['cost_price'] = float(d['cost_price'])
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
            # ИСПРАВЛЕНИЕ: Автоматически определяем in_stock на основе stock
            in_stock = stock > 0
            preorder = str(form.get("preorder", str(existing.get('preorder', False)))).lower() == "true"
            cost_price_str = str(form.get("cost_price","")).strip()
            cost_price = float(cost_price_str) if cost_price_str else existing.get('cost_price')
            image_url = str(form.get("image_url","")).strip()
            image_file = form.get("image_file")

            final_image = existing['image_url']
            if image_file and hasattr(image_file, 'filename') and image_file.filename:
                # Валидация формата
                ext = Path(image_file.filename).suffix.lower()
                allowed_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                if ext not in allowed_formats:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Недопустимый формат файла. Разрешены: {', '.join(allowed_formats)}"
                    )
                
                # Чтение и валидация размера
                file_content = await image_file.read()
                max_size_mb = 10
                if len(file_content) > max_size_mb * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Файл слишком большой. Максимум {max_size_mb}MB"
                    )
                
                # Оптимизация изображения
                try:
                    optimized_content = await optimize_image(file_content)
                    
                    # Конвертируем в base64
                    import base64
                    image_base64 = base64.b64encode(optimized_content).decode('utf-8')
                    image_data_url = f"data:image/jpeg;base64,{image_base64}"
                    
                    final_image = image_data_url
                    
                    print(f"📸 Image updated: {len(file_content)} → {len(optimized_content)} bytes")
                    print(f"✅ Image converted to base64 successfully")
                    
                except Exception as e:
                    print(f"❌ Error updating image: {e}")
                    raise HTTPException(status_code=500, detail=f"Ошибка обновления изображения: {str(e)}")
            elif image_url:
                final_image = image_url

            await conn.execute('''
                UPDATE products SET name=$1,category=$2,price=$3,description=$4,
                image_url=$5,stock=$6,featured=$7,in_stock=$8,preorder=$9,cost_price=$10 WHERE id=$11
            ''', name, category, price, desc, final_image, stock, featured, in_stock, preorder, cost_price, product_id)

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


# ==========================================
# ========== SPECIFICATIONS API ==========
# ==========================================

@app.get("/api/products/{product_id}/images")
async def get_product_images(product_id: int):
    """Получить все дополнительные изображения товара"""
    try:
        async with db.pool.acquire() as conn:
            images = await conn.fetch('''
                SELECT id, image_url, sort_order
                FROM product_images
                WHERE product_id = $1 AND specification_id IS NULL
                ORDER BY sort_order ASC, id ASC
            ''', product_id)
            
            return [dict(img) for img in images]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/products/{product_id}/specifications")
async def get_product_specifications(product_id: int):
    """Получить все спецификации товара"""
    try:
        async with db.pool.acquire() as conn:
            specs = await conn.fetch('''
                SELECT id, name, price, description, image_url, stock, in_stock, preorder, cost_price, sort_order
                FROM product_specifications
                WHERE product_id = $1
                ORDER BY sort_order ASC, id ASC
            ''', product_id)
            
            result = []
            for s in specs:
                d = dict(s)
                d['price'] = float(d['price'])
                if d.get('cost_price'):
                    d['cost_price'] = float(d['cost_price'])
                result.append(d)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/products/{product_id}/specifications")
async def add_product_specification(product_id: int, request: Request, admin=Depends(verify_admin)):
    """Добавить спецификацию к товару"""
    try:
        form = await request.form()
        name = str(form.get("name", "")).strip()
        price = float(form.get("price", 0))
        desc = str(form.get("description", "")).strip()
        stock = int(form.get("stock", 0))
        # ИСПРАВЛЕНИЕ: Автоматически определяем in_stock на основе stock
        in_stock = stock > 0
        preorder = str(form.get("preorder", "false")).lower() == "true"
        cost_price_str = str(form.get("cost_price", "")).strip()
        cost_price = float(cost_price_str) if cost_price_str else None
        image_url = str(form.get("image_url", "")).strip()
        sort_order = int(form.get("sort_order", 0))

        if not name or len(name) < 3:
            raise HTTPException(status_code=400, detail="Название слишком короткое")
        if price <= 0:
            raise HTTPException(status_code=400, detail="Цена должна быть больше 0")

        # Получаем все файлы изображений (до 5 штук)
        image_files = []
        for i in range(5):
            img_file = form.get(f"image_file_{i}")
            if img_file and hasattr(img_file, 'filename') and img_file.filename:
                image_files.append(img_file)
        
        # Обратная совместимость: если используется старое поле image_file
        old_image_file = form.get("image_file")
        if old_image_file and hasattr(old_image_file, 'filename') and old_image_file.filename:
            if not image_files:  # Только если новые файлы не загружены
                image_files.append(old_image_file)

        final_image = None
        additional_images = []

        # Обработка всех загруженных изображений
        for idx, image_file in enumerate(image_files):
            ext = Path(image_file.filename).suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                raise HTTPException(status_code=400, detail=f"Недопустимый формат файла {image_file.filename}")
            
            # Чтение файла
            file_content = await image_file.read()
            
            # Валидация размера
            max_size_mb = 10
            if len(file_content) > max_size_mb * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"Файл {image_file.filename} слишком большой. Максимум {max_size_mb}MB")
            
            # Оптимизация изображения
            try:
                optimized_content = await optimize_image(file_content)
                
                # Конвертируем в base64
                import base64
                image_base64 = base64.b64encode(optimized_content).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{image_base64}"
                
                # Первое изображение - основное
                if idx == 0:
                    final_image = image_data_url
                else:
                    additional_images.append(image_data_url)
                
            except Exception as e:
                print(f"❌ Error processing image {idx + 1}: {e}")
                raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения {idx + 1}: {str(e)}")
        
        # Если файлы не загружены, используем URL
        if not image_files and image_url:
            final_image = image_url

        async with db.pool.acquire() as conn:
            # Проверяем существование товара
            product = await conn.fetchrow("SELECT id, has_specifications FROM products WHERE id=$1", product_id)
            if not product:
                raise HTTPException(status_code=404, detail="Товар не найден")
            
            # Помечаем товар как имеющий спецификации
            if not product['has_specifications']:
                await conn.execute("UPDATE products SET has_specifications=true WHERE id=$1", product_id)
            
            # Создаём спецификацию
            spec = await conn.fetchrow('''
                INSERT INTO product_specifications (product_id, name, price, description, image_url, stock, in_stock, preorder, cost_price, sort_order)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING *
            ''', product_id, name, price, desc, final_image, stock, in_stock, preorder, cost_price, sort_order)
            
            spec_id = spec['id']
            
            # Сохраняем дополнительные изображения
            for idx, img_url in enumerate(additional_images):
                await conn.execute('''
                    INSERT INTO product_images (product_id, specification_id, image_url, sort_order)
                    VALUES ($1, $2, $3, $4)
                ''', product_id, spec_id, img_url, idx + 1)
                print(f"✅ Сохранено дополнительное изображение спецификации {idx + 1}")
            
            d = dict(spec)
            d['price'] = float(d['price'])
            if d.get('cost_price'):
                d['cost_price'] = float(d['cost_price'])
            return {"success": True, "message": "Спецификация добавлена", "specification": d}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/specifications/{spec_id}")
async def update_specification(spec_id: int, request: Request, admin=Depends(verify_admin)):
    """Обновить спецификацию"""
    try:
        form = await request.form()
        async with db.pool.acquire() as conn:
            existing = await conn.fetchrow("SELECT * FROM product_specifications WHERE id=$1", spec_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Спецификация не найдена")

            name = str(form.get("name", existing['name'])).strip()
            price = float(form.get("price", existing['price']))
            desc = str(form.get("description", existing['description'] or "")).strip()
            stock = int(form.get("stock", existing['stock']))
            # ИСПРАВЛЕНИЕ: Автоматически определяем in_stock на основе stock
            in_stock = stock > 0
            preorder = str(form.get("preorder", str(existing.get('preorder', False)))).lower() == "true"
            cost_price_str = str(form.get("cost_price", "")).strip()
            cost_price = float(cost_price_str) if cost_price_str else existing.get('cost_price')
            image_url = str(form.get("image_url", "")).strip()
            image_file = form.get("image_file")
            sort_order = int(form.get("sort_order", existing['sort_order']))

            final_image = existing['image_url']
            if image_file and hasattr(image_file, 'filename') and image_file.filename:
                ext = Path(image_file.filename).suffix.lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    raise HTTPException(status_code=400, detail="Недопустимый формат файла")
                fname = f"{uuid4().hex}{ext}"
                async with aiofiles.open(UPLOAD_DIR / fname, 'wb') as buf:
                    await buf.write(await image_file.read())
                final_image = f"/static/uploads/{fname}"
            elif image_url:
                final_image = image_url

            await conn.execute('''
                UPDATE product_specifications 
                SET name=$1, price=$2, description=$3, image_url=$4, stock=$5, in_stock=$6, preorder=$7, cost_price=$8, sort_order=$9
                WHERE id=$10
            ''', name, price, desc, final_image, stock, in_stock, preorder, cost_price, sort_order, spec_id)

            return {"success": True, "message": "Спецификация обновлена"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/specifications/{spec_id}")
async def delete_specification(spec_id: int, admin=Depends(verify_admin)):
    """Удалить спецификацию"""
    try:
        async with db.pool.acquire() as conn:
            # Получаем product_id перед удалением
            spec = await conn.fetchrow("SELECT product_id FROM product_specifications WHERE id=$1", spec_id)
            if not spec:
                raise HTTPException(status_code=404, detail="Спецификация не найдена")
            
            product_id = spec['product_id']
            
            # Удаляем спецификацию
            await conn.execute("DELETE FROM product_specifications WHERE id=$1", spec_id)
            
            # Проверяем, остались ли еще спецификации у товара
            remaining = await conn.fetchval("SELECT COUNT(*) FROM product_specifications WHERE product_id=$1", product_id)
            if remaining == 0:
                await conn.execute("UPDATE products SET has_specifications=false WHERE id=$1", product_id)
            
            return {"message": "Спецификация удалена"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/products/{product_id}/images")
async def get_product_images(product_id: int, specification_id: Optional[int] = None):
    """Получить дополнительные изображения товара или спецификации"""
    try:
        async with db.pool.acquire() as conn:
            if specification_id:
                images = await conn.fetch('''
                    SELECT id, image_url, sort_order
                    FROM product_images
                    WHERE product_id = $1 AND specification_id = $2
                    ORDER BY sort_order ASC, id ASC
                ''', product_id, specification_id)
            else:
                images = await conn.fetch('''
                    SELECT id, image_url, sort_order
                    FROM product_images
                    WHERE product_id = $1 AND specification_id IS NULL
                    ORDER BY sort_order ASC, id ASC
                ''', product_id)
            
            return [dict(img) for img in images]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/products/{product_id}/characteristics")
async def get_product_characteristics(product_id: int, specification_id: Optional[int] = None):
    """Получить характеристики товара или спецификации"""
    try:
        async with db.pool.acquire() as conn:
            if specification_id:
                chars = await conn.fetch('''
                    SELECT id, char_name, char_value, sort_order
                    FROM product_characteristics
                    WHERE product_id = $1 AND specification_id = $2
                    ORDER BY sort_order ASC, id ASC
                ''', product_id, specification_id)
            else:
                chars = await conn.fetch('''
                    SELECT id, char_name, char_value, sort_order
                    FROM product_characteristics
                    WHERE product_id = $1 AND specification_id IS NULL
                    ORDER BY sort_order ASC, id ASC
                ''', product_id)
            
            return [dict(char) for char in chars]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== ЗАПУСК ==========
@app.get("/api/debug/uploads")
async def debug_uploads():
    """Debug endpoint to check uploads directory"""
    try:
        files = list(UPLOAD_DIR.iterdir())
        return {
            "upload_dir": str(UPLOAD_DIR),
            "exists": UPLOAD_DIR.exists(),
            "files": [f.name for f in files if f.is_file()],
            "count": len(files)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("=" * 70)
    print("⚡ IMPORT v5.1")
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
