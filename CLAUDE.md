# CLAUDE.md — ScooterParts AI Assistant Guide

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository.

---

## Project Overview

**ScooterParts** is a Russian e-commerce web application for scooter spare parts. It is a server-side rendered monolith built with FastAPI and Jinja2 templates, deployed on Render.com using Docker.

- **Live URL:** https://scooterparts.onrender.com
- **Stack:** Python + FastAPI + PostgreSQL + Jinja2 templates + Vanilla JS
- **Jurisdiction:** Russian Federation (compliant with ФЗ-152, РКН requirements)

---

## Repository Structure

```
scooterparts/
├── backend/
│   ├── __init__.py          # Package marker
│   └── main.py              # Entire backend — all routes, models, DB, auth (3800+ lines)
├── static/
│   ├── csrf-init.js         # Globally patches fetch() to inject CSRF token headers
│   ├── cookie-consent.js    # Cookie consent manager (v3.0, ФЗ-152 compliant)
│   └── favicon/             # Favicon assets (SVG, PNG, ICO, Apple touch)
├── templates/               # Jinja2 HTML templates (server-side rendered)
│   ├── index.html           # Homepage
│   ├── auth.html            # Login/register
│   ├── products.html        # Product catalog
│   ├── cart.html            # Shopping cart
│   ├── add_product.html     # Admin: add product
│   ├── admin.html           # Admin dashboard
│   ├── about.html           # About page
│   ├── legal.html           # Legal / РКН compliance page
│   ├── tracking.html        # Order tracking
│   └── payment_stub.html    # Payment gateway placeholder
├── generate_sri.py          # Generates SRI hashes for CDN resources
├── apply_sri.py             # Patches SRI hashes into templates automatically
├── Dockerfile               # Python 3.11-slim, exposes port 8000
├── docker-compose.yml       # PostgreSQL 15 + FastAPI app
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (see section below)
└── AUDIT_FINAL_v11.md       # Security/bug audit log (20 items, all addressed)
```

**Important:** There are no subdirectory layers — the entire backend lives in a single `backend/main.py` file. There are no dedicated services/, routers/, or models/ directories.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI 0.115.0 (async) |
| ASGI server | Uvicorn |
| Database | PostgreSQL 15 via asyncpg |
| Authentication | PyJWT 2.9.0 (JWT in HttpOnly cookies) |
| Password hashing | PBKDF2-SHA256, 100,000 iterations |
| Templating | Jinja2 |
| File I/O | aiofiles, Pillow (image processing) |
| Config | python-dotenv |
| Frontend | Vanilla JS, CSS variables, responsive grid |
| Animations | GSAP 3.12.5 + ScrollTrigger + Lenis (via CDN) |
| Fonts | Google Fonts (Rajdhani, Syne, DM Sans, JetBrains Mono) |
| Containerization | Docker + Docker Compose |

**Note on python-jose:** It is listed in requirements.txt as a comment (`# python-jose[cryptography]`) because it conflicts with PyJWT. Use PyJWT only.

---

## Environment Variables

Defined in `.env` (loaded by docker-compose and python-dotenv):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | asyncpg-compatible PostgreSQL connection string |
| `SECRET_KEY` | JWT signing key |
| `ADMIN_PASSWORD` | Admin account password |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins |
| `AUTH_RATE_LIMIT` | Max auth requests per minute per IP (default: 10) |
| `GLOBAL_RATE_LIMIT` | Max requests per minute per IP (default: 120) |
| `PAYMENT_CALLBACK_URL` | Payment gateway callback URL |

> **Security note:** The `.env` file is tracked in git, which is a known issue. Do not commit new secrets to this file. In production, use Render.com environment variable injection instead.

---

## Backend Architecture (`backend/main.py`)

### Pydantic Models (request/response validation)

- `UserRegister`, `UserLogin`, `AdminLogin`
- `CartItem`, `CartUpdate`, `CartQuantityUpdate`
- `ProductCreate`, `ProductUpdate`
- `CategoryCreate`, `CategoryUpdate`
- `OrderCreate`, `OrderStatusUpdate`
- `PaymentCreate`, `PaymentStatusUpdate`
- `TrackNumberUpdate`

### Authentication

- JWT tokens stored in `access_token` **HttpOnly** cookie (60-minute expiry)
- CSRF token stored in `csrf_token` cookie (readable by JS)
- `get_current_user()` — validates JWT from cookie
- `verify_admin()` — checks `role == "admin"` claim
- `verify_manager_or_admin()` — checks `role in ("manager", "admin")`
- `PasswordHasher` class — wraps PBKDF2-SHA256 with salt

### Security Validators

- `validate_image_url()` — blocks SSRF via regex; rejects localhost, 127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x, 0.0.0.0
- Rate limiting middleware applied to all routes
- CSRF tokens auto-injected in all mutating fetch calls via `csrf-init.js`

### Database

- Connection pool created with `asyncpg.create_pool()` at startup
- All DB queries are raw SQL with parameterized inputs (no ORM)
- Tables are auto-initialized on `startup` event
- Pre-seeded category data in `DEFAULT_CATEGORIES`

### Key API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/register` | None | User registration |
| POST | `/api/login` | None | User login (sets cookies) |
| POST | `/api/logout` | User | Clears access_token + csrf_token cookies |
| GET | `/api/csrf-token` | None | Returns CSRF token |
| GET | `/api/user` | User | Current user info |
| DELETE | `/api/delete-account` | User | Delete own account |
| GET/POST | `/api/cart` | User | View / modify cart |
| POST | `/api/cart/add` | User | Add item to cart |
| PUT | `/api/cart/update` | User | Update cart item |
| DELETE | `/api/cart/remove` | User | Remove cart item |
| GET | `/api/products` | None | List products |
| GET | `/api/products/{id}` | None | Single product |
| POST | `/api/add-product` | Admin | Create product |
| GET/PUT/DELETE | `/api/categories/{id}` | Admin | Manage categories |
| GET | `/api/orders` | User/Admin | List orders |
| PUT | `/api/orders/{id}/status` | Manager+ | Update order status |
| PUT | `/api/orders/{id}/track` | Manager+ | Set tracking number |
| POST | `/api/payment/callback` | Signed | Payment webhook |
| POST | `/api/cookie-consent` | None | Log consent |

---

## Frontend Conventions

### Templates

- All templates use Jinja2 syntax with `{{ variable }}` and `{% block %}`
- CSS custom properties (variables) for theming: dark (#080A0F bg, #E8580A accent) and light modes
- Theme stored in `localStorage` as `import_theme`, toggled via `data-theme` attribute on `<html>`
- Max layout width: 1280px
- Responsive grid with CSS Grid and Flexbox

### JavaScript

- **No frontend framework** — pure vanilla JS
- `csrf-init.js` must be loaded before any fetch calls; it globally patches `window.fetch`
- `cookie-consent.js` manages consent categories: `necessary`, `functional`, `analytics`
- Яндекс.Метрика is loaded only after analytics consent is granted
- GSAP + ScrollTrigger + Lenis loaded from CDN with SRI hashes

### SRI (Subresource Integrity)

CDN scripts include `integrity` attributes. To regenerate:

```bash
pip install requests
python generate_sri.py   # prints hashes
python apply_sri.py      # patches hashes into templates automatically
```

---

## Development Workflow

### Local Setup with Docker Compose

```bash
# Start PostgreSQL + FastAPI
docker-compose up

# App available at http://localhost:8000
# PostgreSQL at 127.0.0.1:5432
```

### Local Setup without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set env variables manually or via .env
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Making Backend Changes

- All code goes in `backend/main.py` (monolithic — no splitting without major refactor)
- Add new Pydantic models near the top model section
- Add new routes after existing route groups
- Keep raw SQL parameterized: `await conn.fetch("SELECT ... WHERE id = $1", id)`
- Never use `python-jose`; use `PyJWT` (`import jwt`)

### Making Template Changes

- Templates are in `templates/`
- Backup files (`*.bak`) are committed but not served — ignore them
- After editing templates that reference CDN scripts, re-run `apply_sri.py`

---

## Testing

**There is no automated test suite.** Manual testing only via browser or curl.

When making changes:
1. Test auth flows (register, login, logout, token refresh)
2. Test cart operations (add, update quantity, remove)
3. Test admin product/category management
4. Verify CSRF protection is working (check X-CSRF-Token header in DevTools)
5. Check rate limiting is not broken

---

## Deployment

Deployed to **Render.com** using Docker:

```bash
# Build image
docker build -t scooterparts .

# Render.com reads Dockerfile automatically
# Environment variables must be set in Render dashboard
```

The Dockerfile creates required directories:
- `/app/static/uploads` — user-uploaded product images
- `/app/static/images` — processed images
- `/app/static/favicon` — favicon assets

---

## Known Issues & Pending Work

See `AUDIT_FINAL_v11.md` for full history. Current outstanding items:

1. **Legal placeholders** — `legal.html` contains `[ИНН]`, `[НОМЕР УВЕДОМЛЕНИЯ РКН]`, `[ДАТА]` that need real values
2. **Payment gateway** — `payment_stub.html` is a placeholder; real integration (YooKassa/Tinkoff/Stripe) not yet implemented
3. **No test suite** — all testing is manual
4. **`.env` in git** — credentials are committed; rotate all secrets before any public exposure
5. **Monolithic main.py** — 3800+ lines; future refactor should split into routers and service layers

---

## Code Conventions

- **Python style:** No formatter configured; follow PEP 8 manually
- **Async everywhere:** All DB calls and route handlers use `async`/`await`
- **Error handling:** Use `HTTPException` with appropriate status codes
- **Logging:** Use `logger = logging.getLogger(__name__)` (already imported at top of main.py)
- **No ORM:** Write raw SQL with `$1`, `$2` placeholders (asyncpg style, not `%s`)
- **Pydantic v2:** Use `model_config = ConfigDict(...)` not `class Config:`
- **JWT:** Import as `import jwt` (PyJWT); sign with `jwt.encode(...)`, decode with `jwt.decode(...)`

---

## File Naming & Git Conventions

- Commit messages in this repo have historically been single characters (`"1"`) — this is the existing pattern but not recommended
- Branch naming: `claude/<description>-<id>` for AI-generated branches
- Do not commit `.env` files with real credentials
- `.bak` template files are kept for reference but should not be served or edited
