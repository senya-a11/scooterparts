# backend/security.py - Модуль безопасности

import secrets
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Request, HTTPException, Response
from fastapi.responses import JSONResponse

# ========== CSRF PROTECTION ==========

class CSRFProtection:
    """
    CSRF защита с использованием Double Submit Cookie паттерна.
    """
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self) -> str:
        """Генерация CSRF токена"""
        return secrets.token_urlsafe(32)
    
    def create_hmac_token(self, token: str) -> str:
        """Создание HMAC для дополнительной защиты"""
        return hmac.new(
            self.secret_key.encode(),
            token.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_token(self, token: str, hmac_token: str) -> bool:
        """Проверка CSRF токена"""
        expected_hmac = self.create_hmac_token(token)
        return hmac.compare_digest(expected_hmac, hmac_token)
    
    def set_csrf_cookie(self, response: Response, token: str):
        """Установка CSRF cookie"""
        response.set_cookie(
            key="csrf_token",
            value=token,
            httponly=False,  # Должен быть доступен из JS
            secure=True,  # Только HTTPS
            samesite="strict",
            max_age=3600  # 1 час
        )
    
    def get_csrf_token(self, request: Request) -> Optional[str]:
        """Получение CSRF токена из cookie"""
        return request.cookies.get("csrf_token")
    
    async def validate_csrf(self, request: Request):
        """Валидация CSRF токена для POST/PUT/DELETE запросов"""
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            # Получаем токен из cookie
            cookie_token = self.get_csrf_token(request)
            if not cookie_token:
                raise HTTPException(status_code=403, detail="CSRF token отсутствует в cookie")
            
            # Получаем токен из заголовка
            header_token = request.headers.get("X-CSRF-Token")
            if not header_token:
                # Пробуем получить из тела формы
                if request.headers.get("content-type") == "application/x-www-form-urlencoded":
                    form = await request.form()
                    header_token = form.get("csrf_token")
            
            if not header_token:
                raise HTTPException(status_code=403, detail="CSRF token отсутствует в запросе")
            
            # Сравниваем токены
            if not hmac.compare_digest(cookie_token, header_token):
                raise HTTPException(status_code=403, detail="Недействительный CSRF token")


# ========== HTTPONLY COOKIE AUTHENTICATION ==========

class CookieAuth:
    """
    Безопасная аутентификация через HttpOnly cookies.
    """
    
    @staticmethod
    def set_auth_cookie(
        response: Response,
        token: str,
        max_age: int = 86400  # 24 часа
    ):
        """
        Установка HttpOnly cookie с JWT токеном.
        
        Параметры:
        - httponly=True: Защита от XSS (недоступен из JavaScript)
        - secure=True: Только HTTPS
        - samesite="lax": Защита от CSRF с балансом удобства
        """
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,  # КРИТИЧНО: защита от XSS
            secure=True,  # Только HTTPS
            samesite="lax",  # Баланс между безопасностью и удобством
            max_age=max_age,
            path="/"
        )
    
    @staticmethod
    def set_refresh_cookie(
        response: Response,
        token: str,
        max_age: int = 604800  # 7 дней
    ):
        """Установка refresh token cookie"""
        response.set_cookie(
            key="refresh_token",
            value=token,
            httponly=True,
            secure=True,
            samesite="strict",  # Более строгая защита для refresh token
            max_age=max_age,
            path="/api/auth/refresh"  # Только для refresh endpoint
        )
    
    @staticmethod
    def get_token_from_cookie(request: Request) -> Optional[str]:
        """Получение токена из cookie"""
        return request.cookies.get("access_token")
    
    @staticmethod
    def clear_auth_cookies(response: Response):
        """Удаление auth cookies (logout)"""
        response.delete_cookie(key="access_token", path="/")
        response.delete_cookie(key="refresh_token", path="/api/auth/refresh")


# ========== COOKIE CONSENT MANAGEMENT ==========

class CookieConsent:
    """
    Управление согласием пользователя на использование cookie.
    """
    
    COOKIE_TYPES = {
        "necessary": "Обязательные cookie для работы сайта",
        "functional": "Функциональные cookie для улучшения опыта",
        "analytics": "Аналитические cookie для изучения поведения",
        "marketing": "Маркетинговые cookie для рекламы"
    }
    
    @staticmethod
    def set_consent_cookie(
        response: Response,
        consent: dict,
        max_age: int = 2592000  # 30 дней
    ):
        """
        Сохранение выбора пользователя на 30 дней.
        
        consent = {
            "necessary": True,
            "functional": True,
            "analytics": False,
            "marketing": False
        }
        """
        import json
        consent_str = json.dumps(consent)
        
        response.set_cookie(
            key="cookie_consent",
            value=consent_str,
            httponly=False,  # Должен быть доступен из JS
            secure=True,
            samesite="lax",
            max_age=max_age,
            path="/"
        )
        
        # Сохраняем дату согласия
        response.set_cookie(
            key="cookie_consent_date",
            value=datetime.now().isoformat(),
            httponly=False,
            secure=True,
            samesite="lax",
            max_age=max_age,
            path="/"
        )
    
    @staticmethod
    def get_consent(request: Request) -> Optional[dict]:
        """Получение текущего согласия пользователя"""
        import json
        consent_str = request.cookies.get("cookie_consent")
        if consent_str:
            try:
                return json.loads(consent_str)
            except:
                return None
        return None
    
    @staticmethod
    def is_consent_given(request: Request, cookie_type: str) -> bool:
        """Проверка, дал ли пользователь согласие на конкретный тип cookie"""
        consent = CookieConsent.get_consent(request)
        if not consent:
            return False
        return consent.get(cookie_type, False)
