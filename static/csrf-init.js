/**
 * IMPORT — CSRF Auto-Injector v1.0
 *
 * Перехватывает window.fetch глобально.
 * Автоматически добавляет X-CSRF-Token к любому POST/PUT/DELETE/PATCH
 * на тот же домен. Не требует изменений в существующем коде.
 *
 * Порядок инициализации:
 *   1. Этот файл загружается в <head> без defer — сразу переопределяет fetch
 *   2. Одновременно запускает prefetch токена (кешируется в Promise)
 *   3. Любой последующий мутирующий запрос автоматически получает токен
 */
(function () {
  'use strict';

  var _tokenPromise = null;

  /** Получить CSRF-токен (с кешированием, один запрос на всю страницу) */
  function _getToken() {
    if (!_tokenPromise) {
      _tokenPromise = window._nativeFetch('/api/csrf-token', { credentials: 'include' })
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (d) { return d.csrf_token || ''; })
        .catch(function () { return ''; });
    }
    return _tokenPromise;
  }

  /** Проверить, что URL — тот же домен (не внешний) */
  function _isSameOrigin(url) {
    if (typeof url !== 'string') {
      // Request object
      try { url = url.url; } catch (e) { return false; }
    }
    // Относительный URL — всегда тот же домен
    if (url.charAt(0) === '/') return true;
    try {
      return new URL(url).origin === location.origin;
    } catch (e) {
      return false;
    }
  }

  // Сохраняем оригинальный fetch
  window._nativeFetch = window.fetch.bind(window);

  /**
   * Переопределяем window.fetch.
   * Для мутирующих запросов на тот же домен — добавляем X-CSRF-Token.
   * Все остальные запросы — проксируем без изменений.
   */
  window.fetch = function (resource, options) {
    options = options || {};
    var method = (options.method || 'GET').toUpperCase();
    var mutating = ['POST', 'PUT', 'DELETE', 'PATCH'].indexOf(method) !== -1;

    if (mutating && _isSameOrigin(resource)) {
      return _getToken().then(function (token) {
        if (token) {
          // Поддержка как plain-object headers, так и Headers instance
          if (options.headers instanceof Headers) {
            options.headers.set('X-CSRF-Token', token);
          } else {
            options.headers = options.headers || {};
            options.headers['X-CSRF-Token'] = token;
          }
        }
        return window._nativeFetch(resource, options);
      });
    }

    return window._nativeFetch(resource, options);
  };

  // Prefetch — запускаем заранее, не ждём пользовательского действия
  _getToken();

  /**
   * Fix A-11: Тихое обновление токена за 10 минут до истечения сессии.
   * JWT живёт 60 минут → таймер срабатывает через 50 минут.
   * Если вкладка была закрыта и открыта — срабатывает сразу при наличии cookie.
   */
  (function scheduleRefresh() {
    var REFRESH_AFTER_MS = 50 * 60 * 1000; // 50 минут
    function doRefresh() {
      window.fetch('/api/refresh', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' }
      })
      .then(function(r) {
        if (r.ok) {
          // Сброс кеша CSRF-токена — после обновления куки нужен новый
          _tokenPromise = null;
          _getToken();
          // Планируем следующее обновление
          setTimeout(doRefresh, REFRESH_AFTER_MS);
        }
        // 401 = сессия истекла — ничего не делаем, пусть пользователь войдёт заново
      })
      .catch(function() { /* сеть недоступна — тихо игнорируем */ });
    }
    setTimeout(doRefresh, REFRESH_AFTER_MS);
  })();


})();
