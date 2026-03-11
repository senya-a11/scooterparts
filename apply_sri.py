#!/usr/bin/env python3
"""
Fm TuN — Автоматический пatcher SRI-хешей
==========================================
Запустите ОДИН РАЗ локально перед деплоем на Render:

    pip install requests
    python3 apply_sri.py

Скрипт:
  1. Скачает все CDN-файлы
  2. Вычислит sha384-хеши
  3. Автоматически вставит integrity="..." во все шаблоны
  4. Создаст резервные копии шаблонов (.bak)
"""

import hashlib
import base64
import sys
import shutil
from pathlib import Path

try:
    import requests
except ImportError:
    print("Установите: pip install requests")
    sys.exit(1)

# ── Все CDN-ресурсы проекта ──
CDN_RESOURCES = [
    {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js",
        "templates": ["products.html", "legal.html", "cart.html"],
    },
    {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/CustomEase.min.js",
        "templates": ["products.html", "legal.html", "cart.html"],
    },
    {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/Draggable.min.js",
        "templates": ["products.html"],
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js",
        "templates": ["index.html", "auth.html", "about.html", "tracking.html"],
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/CustomEase.min.js",
        "templates": ["index.html", "auth.html", "tracking.html"],
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/Flip.min.js",
        "templates": ["auth.html"],
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js",
        "templates": ["about.html"],
    },
    {
        "url": "https://unpkg.com/lenis@1.1.14/dist/lenis.min.js",
        "templates": ["about.html"],
    },
]

TEMPLATES_DIR = Path(__file__).parent / "templates"


def compute_sri(data: bytes, algorithm: str = "sha384") -> str:
    h = hashlib.new(algorithm, data).digest()
    return f"{algorithm}-{base64.b64encode(h).decode()}"


def patch_template(filepath: Path, url: str, sri: str) -> bool:
    """Вставляет integrity= в тег <script src="url"> в файле."""
    content = filepath.read_text(encoding="utf-8")

    # Ищем тег без integrity (несколько вариантов написания)
    import re

    # Паттерн: <script src="...url..."> без integrity
    pattern = re.compile(
        r'(<script\b[^>]*\bsrc=["\']' + re.escape(url) + r'["\'][^>]*?)(?!\s+integrity=)(\s*>)',
        re.IGNORECASE
    )

    def replacer(m):
        tag_start = m.group(1)
        tag_end = m.group(2)
        return f'{tag_start} integrity="{sri}" crossorigin="anonymous"{tag_end}'

    new_content, count = pattern.subn(replacer, content)

    if count == 0:
        # Тег уже содержит integrity или не найден
        if f'integrity=' in content and url in content:
            return False  # уже пропатчен
        return False

    filepath.write_text(new_content, encoding="utf-8")
    return True


def main():
    print("=" * 65)
    print("  Fm TuN — SRI Auto-Patcher")
    print("=" * 65)
    print()

    # Создаём резервные копии шаблонов
    backed_up = set()
    print("📦 Создание резервных копий (.bak)...")
    for res in CDN_RESOURCES:
        for tname in res["templates"]:
            tpath = TEMPLATES_DIR / tname
            bak = tpath.with_suffix(".html.bak")
            if tname not in backed_up and tpath.exists() and not bak.exists():
                shutil.copy2(tpath, bak)
                backed_up.add(tname)
    print(f"   Создано: {len(backed_up)} резервных копий\n")

    # Скачиваем файлы и вычисляем хеши
    results = []
    print("📥 Загрузка CDN-файлов и вычисление SRI-хешей...")
    for res in CDN_RESOURCES:
        url = res["url"]
        filename = url.split("/")[-1]
        print(f"   {filename:35}", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            sri = compute_sri(r.content)
            results.append({**res, "sri": sri, "ok": True})
            print(f"✅  {sri[:28]}...")
        except Exception as e:
            results.append({**res, "sri": None, "ok": False, "error": str(e)})
            print(f"❌  {e}")

    print()

    # Патчим шаблоны
    print("🔧 Вставка integrity= в шаблоны...")
    patched_total = 0
    errors = []

    for res in results:
        if not res["ok"]:
            errors.append(f"Пропущен (ошибка загрузки): {res['url']}")
            continue

        url = res["url"]
        sri = res["sri"]

        for tname in res["templates"]:
            tpath = TEMPLATES_DIR / tname
            if not tpath.exists():
                errors.append(f"Не найден шаблон: {tname}")
                continue

            patched = patch_template(tpath, url, sri)
            filename = url.split("/")[-1]
            if patched:
                print(f"   ✅  {tname:25} ← {filename}")
                patched_total += 1
            else:
                print(f"   —   {tname:25}   {filename} (уже есть или не найден)")

    print()
    print("=" * 65)
    if errors:
        print(f"⚠️  Предупреждения:")
        for e in errors:
            print(f"   {e}")
        print()

    if patched_total > 0:
        print(f"✅ Готово! Пропатчено {patched_total} тегов в шаблонах.")
        print(f"   Резервные копии: templates/*.html.bak")
        print()
        print("   Следующий шаг — деплой на Render:")
        print("   1. git add templates/ && git commit -m 'add SRI hashes'")
        print("   2. git push → Render автоматически пересоберёт сервис")
    else:
        print("ℹ️  Все теги уже содержат integrity= или не были найдены.")

    print("=" * 65)


if __name__ == "__main__":
    main()
