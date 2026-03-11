#!/usr/bin/env python3
"""
Fm TuN — Генератор SRI-хешей (Fix A-9)
=======================================
Запустите ОДИН РАЗ на машине с доступом в интернет:

    pip install requests
    python3 generate_sri.py

Скрипт скачает все CDN-файлы, вычислит sha384-хеши и
распечатает готовые атрибуты integrity= для каждого шаблона.

После этого вручную добавьте integrity="sha384-..." crossorigin="anonymous"
к соответствующим <script src="..."> тегам в шаблонах.
"""

import hashlib
import base64
import sys

try:
    import requests
except ImportError:
    print("Установите: pip install requests")
    sys.exit(1)

CDN_RESOURCES = [
    # gsap.min.js — используется везде через cdnjs И jsdelivr (разные файлы — разные хеши!)
    {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js",
        "templates": ["products.html", "legal.html"],
        "tag": '<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js">',
    },
    {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/CustomEase.min.js",
        "templates": ["products.html", "legal.html"],
        "tag": '<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/CustomEase.min.js">',
    },
    {
        "url": "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/Draggable.min.js",
        "templates": ["products.html"],
        "tag": '<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/Draggable.min.js">',
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js",
        "templates": ["index.html", "auth.html", "about.html", "tracking.html"],
        "tag": '<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js">',
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/CustomEase.min.js",
        "templates": ["index.html", "auth.html", "tracking.html"],
        "tag": '<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/CustomEase.min.js">',
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/Flip.min.js",
        "templates": ["auth.html"],
        "tag": '<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/Flip.min.js">',
    },
    {
        "url": "https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js",
        "templates": ["about.html"],
        "tag": '<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js">',
    },
    {
        "url": "https://unpkg.com/lenis@1.1.14/dist/lenis.min.js",
        "templates": ["about.html"],
        "tag": '<script src="https://unpkg.com/lenis@1.1.14/dist/lenis.min.js">',
    },
]


def compute_sri(data: bytes, algorithm: str = "sha384") -> str:
    h = hashlib.new(algorithm, data).digest()
    return f"{algorithm}-{base64.b64encode(h).decode()}"


def main():
    print("=" * 70)
    print("Fm TuN — SRI Hash Generator")
    print("=" * 70)
    print()

    results = []
    for res in CDN_RESOURCES:
        url = res["url"]
        print(f"  Downloading: {url.split('/')[-1]} ...", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            sri = compute_sri(r.content)
            results.append({**res, "sri": sri})
            print(f"✅  {sri[:30]}...")
        except Exception as e:
            print(f"❌  {e}")
            results.append({**res, "sri": None})

    print()
    print("=" * 70)
    print("ЗАМЕНИТЕ теги <script src=...> в шаблонах:")
    print("=" * 70)
    print()

    for res in results:
        if not res["sri"]:
            print(f"⚠️  ПРОПУЩЕН (ошибка загрузки): {res['url']}")
            continue
        old_tag = res["tag"]
        new_tag = old_tag.rstrip(">") + f' integrity="{res["sri"]}" crossorigin="anonymous">'
        print(f"Шаблоны: {', '.join(res['templates'])}")
        print(f"  Было:  {old_tag}</script>")
        print(f"  Стало: {new_tag}</script>")
        print()


if __name__ == "__main__":
    main()
