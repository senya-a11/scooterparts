/**
 * IMPORT — Cookie Consent Manager v3.0
 * ─────────────────────────────────────────────────────────────────
 * Соответствует:
 *   - ФЗ от 27.07.2006 № 152-ФЗ «О персональных данных»
 *   - Методические рекомендации Роскомнадзора по Cookie (2023)
 *   - Позиция РКН: Cookie-идентификаторы = персональные данные
 *
 * Яндекс.Метрика:
 *   - Скрипт НЕ загружается до явного согласия на аналитику
 *   - При согласии: динамически инжектирует тег <script>
 *   - При отзыве: деактивирует счётчик, очищает _ym_* Cookie
 *
 * НАСТРОЙТЕ YM_COUNTER_ID перед запуском!
 * ─────────────────────────────────────────────────────────────────
 */
(function(){
  'use strict';

  // ⚙️ КОНФИГУРАЦИЯ — ЗАПОЛНИТЕ!
  var YM_COUNTER_ID = 0;   // ← УКАЖИТЕ НОМЕР ВАШЕГО СЧЁТЧИКА (число, например 12345678)
  var YM_OPTIONS = {
    clickmap: true,
    trackLinks: true,
    accurateTrackBounce: true,
    webvisor: false          // ← true для Вебвизора (усиленные требования к согласию)
  };

  var SK = 'import_cookie_consent_v2';
  var SS = 'import_cookie_session';

  var TABLE = [
    {n:'access_token',              c:'necessary', p:'JWT-токен сессии (HttpOnly).',                     d:'1 час',   s:'Cookie (HttpOnly)', w:'Первая сторона'},
    {n:'csrf_token',                c:'necessary', p:'Защита от CSRF-атак.',                             d:'1 час',   s:'Cookie',            w:'Первая сторона'},
    {n:'import_cookie_consent_v2',  c:'necessary', p:'Хранит настройки Cookie (ст. 9 ФЗ-152).',         d:'6 мес.',  s:'localStorage',      w:'Первая сторона'},
    {n:'import_cookie_session',     c:'necessary', p:'ID сессии для аудит-лога согласий.',               d:'6 мес.',  s:'localStorage',      w:'Первая сторона'},
    {n:'import_theme',              c:'functional',p:'Тема оформления (тёмная/светлая).',                d:'12 мес.', s:'localStorage',      w:'Первая сторона'},
    {n:'_ym_uid',                   c:'analytics', p:'Яндекс.Метрика: ID посетителя. Только с согласия.',d:'1 год',  s:'Cookie',            w:'ООО «Яндекс»'},
    {n:'_ym_d',                     c:'analytics', p:'Яндекс.Метрика: дата первого визита. Только с согласия.',d:'1 год',s:'Cookie',         w:'ООО «Яндекс»'},
    {n:'_ym_isad',                  c:'analytics', p:'Яндекс.Метрика: блокировщики рекламы. Только с согласия.',d:'2 дня',s:'Cookie',        w:'ООО «Яндекс»'},
    {n:'_ym_visorc',                c:'analytics', p:'Яндекс.Метрика: данные Вебвизора. Только с согласия.',d:'30 мин.',s:'Cookie',          w:'ООО «Яндекс»'}
  ];

  var CATS = {
    necessary:  {l:'Необходимые',    color:'#00D4FF'},
    functional: {l:'Функциональные', color:'#A78BFA'},
    analytics:  {l:'Аналитические',  color:'#34D399'}
  };

  function sid(){
    var i = localStorage.getItem(SS);
    if(!i){ i = 'sess_'+Date.now()+'_'+Math.random().toString(36).slice(2,10); localStorage.setItem(SS,i); }
    return i;
  }

  function load(){
    try {
      var r = localStorage.getItem(SK); if(!r) return null;
      var o = JSON.parse(r);
      if(!o.ts || Date.now()-o.ts > 180*24*3600*1000){ localStorage.removeItem(SK); return null; }
      if(o.type && !o.categories){
        o.categories = {necessary:true, functional:o.type==='all', analytics:o.type==='all'};
      }
      return o;
    } catch(e){ return null; }
  }

  function save(cats){
    var rec = {categories:cats, ts:Date.now(), session:sid()};
    localStorage.setItem(SK, JSON.stringify(rec));
    var t = (cats.analytics&&cats.functional)?'all':(cats.functional||cats.analytics)?'partial':'necessary';
    fetch('/api/cookie-consent',{method:'POST',credentials:'include',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({consent_type:t, session_id:rec.session})
    }).catch(function(){});
  }

  function apply(cats){
    window.COOKIE_CONSENT = cats;
    if(cats.analytics){ loadYandexMetrika(); }
    else              { disableYandexMetrika(); }
    document.dispatchEvent(new CustomEvent(
      cats.analytics ? 'cookieConsentGranted' : 'cookieConsentDeclined',
      {detail:{categories:cats}}
    ));
  }

  /* ════════════════════════════════════════════════════════════
     ЯНДЕКС.МЕТРИКА — ГЕЙТ СООТВЕТСТВИЯ ФЗ-152 / РКН 2023
     Скрипт НЕ грузится без явного согласия пользователя.
  ═════════════════════════════════════════════════════════════*/
  var ymLoaded = false;

  function loadYandexMetrika(){
    if(!YM_COUNTER_ID || ymLoaded) return;
    ymLoaded = true;
    window.ym = window.ym || function(){ (window.ym.a=window.ym.a||[]).push(arguments); };
    window.ym.l = 1 * new Date();
    var s = document.createElement('script');
    s.async = true;
    s.src = 'https://mc.yandex.ru/metrika/tag.js';
    s.onload = function(){ ym(YM_COUNTER_ID, 'init', YM_OPTIONS); };
    var f = document.getElementsByTagName('script')[0];
    f.parentNode.insertBefore(s, f);
  }

  function disableYandexMetrika(){
    ymLoaded = false;
    var names = ['_ym_uid','_ym_d','_ym_isad','_ym_visorc','_ym_metrika_enabled','_ym_wv'];
    names.forEach(function(n){
      document.cookie = n+'=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; domain='+location.hostname;
      document.cookie = n+'=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';
    });
  }

  function rows(){
    return TABLE.map(function(c){
      var ci = CATS[c.c];
      return '<tr><td><code style="color:#00D4FF;font-size:.72rem;">'+c.n+'</code></td>'
        +'<td><span style="color:'+ci.color+';font-size:.7rem;font-weight:600;">'+ci.l+'</span></td>'
        +'<td style="font-size:.72rem;color:#9AA3B8;max-width:200px;">'+c.p+'</td>'
        +'<td style="font-size:.72rem;color:#9AA3B8;white-space:nowrap;">'+c.d+'</td>'
        +'<td style="font-size:.72rem;color:#9AA3B8;white-space:nowrap;">'+c.w+'</td></tr>';
    }).join('');
  }

  function tog(k){
    var i = CATS[k];
    var d = k==='functional'?'Настройки темы интерфейса':'Анонимная статистика посещаемости (Яндекс.Метрика)';
    return '<label class="ck-tr" for="ck-'+k+'">'
      +'<div class="ck-ti"><span class="ck-tt" style="color:'+i.color+';">'+i.l+'</span>'
      +'<span class="ck-td">'+d+'</span></div>'
      +'<div class="ck-sw"><input type="checkbox" id="ck-'+k+'" class="ck-cb" data-category="'+k+'"><span class="ck-sl"></span></div>'
      +'</label>';
  }

  function injectCSS(){
    if(document.getElementById('ck-styles')) return;
    var s = document.createElement('style'); s.id='ck-styles';
    s.textContent=
      '.ck-banner{position:fixed;bottom:0;left:0;right:0;z-index:9900;transform:translateY(100%);transition:transform .35s cubic-bezier(.16,1,.3,1);}'
      +'.ck-banner.show{transform:translateY(0);}'
      +'.ck-panel{background:rgba(8,10,15,.98);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-top:1px solid rgba(0,212,255,.15);}'
      +'.ck-main{display:flex;align-items:center;justify-content:space-between;gap:1.25rem;flex-wrap:wrap;max-width:1200px;margin:0 auto;padding:.875rem 2rem;}'
      +'.ck-lft{display:flex;align-items:center;gap:.875rem;flex:1;min-width:0;}'
      +'.ck-ico{font-size:1.4rem;flex-shrink:0;}'
      +'.ck-ttl{display:block;font-size:.875rem;font-weight:700;color:#F0F4F8;margin-bottom:.15rem;}'
      +'.ck-sub{font-size:.75rem;color:#7B8599;line-height:1.5;margin:0;}'
      +'.ck-act{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;flex-shrink:0;}'
      +'.ck-ba{background:#00D4FF;color:#080A0F;border:none;padding:.45rem 1.1rem;border-radius:6px;font-size:.8125rem;font-weight:700;cursor:pointer;transition:opacity .2s;white-space:nowrap;}'
      +'.ck-ba:hover{opacity:.85;}'
      +'.ck-bn{background:transparent;color:#7B8599;border:1px solid rgba(255,255,255,.12);padding:.45rem .9rem;border-radius:6px;font-size:.8125rem;cursor:pointer;transition:all .2s;white-space:nowrap;}'
      +'.ck-bn:hover{border-color:rgba(255,255,255,.3);color:#F0F4F8;}'
      +'.ck-bs{background:transparent;color:#3E4560;border:none;padding:.4rem .6rem;font-size:.75rem;cursor:pointer;text-decoration:underline;white-space:nowrap;transition:color .2s;}'
      +'.ck-bs:hover{color:#7B8599;}'
      +'.ck-exp{max-height:0;overflow:hidden;transition:max-height .45s cubic-bezier(.16,1,.3,1);}'
      +'.ck-exp.open{max-height:800px;}'
      +'.ck-ei{max-width:1200px;margin:0 auto;padding:.875rem 2rem 1.25rem;border-top:1px solid rgba(0,212,255,.1);}'
      +'.ck-tgs{display:flex;flex-direction:column;gap:.4rem;margin-bottom:.875rem;}'
      +'.ck-tr{display:flex;align-items:center;justify-content:space-between;padding:.55rem .75rem;border-radius:7px;background:rgba(255,255,255,.03);cursor:pointer;gap:1rem;user-select:none;}'
      +'.ck-tr-d{cursor:default;opacity:.65;}'
      +'.ck-ti{flex:1;}'
      +'.ck-tt{display:block;font-size:.8rem;font-weight:700;margin-bottom:.1rem;}'
      +'.ck-td{font-size:.71rem;color:#7B8599;}'
      +'.ck-sw{position:relative;width:38px;height:22px;flex-shrink:0;}'
      +'.ck-cb{position:absolute;opacity:0;width:0;height:0;}'
      +'.ck-sl{position:absolute;inset:0;background:rgba(255,255,255,.12);border-radius:22px;transition:background .2s;}'
      +'.ck-sl::before{content:"";position:absolute;width:16px;height:16px;left:3px;top:3px;background:#fff;border-radius:50%;transition:transform .2s;}'
      +'.ck-cb:checked+.ck-sl{background:#00D4FF;}'
      +'.ck-cb:checked+.ck-sl::before{transform:translateX(16px);}'
      +'.ck-aon{display:inline-flex;align-items:center;width:38px;height:22px;background:#00D4FF;border-radius:22px;flex-shrink:0;}'
      +'.ck-aon::before{content:"";width:16px;height:16px;background:#fff;border-radius:50%;margin-left:19px;}'
      +'.ck-tit{font-size:.72rem;font-weight:700;color:#7B8599;margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.05em;}'
      +'.ck-tbl{width:100%;border-collapse:collapse;}'
      +'.ck-tbl th{text-align:left;padding:.35rem .5rem;font-size:.7rem;font-weight:600;color:#7B8599;border-bottom:1px solid rgba(0,212,255,.1);white-space:nowrap;}'
      +'.ck-tbl td{padding:.35rem .5rem;border-bottom:1px solid rgba(255,255,255,.03);vertical-align:top;}'
      +'.ck-ea{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.875rem;}'
      +'.ck-bsv{background:transparent;color:#00D4FF;border:1px solid rgba(0,212,255,.3);padding:.45rem 1.1rem;border-radius:6px;font-size:.8125rem;font-weight:600;cursor:pointer;transition:all .2s;}'
      +'.ck-bsv:hover{background:rgba(0,212,255,.08);}'
      +'.ck-pl{font-size:.74rem;color:#7B8599;text-decoration:none;transition:color .2s;}'
      +'.ck-pl:hover{color:#00D4FF;}'
      +'.ck-ym-note{font-size:.7rem;color:#5E6878;margin-top:.35rem;padding:.35rem .5rem;background:rgba(52,211,153,.05);border-radius:4px;border-left:2px solid rgba(52,211,153,.3);}'
      +'@media(max-width:700px){.ck-main{flex-direction:column;align-items:flex-start;}.ck-act{width:100%;}.ck-ba,.ck-bn{flex:1;text-align:center;}}';
    document.head.appendChild(s);
  }

  function injectHTML(){
    if(document.getElementById('cookieBanner')) return;
    var h = '<div id="cookieBanner" class="ck-banner" role="dialog" aria-modal="true" aria-label="Настройки Cookie">'
      +'<div class="ck-panel">'
      +'<div class="ck-main">'
      +'<div class="ck-lft"><span class="ck-ico" aria-hidden="true">🍪</span>'
      +'<div><strong class="ck-ttl">Мы используем Cookie-файлы</strong>'
      +'<p class="ck-sub">Необходимые Cookie обеспечивают работу сайта. С вашего согласия подключаем Яндекс.Метрику для анонимной статистики.</p></div></div>'
      +'<div class="ck-act">'
      +'<button class="ck-ba" id="cookieAccept">Принять все</button>'
      +'<button class="ck-bn" id="cookieDecline">Только необходимые</button>'
      +'<button class="ck-bs" id="cookieSettings">Настроить ↓</button>'
      +'</div></div>'
      +'<div class="ck-exp" id="ckExpanded" aria-hidden="true">'
      +'<div class="ck-ei">'
      +'<div class="ck-tgs">'
      +'<div class="ck-tr ck-tr-d"><div class="ck-ti"><span class="ck-tt" style="color:#00D4FF;">Необходимые</span><span class="ck-td">Авторизация, корзина, CSRF. Отключить нельзя.</span></div><span class="ck-aon" aria-label="Всегда включены"></span></div>'
      +tog('functional')+tog('analytics')
      +'</div>'
      +'<div class="ck-ym-note">⚠ Яндекс.Метрика относится к «Аналитическим» Cookie и загружается только с явного согласия (ФЗ-152, рекомендации РКН 2023).</div>'
      +'<div class="ck-tit" style="margin-top:.875rem;">Все используемые Cookie</div>'
      +'<div style="overflow-x:auto;"><table class="ck-tbl"><thead><tr><th>Имя</th><th>Категория</th><th>Цель</th><th>Срок</th><th>Сторона</th></tr></thead><tbody>'+rows()+'</tbody></table></div>'
      +'<div class="ck-ea"><button class="ck-bsv" id="cookieSave">Сохранить настройки</button>'
      +'<a class="ck-pl" href="/legal?doc=cookies" target="_blank" rel="noopener">Политика Cookie →</a>'
      +'<a class="ck-pl" href="/legal?doc=privacy" target="_blank" rel="noopener">Конфиденциальность →</a></div>'
      +'</div></div></div></div>';
    document.body.insertAdjacentHTML('beforeend', h);
  }

  function show(){ var b=document.getElementById('cookieBanner'); if(b){b.classList.add('show');b.removeAttribute('aria-hidden');} }
  function hide(){ var b=document.getElementById('cookieBanner'); if(b){b.classList.remove('show');b.setAttribute('aria-hidden','true');} }
  function expand(on){
    var e=document.getElementById('ckExpanded'),bt=document.getElementById('cookieSettings');
    if(!e)return;
    if(on){e.classList.add('open');e.removeAttribute('aria-hidden');if(bt)bt.textContent='Скрыть ↑';}
    else{e.classList.remove('open');e.setAttribute('aria-hidden','true');if(bt)bt.textContent='Настроить ↓';}
  }
  function readCB(){ var c={necessary:true}; document.querySelectorAll('.ck-cb[data-category]').forEach(function(b){c[b.dataset.category]=b.checked;}); return c; }
  function setCB(c){ document.querySelectorAll('.ck-cb[data-category]').forEach(function(b){b.checked=!!(c&&c[b.dataset.category]);}); }

  function init(){
    injectCSS(); injectHTML();
    var sv = load();
    if(sv && sv.categories){ apply(sv.categories); setCB(sv.categories); }
    else{ show(); }
    document.addEventListener('click', function(e){
      var id = e.target.id;
      if(id==='cookieAccept'){
        var c={necessary:true,functional:true,analytics:true};
        save(c);apply(c);setCB(c);hide();
        if(typeof toast==='function')toast('success','Cookie приняты');
      } else if(id==='cookieDecline'){
        var c={necessary:true,functional:false,analytics:false};
        save(c);apply(c);setCB(c);hide();
      } else if(id==='cookieSettings'){
        var ex=document.getElementById('ckExpanded');
        expand(!ex.classList.contains('open'));
      } else if(id==='cookieSave'){
        var c=readCB();
        save(c);apply(c);hide();
        if(typeof toast==='function')toast('success','Настройки сохранены');
      }
    });
  }

  if(document.readyState==='loading'){ document.addEventListener('DOMContentLoaded',init); }
  else{ init(); }

  window.cookieConsent = {
    reset: function(){ localStorage.removeItem(SK); setCB({necessary:true,functional:false,analytics:false}); expand(false); show(); },
    get: function(){ return load(); },
    isGranted: function(cat){ if(cat==='necessary')return true; var s=load(); return!!(s&&s.categories&&s.categories[cat]); },
    loadYM: loadYandexMetrika,
    getYMId: function(){ return YM_COUNTER_ID; }
  };
})();
