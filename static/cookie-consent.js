/**
 * IMPORT — Cookie Consent Manager v2.0
 * 152-ФЗ «О персональных данных» + методические рекомендации РКН 2023
 * Изменения: само-инжекция HTML, гранулярные категории, таблица Cookie, backward compat
 */
(function(){
  'use strict';
  var SK='import_cookie_consent_v2', SS='import_cookie_session';
  var TABLE=[
    {n:'access_token',c:'necessary',p:'Авторизация. JWT-токен сессии.',d:'1 час',s:'Cookie (HttpOnly)',w:'Первая сторона'},
    {n:'csrf_token',c:'necessary',p:'Защита от CSRF-атак.',d:'1 час',s:'Cookie',w:'Первая сторона'},
    {n:'import_cookie_consent_v2',c:'necessary',p:'Хранит выбор настроек Cookie (152-ФЗ ст.9).',d:'6 мес.',s:'localStorage',w:'Первая сторона'},
    {n:'import_cookie_session',c:'necessary',p:'ID сессии согласия для серверного лога.',d:'6 мес.',s:'localStorage',w:'Первая сторона'},
    {n:'import_theme',c:'functional',p:'Тема оформления (светлая/тёмная).',d:'12 мес.',s:'localStorage',w:'Первая сторона'},
    {n:'_ym_*',c:'analytics',p:'Яндекс.Метрика — анонимная статистика.',d:'До 1 года',s:'Cookie',w:'Яндекс (3-я сторона)'}
  ];
  var CATS={necessary:{l:'Необходимые',color:'#00D4FF'},functional:{l:'Функциональные',color:'#A78BFA'},analytics:{l:'Аналитические',color:'#34D399'}};

  function sid(){var i=localStorage.getItem(SS);if(!i){i='sess_'+Date.now()+'_'+Math.random().toString(36).slice(2,10);localStorage.setItem(SS,i);}return i;}
  function load(){
    try{
      var r=localStorage.getItem(SK);if(!r)return null;
      var o=JSON.parse(r);
      if(!o.ts||Date.now()-o.ts>180*24*3600*1000){localStorage.removeItem(SK);return null;}
      if(o.type&&!o.categories){o.categories={necessary:true,functional:o.type==='all',analytics:o.type==='all'};}
      return o;
    }catch(e){return null;}
  }
  function save(cats){
    var rec={categories:cats,ts:Date.now(),session:sid()};
    localStorage.setItem(SK,JSON.stringify(rec));
    var t=(cats.analytics&&cats.functional)?'all':(cats.functional||cats.analytics)?'partial':'necessary';
    fetch('/api/cookie-consent',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({consent_type:t,session_id:rec.session})}).catch(function(){});
  }
  function apply(cats){
    window.COOKIE_CONSENT=cats;
    document.dispatchEvent(new CustomEvent(cats.analytics?'cookieConsentGranted':'cookieConsentDeclined',{detail:{categories:cats}}));
  }
  function rows(){
    return TABLE.map(function(c){
      var ci=CATS[c.c];
      return '<tr><td><code style="color:#00D4FF;font-size:.72rem;">'+c.n+'</code></td>'
        +'<td><span style="color:'+ci.color+';font-size:.7rem;font-weight:600;">'+ci.l+'</span></td>'
        +'<td style="font-size:.72rem;color:#9AA3B8;max-width:200px;">'+c.p+'</td>'
        +'<td style="font-size:.72rem;color:#9AA3B8;white-space:nowrap;">'+c.d+'</td>'
        +'<td style="font-size:.72rem;color:#9AA3B8;white-space:nowrap;">'+c.w+'</td></tr>';
    }).join('');
  }
  function tog(k){
    var i=CATS[k],d=k==='functional'?'Настройки темы оформления':'Анонимная статистика посещаемости';
    return '<label class="ck-tr" for="ck-'+k+'">'
      +'<div class="ck-ti"><span class="ck-tt" style="color:'+i.color+';">'+i.l+'</span>'
      +'<span class="ck-td">'+d+'</span></div>'
      +'<div class="ck-sw"><input type="checkbox" id="ck-'+k+'" class="ck-cb" data-category="'+k+'"><span class="ck-sl"></span></div>'
      +'</label>';
  }
  function injectCSS(){
    if(document.getElementById('ck-styles'))return;
    var s=document.createElement('style');s.id='ck-styles';
    s.textContent='.ck-banner{position:fixed;bottom:0;left:0;right:0;z-index:9900;transform:translateY(100%);transition:transform .35s cubic-bezier(.16,1,.3,1);}'
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
      +'.ck-exp.open{max-height:720px;}'
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
      +'@media(max-width:700px){.ck-main{flex-direction:column;align-items:flex-start;}.ck-act{width:100%;}.ck-ba,.ck-bn{flex:1;text-align:center;}}';
    document.head.appendChild(s);
  }
  function injectHTML(){
    if(document.getElementById('cookieBanner'))return;
    var h='<div id="cookieBanner" class="ck-banner" role="dialog" aria-modal="true" aria-label="Настройки Cookie">'
      +'<div class="ck-panel">'
      +'<div class="ck-main">'
      +'<div class="ck-lft"><span class="ck-ico" aria-hidden="true">🍪</span>'
      +'<div><strong class="ck-ttl">Мы используем Cookie-файлы</strong>'
      +'<p class="ck-sub">Необходимые Cookie обеспечивают работу сайта. С вашего согласия мы используем дополнительные категории.</p></div></div>'
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
      +'<div class="ck-tit">Полный список используемых Cookie</div>'
      +'<div style="overflow-x:auto;"><table class="ck-tbl"><thead><tr><th>Имя</th><th>Категория</th><th>Цель</th><th>Срок</th><th>Сторона</th></tr></thead><tbody>'+rows()+'</tbody></table></div>'
      +'<div class="ck-ea"><button class="ck-bsv" id="cookieSave">Сохранить настройки</button>'
      +'<a class="ck-pl" href="/privacy-policy#cookies" target="_blank" rel="noopener">Политика конфиденциальности →</a></div>'
      +'</div></div></div></div>';
    document.body.insertAdjacentHTML('beforeend',h);
  }
  function show(){var b=document.getElementById('cookieBanner');if(b){b.classList.add('show');b.removeAttribute('aria-hidden');}}
  function hide(){var b=document.getElementById('cookieBanner');if(b){b.classList.remove('show');b.setAttribute('aria-hidden','true');}}
  function expand(on){
    var e=document.getElementById('ckExpanded'),bt=document.getElementById('cookieSettings');
    if(!e)return;
    if(on){e.classList.add('open');e.removeAttribute('aria-hidden');if(bt)bt.textContent='Скрыть ↑';}
    else{e.classList.remove('open');e.setAttribute('aria-hidden','true');if(bt)bt.textContent='Настроить ↓';}
  }
  function readCB(){var c={necessary:true};document.querySelectorAll('.ck-cb[data-category]').forEach(function(b){c[b.dataset.category]=b.checked;});return c;}
  function setCB(c){document.querySelectorAll('.ck-cb[data-category]').forEach(function(b){b.checked=!!(c&&c[b.dataset.category]);});}
  function init(){
    injectCSS();injectHTML();
    var sv=load();
    if(sv&&sv.categories){apply(sv.categories);setCB(sv.categories);}else{show();}
    document.addEventListener('click',function(e){
      var id=e.target.id;
      if(id==='cookieAccept'){var c={necessary:true,functional:true,analytics:true};save(c);apply(c);setCB(c);hide();if(typeof toast==='function')toast('success','Cookie приняты');}
      else if(id==='cookieDecline'){var c={necessary:true,functional:false,analytics:false};save(c);apply(c);setCB(c);hide();}
      else if(id==='cookieSettings'){var ex=document.getElementById('ckExpanded');expand(!ex.classList.contains('open'));}
      else if(id==='cookieSave'){var c=readCB();save(c);apply(c);hide();if(typeof toast==='function')toast('success','Настройки Cookie сохранены');}
    });
  }
  if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{init();}
  window.cookieConsent={
    reset:function(){localStorage.removeItem(SK);setCB({necessary:true,functional:false,analytics:false});show();expand(false);},
    get:function(){return load();},
    isGranted:function(cat){if(cat==='necessary')return true;var s=load();return!!(s&&s.categories&&s.categories[cat]);}
  };
})();
