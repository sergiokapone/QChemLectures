#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Генерує PyscfBook/index.html — головну сторінку онлайн-версії посібника (GitHub Pages).

Запуск (після збірки html/build-html.ps1):  python -I make_site.py [--online ../online]

Зміст беруть із бічної панелі головної сторінки самої онлайн-книги (../online/index.html): тоді частини, назви розділів,
нумерація і посилання завжди збігаються з книгою (імена сторінок — після publish_html.py). Сторінка самодостатня:
стилі вбудовані, тема (світла/темна) спільна з книгою (localStorage 'pyscfbook-theme').
"""
import re
import sys
from html import escape, unescape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONLINE = HERE.parent / "online"
if "--online" in sys.argv:
    ONLINE = (HERE / sys.argv[sys.argv.index("--online") + 1]).resolve()
OUT = HERE / "index.html"
SITE = "https://sergiokapone.github.io/QChemLectures/PyscfBook/"
REPO = "https://github.com/sergiokapone/QChemLectures/tree/master/PyscfBook"
PDF = "pyscfBook.pdf"

LINK = re.compile(r'<p>\s*<a href="([^"]+)" class="(toc(?:part|chapter|section)|linkhome)"\s*>(.*?)</a>\s*</p>', re.S)
NUM = re.compile(r'<span class="sectionnumber">(.*?)</span>\s*<span class="toctitle">(.*?)</span>', re.S)


def clean(s: str) -> str:
    s = re.sub(r"<kbd>(.*?)</kbd>", r"<code>\1</code>", s, flags=re.S)
    s = re.sub(r"<(?!/?code)[^>]+>", "", s)
    s = unescape(s).replace(" ", " ").replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return escape(s).replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>")


def parse():
    page = ONLINE / "index.html"
    if not page.exists():
        sys.exit("Немає %s — спершу зберіть онлайн-версію (html/build-html.ps1)." % page)
    html = page.read_text(encoding="utf-8")
    i = html.find('class="sidetoccontents"')
    if i < 0:
        sys.exit("У %s немає бічного змісту." % page)
    html = html[i:html.find("</nav>", i)]
    parts, cur_part, cur_ch = [], None, None
    front, back = [], []
    for href, kind, body in LINK.findall(html):
        if kind == "linkhome":
            continue
        m = NUM.search(body)
        num, title = (m.group(1).strip(), clean(m.group(2))) if m else ("", clean(body))
        url = "../online/" + href
        if kind == "tocpart":
            cur_part = {"num": num, "title": title, "url": url, "chapters": []}
            parts.append(cur_part)
            cur_ch = None
        elif kind == "tocchapter":
            ch = {"num": num, "title": title, "url": url, "subs": []}
            if num and cur_part is not None:
                cur_part["chapters"].append(ch)
                cur_ch = ch
            else:                                   # Передмова, Додатки, Література: без номера
                (back if parts else front).append(ch)
                cur_ch = ch
        elif kind == "tocsection" and cur_ch is not None:
            cur_ch["subs"].append((num, title, url))
    return front, parts, back


def render(front, parts, back) -> str:
    out = []
    a = out.append

    def chapter(ch, numbered=True):
        a("<li class=\"ch\"><details>")
        n = '<span class="num">%s</span>' % escape(ch["num"]) if ch["num"] else ""
        a('<summary>%s<a class="ch-title" href="%s">%s</a></summary>' % (n, ch["url"], ch["title"]))
        if ch["subs"]:
            a("<ul class=\"subs\">")
            for lab, t, u in ch["subs"]:
                l = '<span class="snum">%s</span>' % escape(lab) if lab else ""
                a('<li><a href="%s">%s%s</a></li>' % (u, l, t))
            a("</ul>")
        a("</details></li>")

    if front:
        a('<ul class="chapters front">')
        for ch in front:
            a('<li class="ch plain"><a class="ch-title" href="%s">%s</a></li>' % (ch["url"], ch["title"]))
        a("</ul>")
    for p in parts:
        a('<section class="part">')
        a('<h3><span class="part-num">Частина %s</span><a href="%s">%s</a></h3>' % (escape(p["num"]), p["url"], p["title"]))
        a('<ul class="chapters">')
        for ch in p["chapters"]:
            chapter(ch)
        a("</ul></section>")
    if back:
        a('<section class="part appendix"><h3><span class="part-num">Додатково</span>Додатки та література</h3><ul class="chapters">')
        for ch in back:
            if ch["subs"]:
                chapter(ch)
            else:
                a('<li class="ch plain"><a class="ch-title" href="%s">%s</a></li>' % (ch["url"], ch["title"]))
        a("</ul></section>")
    return "\n".join(out)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Квантово-механічні методи обчислення. Використання PySCF | Навчальний посібник</title>
<meta name="description" content="Навчальний посібник з квантово-механічних обчислень атомів і молекул у PySCF: метод Хартрі–Фока, пост-HF, DFT, молекулярні властивості, коливання, збуджені стани. Код прикладів можна копіювати.">
<meta name="robots" content="index, follow">
<link rel="canonical" href="@SITE@">
<meta property="og:title" content="Квантово-механічні методи обчислення. Використання PySCF">
<meta property="og:description" content="Навчальний посібник: теорія й готові приклади коду для PySCF.">
<meta property="og:type" content="website">
<meta property="og:url" content="@SITE@">
<script>(function(){var r=document.documentElement;try{var t=localStorage.getItem('pyscfbook-theme');if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches)){r.classList.add('dark');}}catch(e){}})();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400..700&family=JetBrains+Mono:wght@400;500;700&family=Literata:ital,opsz,wght@0,7..72,300..700;1,7..72,300..700&display=swap">
<style>
:root{--ink:#0d1b36;--navy:#002e64;--indigo:#3333b2;--teal:#006766;--muted:#58637a;--page:#fdfdfe;--desk:#e9edf5;--rule:#d5dbe8;
  --tint:#e8eef7;--card:#fff;--link:#2b2bb0;--gold:#e0b400;--serif:"Literata",Georgia,serif;--sans:"Golos Text","Segoe UI",Roboto,Arial,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Consolas,monospace;--shadow:0 1px 2px rgba(13,27,54,.08),0 14px 34px -14px rgba(13,27,54,.28)}
:root.dark{--ink:#e6ebf6;--navy:#9fb8e8;--indigo:#9aa2ff;--teal:#5fd0cb;--muted:#9aa6bf;--page:#121a2c;--desk:#0b1120;--rule:#27324c;
  --tint:#18233d;--card:#16213a;--link:#9fa6ff;--shadow:0 1px 2px rgba(0,0,0,.4),0 16px 38px -16px rgba(0,0,0,.7)}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--desk);color:var(--ink);font:1.02rem/1.65 var(--serif)}
a{color:var(--link)}
.wrap{max-width:62rem;margin:0 auto;padding:0 1.2rem 3rem}
.hero{position:relative;margin:0 -1.2rem;padding:3.2rem 1.2rem 2.6rem;color:#fff;text-align:center;overflow:hidden;
  background:radial-gradient(1200px 380px at 50% -10%,#3a3ad0 0%,transparent 60%),linear-gradient(160deg,#00224b 0%,#002e64 55%,#14307a 100%)}
.hero::before{content:"";position:absolute;inset:0;opacity:.18;
  background:radial-gradient(circle at 20% 30%,#fff 0 1.5px,transparent 2px),radial-gradient(circle at 78% 62%,#fff 0 1.5px,transparent 2px);
  background-size:70px 70px,110px 110px}
.hero>*{position:relative}
.tools{position:absolute;right:1rem;top:.9rem}
.tbtn{font:600 .78rem var(--sans);color:#fff;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.35);border-radius:8px;padding:.4rem .75rem;cursor:pointer}
.tbtn:hover{background:rgba(255,255,255,.22)}
.eyebrow{font:600 .78rem var(--sans);letter-spacing:.18em;text-transform:uppercase;color:#bcd0ff;margin:0 0 .8rem}
h1{font:700 clamp(1.8rem,4.6vw,2.9rem)/1.15 var(--sans);margin:0 0 .5rem;letter-spacing:-.01em}
.sub{font:500 clamp(1.05rem,2.4vw,1.35rem) var(--sans);color:#dbe6ff;margin:0}
.sub code{font:600 .95em var(--mono);background:rgba(255,255,255,.14);padding:.05em .4em;border-radius:6px}
.actions{display:flex;flex-wrap:wrap;gap:.8rem;justify-content:center;margin:1.9rem 0 .4rem}
.btn{display:inline-flex;align-items:center;gap:.5rem;font:600 .95rem var(--sans);text-decoration:none;color:#fff;border:1px solid rgba(255,255,255,.45);
  border-radius:10px;padding:.7rem 1.15rem;background:rgba(255,255,255,.08);transition:background .15s,transform .15s}
.btn:hover{background:rgba(255,255,255,.2);transform:translateY(-1px)}
.btn.primary{background:var(--gold);border-color:var(--gold);color:#1a1400}
.btn.primary:hover{background:#f0c818}
.note{font:.85rem var(--sans);color:#bcd0ff;margin:.9rem 0 0}
section.card{background:var(--page);border:1px solid var(--rule);border-radius:14px;box-shadow:var(--shadow);padding:1.6rem 1.7rem;margin-top:1.6rem}
h2{font:700 1.25rem var(--sans);margin:0 0 .9rem;color:var(--navy)}
.feat{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:1rem;margin-top:1rem}
.feat div{background:var(--tint);border-radius:10px;padding:.9rem 1rem;font-size:.95rem}
.feat b{display:block;font:650 .95rem var(--sans);color:var(--indigo);margin-bottom:.2rem}
.demo{margin:1.1rem 0 0;border:1px solid var(--rule);border-radius:9px;overflow:hidden;font:.8rem/1.6 var(--mono);background:#f7f9fc;color:#1d2433}
:root.dark .demo{background:#0d1526;color:#dde5f5}
.demo .h{display:flex;justify-content:space-between;padding:.3rem .8rem;background:#eaeff7;font:600 .7rem var(--sans);letter-spacing:.08em;text-transform:uppercase;color:#3333b2}
:root.dark .demo .h{background:#16213a;color:#9aa2ff}
.demo pre{margin:0;padding:.8rem 1rem;overflow-x:auto}
.k{color:#8a2bb6}.s{color:#1f7a3a}.c{color:#76809a;font-style:italic}.f{color:#2150c8}
:root.dark .k{color:#c8a2f0}:root.dark .s{color:#9fdc9c}:root.dark .c{color:#7d89a6}:root.dark .f{color:#8db4ff}
.authors{display:flex;gap:1.2rem;align-items:center}
.authors img{width:96px;height:96px;border-radius:50%;object-fit:cover;border:3px solid var(--rule)}
.authors p{margin:.2rem 0 0;color:var(--muted);font-size:.95rem}
.toc-controls{display:flex;gap:.6rem;margin:0 0 1rem}
.cbtn{font:600 .78rem var(--sans);color:var(--indigo);background:transparent;border:1px solid var(--rule);border-radius:7px;padding:.35rem .7rem;cursor:pointer}
.cbtn:hover{border-color:var(--indigo)}
.part{margin:1.4rem 0 0}
.part h3{font:700 1.08rem var(--sans);margin:0 0 .5rem;display:flex;flex-wrap:wrap;align-items:baseline;gap:.6rem;
  border-bottom:2px solid var(--navy);padding-bottom:.35rem}
.part h3 a{color:var(--ink);text-decoration:none}.part h3 a:hover{color:var(--link)}
.part-num{font:700 .7rem var(--sans);letter-spacing:.14em;text-transform:uppercase;color:#fff;background:var(--navy);border-radius:5px;padding:.2rem .5rem}
:root.dark .part-num{color:#0b1120}
.chapters{list-style:none;margin:0;padding:0}
.ch{border-bottom:1px solid var(--rule)}
.ch summary{display:flex;gap:.8rem;align-items:baseline;padding:.55rem .2rem;cursor:pointer;list-style:none}
.ch summary::-webkit-details-marker{display:none}
.ch summary::after{content:"+";margin-left:auto;font:600 1rem var(--sans);color:var(--muted)}
.ch details[open]>summary::after{content:"−"}
.ch.plain{padding:.55rem .2rem}
.num{flex:none;min-width:1.7rem;font:700 1.05rem var(--sans);color:var(--indigo)}
.ch-title{color:var(--ink);text-decoration:none;font:600 1rem var(--sans)}
.ch-title:hover{color:var(--link);text-decoration:underline}
.ch code,.subs code,.part code{font:.88em var(--mono)}
.subs{list-style:none;margin:0 0 .7rem;padding:0 0 0 2.5rem;columns:2 18rem;column-gap:2rem}
.subs li{break-inside:avoid;padding:.12rem 0;font:.93rem/1.4 var(--sans)}
.subs a{color:var(--muted);text-decoration:none}.subs a:hover{color:var(--link);text-decoration:underline}
.snum{color:var(--indigo);font-weight:600;margin-right:.45rem}
footer{margin-top:2.2rem;text-align:center;font:.85rem var(--sans);color:var(--muted)}
#top{position:fixed;right:1rem;bottom:1rem;width:2.6rem;height:2.6rem;border-radius:50%;border:0;background:var(--navy);color:#fff;font-size:1.2rem;cursor:pointer;opacity:0;pointer-events:none;transition:opacity .2s}
:root.dark #top{color:#0b1120}
#top.on{opacity:.9;pointer-events:auto}
@media(max-width:560px){.authors{flex-direction:column;text-align:center}.subs{padding-left:.6rem}}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <div class="tools"><button class="tbtn" id="theme" type="button" aria-label="Перемкнути тему">Темна тема</button></div>
  <p class="eyebrow">Навчальний посібник</p>
  <h1>Квантово-механічні методи обчислення</h1>
  <p class="sub">Використання <code>PySCF</code></p>
  <div class="actions">
    <a class="btn primary" href="../online/index.html">Читати онлайн</a>
    <a class="btn" href="@PDF@">Завантажити PDF</a>
    <a class="btn" href="@REPO@" target="_blank" rel="noopener">Код прикладів на GitHub</a>
  </div>
  <p class="note">Онлайн-версія містить усі формули й рисунки книги; код у ній можна копіювати одним натисканням.</p>
</header>

<section class="card">
  <h2>Про посібник</h2>
  <p>Практичне введення до курсу «Квантова хімія та квантово-механічні методи обчислень»: як за допомогою відкритого пакета <a href="https://pyscf.org/" target="_blank" rel="noopener">PySCF</a> (<em>Python-based Simulations of Chemistry Framework</em>) розраховувати електронну структуру, енергії, властивості й спектри атомів та молекул — і розуміти, що саме при цьому обчислюється.</p>
  <p>Посібник створено для студентів магістратури Навчально-наукового фізико-технічного інституту КПІ ім. Ігоря Сікорського (ОП «E6 Прикладна фізика та наноматеріали»). Теорія йде поруч із робочими прикладами: кожен приклад — окремий файл у <a href="@REPO@" target="_blank" rel="noopener">репозиторії</a>.</p>
  <div class="feat">
    <div><b>Теорія й код разом</b>Від молекулярного об’єкта й базисних наборів до Хартрі–Фока, пост-HF та DFT, молекулярних властивостей, коливань і збуджених станів.</div>
    <div><b>Код можна копіювати</b>Кожен блок коду має кнопку «Копіювати»; довгі приклади згортаються, а на файли-приклади є посилання на GitHub.</div>
    <div><b>Світла й темна тема</b>Зручне читання на екрані та на телефоні; пошук по тексту розділів.</div>
  </div>
  <div class="demo" aria-hidden="true"><div class="h"><span>Python</span><span>Копіювати</span></div><pre><span class="k">from</span> pyscf <span class="k">import</span> gto, scf
mol = gto.<span class="f">M</span>(atom=<span class="s">'H 0 0 0; H 0 0 0.74'</span>, basis=<span class="s">'cc-pvdz'</span>)
mf = scf.<span class="f">RHF</span>(mol).<span class="f">run</span>()   <span class="c"># енергія Хартрі–Фока</span></pre></div>
</section>

<section class="card">
  <h2>Автор</h2>
  <div class="authors">
    <img src="https://apd.ipt.kpi.ua/wp-content/uploads/2025/01/MyPhoto2.jpg" alt="Сергій Пономаренко" loading="lazy">
    <div>
      <strong><a href="https://apd.ipt.kpi.ua/kafedra/stuff/ponomarenko/" target="_blank" rel="noopener">Пономаренко Сергій Миколайович</a></strong>
      <p>Кандидат фізико-математичних наук, доцент кафедри прикладної фізики КПІ ім. Ігоря Сікорського.</p>
    </div>
  </div>
</section>

<section class="card" id="toc">
  <h2>Зміст</h2>
  <div class="toc-controls">
    <button type="button" class="cbtn" id="open-all">Розгорнути все</button>
    <button type="button" class="cbtn" id="close-all">Згорнути все</button>
  </div>
@TOC@
</section>

<footer>Квантово-механічні методи обчислення (PySCF) &nbsp;|&nbsp; Кафедра прикладної фізики, КПІ ім. Ігоря Сікорського</footer>
</div>
<button id="top" type="button" aria-label="Нагору">↑</button>
<script>
(function(){
  var root=document.documentElement,btn=document.getElementById('theme');
  function label(){btn.textContent=root.classList.contains('dark')?'Світла тема':'Темна тема';}
  label();
  btn.addEventListener('click',function(){var d=root.classList.toggle('dark');try{localStorage.setItem('pyscfbook-theme',d?'dark':'light');}catch(e){}label();});
  function all(o){document.querySelectorAll('#toc details').forEach(function(d){d.open=o;});}
  document.getElementById('open-all').addEventListener('click',function(){all(true);});
  document.getElementById('close-all').addEventListener('click',function(){all(false);});
  var top=document.getElementById('top');
  addEventListener('scroll',function(){top.classList.toggle('on',scrollY>600);},{passive:true});
  top.addEventListener('click',function(){scrollTo({top:0,behavior:'smooth'});});
})();
</script>
</body>
</html>
"""


def main() -> int:
    front, parts, back = parse()
    if not parts:
        print("У змісті не знайдено частин/розділів.")
        return 1
    html = (TEMPLATE.replace("@TOC@", render(front, parts, back))
            .replace("@PDF@", PDF).replace("@SITE@", SITE).replace("@REPO@", REPO))
    OUT.write_text(html, encoding="utf-8", newline="\n")
    nch = sum(len(p["chapters"]) for p in parts)
    nsub = sum(len(c["subs"]) for p in parts for c in p["chapters"])
    print("index.html: %d частин, %d розділів, %d підрозділів -> %s" % (len(parts), nch, nsub, OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
