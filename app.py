# rag works workbench. chatbot on the surface, rag lab underneath
# run from the rag-works folder:  streamlit run app.py
import base64
import html
import os
import re
import threading
import time
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from rag import evals, framework, index, kb as kbs, testbed
from rag.answer import ask
from rag.config import (CHECKS, CHUNKERS, DEFAULT_EMBEDDER, DEFAULT_PROVIDER, DESCRIBE_STYLES, EMBEDDERS, PROVIDERS, RERANKERS, RETRIEVERS, ROOT,
                        has_key, ix_key, parse_ix)
from rag.index import GRIDS
from rag.llm import set_key

Image.MAX_IMAGE_PIXELS = None
st.set_page_config(page_title="RAG Works", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(ROOT / 'assets' / 'citrine.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
esc = html.escape
SERIES = ["#B57A1A", "#0F8A82", "#7A2E5A", "#5F574B"]   # amber, teal, plum, graphite
BOX_COLOR = {"text": "#0F8A82", "table": "#B57A1A", "visual": "#7A2E5A"}

# small stroke icons, 24 grid, drawn with currentColor
_ICON = {
    "file": '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "check": '<path d="M5 4h14v16H5z"/><path d="M9 12l2 2 4-4"/>',
    "shield": '<path d="M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    "link": '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
    "bolt": '<path d="M13 3L5 13h6l-1 8 8-10h-6z"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
}


def icon(name, size=16):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" '
            f'stroke-linecap="round" stroke-linejoin="round">{_ICON[name]}</svg>')


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def header(name, title, right=""):
    md(f'<div class="rw-ph">{icon(name)}<span>{title}</span><span class="sp"></span>{right}</div>')


@st.cache_data(show_spinner=False, max_entries=500)
def data_uri(path, mtime, px=160):
    img = Image.open(path).convert("RGB")
    img.thumbnail((px, px))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def uri(rel_or_path, px=160):
    p = rel_or_path if hasattr(rel_or_path, "stat") else ROOT / rel_or_path
    return data_uri(str(p), p.stat().st_mtime, px) if p and p.exists() else ""


def cite_marks(text):
    return re.sub(r"\[(\d+)\]", r'<span class="rw-n">[\1]</span>', esc(text))


def pct(v):
    return "–" if v is None else f"{v:.2f}"


def badge(diag):
    cls = "pass" if diag.startswith("pass") else "partial" if diag.startswith("partial") else "fail" if diag.startswith(("fail", "error")) else ""
    return f'<span class="rw-badge {cls}">{esc(diag)}</span>'


def size_txt(p):
    b = p.stat().st_size
    return f"{b / 1048576:.1f} MB" if b > 1048576 else f"{b / 1024:.0f} KB"


def font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


@st.cache_data(show_spinner=False, max_entries=400)
def _preview(path, mtime, px):
    img = Image.open(path).convert("RGB")
    img.thumbnail((px, px))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def pv(path, px=640):
    """small jpeg of an image, cached. st.image on a 3000 px png re-reads and re-sends megabytes every click"""
    p = Path(path)
    return _preview(str(p), p.stat().st_mtime_ns, px)


def draw_boxes(path, boxes, width=1100):
    p = Path(path)
    key = tuple((tuple(round(x, 4) for x in b) if b else None, lab, col) for b, lab, col in boxes)
    return _draw_boxes(str(p), p.stat().st_mtime_ns, key, width)


@st.cache_data(show_spinner=False, max_entries=100)
def _draw_boxes(path, mtime, boxes, width):
    """page image with labeled boxes. boxes = [(box, label, color)]. cached: same page + boxes never redraws"""
    img = Image.open(path).convert("RGB")
    img.thumbnail((width, width))
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    f = font(max(12, w // 55))
    for b, lab, col in boxes:
        if not b:
            continue
        r = [b[0] * w, b[1] * h, b[2] * w - 1, b[3] * h - 1]
        d.rectangle(r, outline=col, width=max(2, w // 400))
        if lab:
            tw = d.textlength(lab, font=f)
            d.rectangle([r[0], r[1], r[0] + tw + 8, r[1] + f.size + 6], fill=col)
            d.text((r[0] + 4, r[1] + 2), lab, fill="#FFFCF4", font=f)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


kb_list = kbs.list_kbs()
if not kb_list:
    st.error("no knowledge bases in corpus/. make corpus/<name>/docs/ and drop files in")
    st.stop()

# top bar
t1, t2, t3, t4, t5, t9, t6, t7, t8 = st.columns([2.1, 1.1, 1.15, 0.85, 0.85, 0.85, 1.05, 0.75, 0.9], vertical_alignment="bottom")
t1.markdown('<div class="rw-brand"><div class="rw-mark">RAG<span>_</span>Works</div><div class="rw-sub">multimodal rag chatbot + evaluation workbench</div></div>', unsafe_allow_html=True)
kb = t2.selectbox("knowledge base", kb_list, format_func=lambda k: kbs.info(k).get("name", k))
# switches live in session state so the framework view's "try it" can flip them. defaults = the fp10 setup
WIDGET_DEFAULTS = {"w_chunker": "layout", "w_retriever": "hybrid", "w_reranker": "cross", "w_check": "auto", "w_embedder": DEFAULT_EMBEDDER,
                   "w_style": "structured", "w_ctx": True, "w_labels": True, "w_spatial": True}
for wk, wv in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(wk, wv)
chunker = t3.selectbox("chunking", list(CHUNKERS), key="w_chunker", format_func=lambda s: CHUNKERS[s]["label"])
retriever = t4.selectbox("retrieval", list(RETRIEVERS), key="w_retriever")
reranker = t5.selectbox("rerank", list(RERANKERS), key="w_reranker")
check_mode = t9.selectbox("self-check", list(CHECKS), key="w_check", help="\n\n".join(f"**{k}**: {v}" for k, v in CHECKS.items()))
plist = list(PROVIDERS)
# bring your own key: anyone who downloads this can paste a free key here instead of editing .env.
# set RAG_PUBLIC=1 when hosting for others, so visitors can't put keys into a shared server process
if os.getenv("RAG_PUBLIC") != "1" and DEFAULT_PROVIDER != "ollama" and not any(has_key(p) for p in plist if PROVIDERS[p]["key"]):
    with st.expander("no api key yet · paste a free one here (kept in memory only, not saved)", expanded=True):
        c1, c2 = st.columns(2)
        gk = c1.text_input("GEMINI_API_KEY · free at aistudio.google.com", type="password")
        qk = c2.text_input("GROQ_API_KEY · free at console.groq.com", type="password")
        if st.button("use these keys", type="primary"):
            for p, kv in (("gemini", gk), ("groq", qk)):
                if kv.strip():
                    set_key(p, kv)
            st.rerun()
        st.caption("or run fully local with ollama, no key: see the readme. or put keys in .env so you never see this box")
provider = t6.selectbox("model", plist, index=plist.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in plist else 0,
                        format_func=lambda p: f"{p} · {PROVIDERS[p]['vision']}" if has_key(p) else f"{p} (no key)")
P = PROVIDERS[provider]
with t7.popover("settings", width="stretch"):
    a, b = st.columns(2, gap="medium")
    with a:
        md('<div class="rw-eyebrow">index (rebuild after changing)</div>')
        st.caption(CHUNKERS[chunker]["about"])
        embedder = st.selectbox("embedding model", list(EMBEDDERS), key="w_embedder", format_func=lambda e: EMBEDDERS[e]["label"])
        style = st.selectbox("image descriptions", list(DESCRIBE_STYLES), key="w_style", help="\n".join(f"{k}: {v}" for k, v in DESCRIBE_STYLES.items()))
        contextual = st.toggle("contextual chunks", key="w_ctx", help="one line of page context in front of every chunk before embedding")
    with b:
        md('<div class="rw-eyebrow">retrieval + answer</div>')
        pool = st.select_slider("candidate pool (before rerank)", [10, 20, 30, 50], value=20)
        max_images = st.slider("max image crops sent", 0, P["max_images"], min(2, P["max_images"]))
        max_text = st.slider("max text chunks sent", 0, 6, 3)
        with_page = st.toggle("also send the whole page (orientation)", value=False, help="full page next to the crops. uses one image slot")
        use_labels = st.toggle("label positions from the pdf", key="w_labels", help="exact x,y of every text label on the page, legend skipped (fp9)")
        use_spatial = st.toggle("spatial facts", key="w_spatial", help="code works out directions between the places the question names, from the label positions (fp10)")
        md('<div class="rw-eyebrow" style="margin-top:10px">evals</div>')
        use_judge = st.toggle("auto-grade with the llm judge", value=True)
        split = st.radio("gold split", ["dev", "test", "all"], horizontal=True, help="tune on dev, report test once at the end")
        note = st.text_input("run note", placeholder="what changed in this run?")
    st.caption(f"vision `{P['vision']}` · text `{P['text']}` · retrieval: {RETRIEVERS[retriever]} · rerank: {RERANKERS[reranker]}")
key = ix_key(chunker, embedder, style, contextual)
k = st.session_state.get("topk", 8)
qcfg = {"retriever": retriever, "reranker": reranker, "k": k, "pool": pool, "max_images": max_images, "max_text": max_text, "with_page": with_page,
        "labels": use_labels, "check": check_mode, "spatial": use_spatial}
stat = index.status(kb, key)
docs = kbs.docs(kb)


@st.cache_resource(show_spinner=False)
def warm(kb, embedder, reranker):
    """load the local models + analyze pdf pages in the background, so the first question doesn't wait on us"""
    def go():
        try:
            from rag.embed import embed
            embed(embedder, ["warm up"], query=True)
            if reranker == "cross":
                from rag.retrieve import cross_model
                cross_model()
            from rag import layout as lay
            for d in kbs.docs(kb):
                if kbs.kind(d) == "pdf" and kbs.page_count(d) <= 80:
                    for n in range(1, kbs.page_count(d) + 1):
                        lay.analyze(d, n)
        except Exception as e:
            print("warm-up skipped:", e)
    threading.Thread(target=go, daemon=True).start()
    return True


if stat and embedder != "gemini":
    warm(kb, embedder, reranker)
gold_all = evals.gold(kb)
gold = evals.gold(kb, split)
run_clicked = t8.button("run eval", type="primary" if stat else "secondary", disabled=not (stat and gold), width="stretch",
                        help=None if stat else "build the index first")
md('<hr class="rw-rule">')
if run_clicked:
    with st.status(f"running {len(gold)} {split} questions · {key} · {retriever}+{reranker} · {provider}", expanded=True) as box:
        bar = st.progress(0.0)
        try:
            out = evals.run(kb, key, provider, qcfg, use_judge, split=split, note=note,
                            progress=lambda i, n, q: bar.progress((i - 1) / n, text=f"{i}/{n} · {q}"))
            st.session_state["eval_pick"] = out["file"]
            box.update(label=f"eval saved · {out['file']}", state="complete", expanded=False)
        except Exception as e:
            box.update(label="eval failed", state="error")
            st.error(f"{type(e).__name__}: {e}")

left, mid, right = st.columns([1.05, 2.2, 1.55], gap="medium")

# documents panel
with left, st.container(key="pane_docs", height=640):
    header("file", "documents", f"<small>{len(docs)}</small>")
    with st.popover("+ upload documents", width="stretch"):
        ups = st.file_uploader("pdf, image, txt or md", type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"], accept_multiple_files=True)
        if ups and st.button("save to knowledge base", type="primary"):
            for u in ups:
                kbs.save_upload(kb, u.name, u.getvalue())
            st.rerun()
        name = st.text_input("or make a new knowledge base", placeholder="testbed")
        if st.button("create", type="secondary") and name:
            try:
                st.success(f"created corpus/{kbs.create_kb(name)}/docs. pick it up top")
            except ValueError as e:
                st.error(str(e))
    rows = []
    for d in docs:
        kd = kbs.kind(d)
        th = kbs.thumb(kb, d)
        style_attr = f' style="background-image:url({uri(th, 120)})"' if th else ""
        pages_n = kbs.page_count(d)
        rows.append(f'<div class="rw-doc"><div class="rw-thumb"{style_attr}>{"" if th else "TXT"}</div><div><div class="nm">{esc(d.name)}</div>'
                    f'<div class="mt"><span class="rw-ft {kd}">{kd.upper()}</span>{size_txt(d)}{f" · {pages_n} pages" if pages_n > 1 else ""}</div></div></div>')
    md("".join(rows) or '<div class="rw-empty">no documents yet</div>')
    if stat:
        warn = stat["stale"]
        md(f'<div class="rw-idx{" warn" if warn else ""}">index <b>{key}</b><br>{stat["chunks"]} chunks ({stat.get("visual", "?")} visual) · {stat["built"][:16]}'
           + (" · docs changed, rebuild" if warn else "") + "</div>")
    else:
        md(f'<div class="rw-idx warn">no <b>{key}</b> index yet. build it to chat</div>')
    if st.button("build index" if not stat else "rebuild index", type="primary" if not stat else "secondary", width="stretch"):
        with st.status(f"building {key}", expanded=True) as box:
            try:
                index.build(kb, chunker, provider, embedder, style, contextual, log=box.write)
                box.update(label="index built", state="complete")
            except Exception as e:
                box.update(label="build failed", state="error")
                st.error(f"{type(e).__name__}: {e}")
        st.rerun()

def check_line(chk):
    if not chk or chk.get("mode", "none") == "none":
        return ""
    if chk.get("abstained"):
        bits = ["said not found instead of guessing"] + (["retried with a better search"] if chk.get("retried") else [])
        return f'<div class="rw-chk"><span class="rw-conf mid">no answer</span>{esc(" · ".join(bits))}</div>'
    cf = chk.get("confidence")
    lvl = "" if cf is None else "hi" if cf >= .75 else "mid" if cf >= .5 else "lo"
    bits = [f"check {chk['mode']}"]
    if chk.get("grade"):
        bits.append(f"{len(chk['grade']['relevant'])} evidence kept")
    if chk.get("retried"):
        bits.append("retried search")
    if chk.get("claims") is not None:
        bits.append(f"{sum(c['supported'] for c in chk['claims'])}/{len(chk['claims'])} claims ok")
    if chk.get("corrected"):
        bits.append("corrected")
    return f'<div class="rw-chk"><span class="rw-conf {lvl}">confidence {"–" if cf is None else f"{cf:.2f}"}</span>{esc(" · ".join(bits))}</div>'


def time_line(r):
    t = r.get("timing")
    if not t:
        return f'<div class="rw-meta">{esc(r["model"])} · {r["seconds"]}s · {r.get("at", "")}</div>'
    return (f'<div class="rw-meta">{esc(r["model"])} · {r.get("llm_calls", "?")} model calls · our code {t["program"] * 1000:.0f} ms · '
            f'model api {t["model"]:.1f}s · {r.get("at", "")}</div>')


def time_bars(t):
    """one bar per stage, our code in amber, the model api in teal, so it's obvious where the time went"""
    total = max(t["total"], 1e-6)
    segs = "".join(f'<i class="{"m" if s["model"] else "p"}" style="width:{max(.5, 100 * s["ms"] / 1000 / total):.2f}%" title="{esc(s["name"])} {s["ms"]:.0f} ms"></i>'
                   for s in t["stages"] if s["ms"] > 0)
    prog = " · ".join(f'{esc(s["name"])} {s["ms"]:.0f}ms' for s in t["stages"] if not s["model"] and s["ms"] >= 1)
    mdl = " · ".join(f'{esc(s["name"])} {s["ms"] / 1000:.1f}s' for s in t["stages"] if s["model"])
    return (f'<div class="rw-time-bar">{segs}</div><div class="rw-hm"><span class="tp">our code {t["program"] * 1000:.0f} ms</span>'
            + (f": {prog}" if prog else "") + f'</div><div class="rw-hm"><span class="tm">model api {t["model"]:.2f}s</span>' + (f": {mdl}" if mdl else "") + "</div>")


# chat panel
hist_key = f"hist_{kb}"
st.session_state.setdefault(hist_key, [])
hist = st.session_state[hist_key]
with mid, st.container(key="pane_chat", height=640):
    header("chat", "chat", f"<small>{esc(CHUNKERS[chunker]['label'])} · {retriever}{'+' + reranker if reranker != 'none' else ''} · {esc(provider)}</small>")
    box = st.container(key="scroll_chat", height=530, border=False)
    q = st.chat_input("ask a question about your documents")
    q = q or st.session_state.pop("pending_q", None)
    if q:
        if not stat:
            hist.append({"question": q, "error": "build the index first (documents panel)", "at": time.strftime("%I:%M %p")})
        else:
            with box, st.spinner("retrieving + generating"):
                try:
                    r = ask(kb, key, q, provider, qcfg)
                except Exception as e:
                    r = {"question": q, "error": f"{type(e).__name__}: {e}"}
            r["at"] = time.strftime("%I:%M %p").lstrip("0")
            hist.append(r)
    with box:
        if not hist:
            md('<div class="rw-empty">no questions yet. try one from the gold set:</div>')
            for g in gold_all[:6]:
                if st.button(g["question"], key=f"sample_{g['id']}", type="tertiary"):
                    st.session_state["pending_q"] = g["question"]
                    st.rerun()
        for i, r in enumerate(hist):
            md(f'<div class="rw-user"><div>{esc(r["question"])}</div></div><div class="rw-time">{r.get("at", "")}</div>')
            with st.container(key=f"amsg_{i}"):
                if r.get("error"):
                    md(f'<div class="rw-err">{esc(r["error"])}</div>')
                    continue
                chk = r.get("check") or {}
                cites = "".join(f'<span class="rw-cite">[{c["n"]}] {esc(c["label"])}</span>' for c in r["citations"])
                md(f'<div class="rw-msg{" abstain" if chk.get("abstained") else ""}"><div class="rw-answer">{cite_marks(r["answer"])}</div>'
                   + (f'<div class="rw-why">{cite_marks(r["why"])}</div>' if r["why"] else "")
                   + (f'<div class="rw-cites">{cites}</div>' if cites else "") + check_line(chk) + "</div>")
                claims = chk.get("claims") or []
                if claims or chk.get("corrected"):
                    with st.expander(f"self-check · {sum(c['supported'] for c in claims)}/{len(claims)} claims backed by evidence"
                                     + (" · answer was corrected" if chk.get("corrected") else "")):
                        md("".join(f'<div class="rw-claim {"ok" if c["supported"] else "bad"}"><b>{"✓" if c["supported"] else "✗"}</b> {esc(c["claim"])}'
                                   f' <span>{" ".join(f"[{n}]" for n in c["evidence"])} {esc(c["note"])}</span></div>' for c in claims))
                        if chk.get("original"):
                            md(f'<div class="rw-eyebrow" style="margin-top:8px">answer before the fix</div><div class="rw-why">{cite_marks(chk["original"])}</div>')
                cited = {c["n"] for c in r["citations"]}
                vis = [h for h in r["sent"] if h["kind"] == "visual"]
                show = [h for h in vis if h["n"] in cited] or ([] if chk.get("abstained") else vis)
                if show:
                    cols = st.columns(min(2, len(show)))
                    for c, h in zip(cols, show[:2]):
                        c.image(pv(ROOT / h["crop"], 560), caption=f"[{h['n']}] {h['doc']} p{h['page']} {h.get('label', '')}"
                                + ("" if h["n"] in cited else " · sent, not cited"), width=280)
                md(time_line(r))
        if hist and st.button("clear chat", type="tertiary"):
            st.session_state[hist_key] = []
            st.rerun()


def hit_cards(hits):
    out = []
    for h in hits:
        sc = h.get("score")
        lvl = "" if sc is None else "hi" if sc >= .6 else "mid" if sc >= .4 else ""
        vis = h["kind"] == "visual"
        where = h.get("label") or ("table" if h.get("sub") == "table" else "text")
        th = f'<div class="rw-hthumb" style="background-image:url({uri(h["crop"])})"></div>' if vis and h.get("crop") else \
             '<div class="rw-hthumb txt"><i></i><i style="width:80%"></i><i></i><i style="width:60%"></i><i></i></div>'
        sent = f' · <span class="s">sent as [{h.get("n")}]</span>' if h.get("sent") else ""
        ranks = []
        if h.get("vec_rank") or h.get("bm25_rank"):
            ranks.append(f'vec #{h.get("vec_rank") or "–"} · bm25 #{h.get("bm25_rank") or "–"}')
        mv = h.get("moved") or 0
        if mv:
            ranks.append(f'<span class="mv {"up" if mv > 0 else "down"}">rerank {"↑" if mv > 0 else "↓"}{abs(mv)}</span>')
        rank_line = f'<div class="rw-hm">{" · ".join(ranks)}</div>' if ranks else ""
        snippet = f'<div class="rw-hs">{esc(h["text"][:220])}</div>' if h.get("text") else ""
        out.append(f'<div class="rw-hit{" sent" if h.get("sent") else ""}"><div class="rw-rank">{h["rank"]}</div>{th}<div class="rw-hb">'
                   f'<div class="rw-ht"><b>{esc(h["doc"])} · {esc(where)}</b><span class="rw-score {lvl}">{"–" if sc is None else f"{sc:.2f}"}</span></div>'
                   f'<div class="rw-hm">page {h["page"]} · {h["kind"]}{sent}</div>{rank_line}{snippet}</div></div>')
    return "".join(out)


last = next((r for r in reversed(hist) if not r.get("error")), None)
with right, st.container(key="pane_trace", height=640):
    a, b = st.columns([2.4, 1], vertical_alignment="center")
    with a:
        header("db", "retrieval trace", "<small>top k</small>")
    b.selectbox("top k", [3, 5, 8, 10, 15, 20], index=2, key="topk", label_visibility="collapsed")
    if not last:
        md('<div class="rw-empty">ask something to see what was retrieved, how each retriever ranked it, what the reranker moved, and what got sent to the model</div>')
    else:
        s = last["settings"]
        md(f'<div class="rw-flow"><span>{esc(s["retriever"])} top {s["pool"]}</span><em>→</em><span>{esc(s["reranker"])} rerank</span><em>→</em>'
           f'<span>top {s["k"]}</span><em>→</em><span class="on">{len(last["sent"])} sent</span><em>→</em><span>{len(last["citations"])} cited</span></div>')
        if last.get("timing"):
            md(time_bars(last["timing"]))
        with st.container(key="scroll_trace", height=440, border=False):
            md(hit_cards(last["hits"]))
            st.caption("score = what ordered it last (similarity, keyword match, fused rank or reranker). filled rank = sent to the model"
                       + (". whole page also sent for orientation" if last["page_image"] else ""))


def tile(ic, label, v, prev, color, fmt="{:.2f}", lower_better=False, sub=""):
    if v is None:
        val, delta, width = "–", "", 0
    else:
        val, width = fmt.format(v), (min(1, v) if not lower_better else max(0.05, 1 - min(v, 20) / 20))
        delta = ""
        if prev is not None:
            dv = v - prev
            good = dv < 0 if lower_better else dv > 0
            cls = "flat" if abs(dv) < 1e-9 else "up" if good else "down"
            delta = f'<span class="d {cls}">{"↑" if dv > 0 else "↓" if dv < 0 else "·"} {abs(dv):.2f}</span>'
    return (f'<div class="rw-tile"><div class="l">{icon(ic, 14)}{label}</div><div class="row"><span class="v">{val}</span>{delta}</div>'
            f'<div class="bar"><i style="width:{width * 100:.0f}%;background:{color}"></i></div><div class="s">{sub}</div></div>')


METRICS = [("retrieval", "retrieval_pass"), ("recall@5", "recall@5"), ("mrr", "mrr"), ("answer", "answer_score"), ("grounding", "grounded"), ("citation", "citation_ok")]


def chart(rows, series, height=220):
    return alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("metric:N", sort=[m for m, _ in METRICS], title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("run:N", sort=series),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format=".1f", tickCount=5)),
        color=alt.Color("run:N", sort=series, scale=alt.Scale(domain=series, range=SERIES[:len(series)]), legend=alt.Legend(orient="bottom", title=None, labelLimit=500, columns=1)),
        tooltip=["run", "metric", alt.Tooltip("value:Q", format=".2f")],
    ).properties(height=height).configure_axis(labelColor="#5F574B", labelFont="JetBrains Mono", labelFontSize=11,
                                               gridColor="rgba(26, 24, 20, 0.08)", domainColor="rgba(26, 24, 20, 0.12)", tickColor="rgba(26, 24, 20, 0.12)"
    ).configure_legend(labelFont="JetBrains Mono", labelColor="#5F574B", symbolType="square").configure_view(strokeWidth=0).configure(background="#FFFCF4")


def type_heat(rows):
    """question type x something heatmap. rows = [{type, col, value}]"""
    return alt.Chart(pd.DataFrame(rows)).mark_rect(cornerRadius=2).encode(
        x=alt.X("col:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=220)), y=alt.Y("type:N", title=None),
        color=alt.Color("value:Q", scale=alt.Scale(domain=[0, 1], range=["#FAF6EC", "#B57A1A"]), legend=None),
        tooltip=["type", "col", alt.Tooltip("value:Q", format=".2f"), "n"],
    ).properties(height=max(120, 26 * len({r["type"] for r in rows}))).configure_axis(labelColor="#5F574B", labelFont="JetBrains Mono", labelFontSize=11
    ).configure_view(strokeWidth=0).configure(background="#FFFCF4")


md("<div style='height:6px'></div>")
# only the picked view runs (st.tabs would run all five on every click)
view = st.segmented_control("view", ["framework", "evals", "retrieval", "chunks", "test bed", "compare"], default="framework", key="view",
                            label_visibility="collapsed") or "framework"

FW_CSS = """<style>
.fw-pipe{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 4px}
.fw-st{flex:1 1 120px;background:#FFFCF4;border:1px solid rgba(26,24,20,.12);border-radius:6px;padding:8px 10px;min-width:120px}
.fw-st .n{font-family:'JetBrains Mono',monospace;font-size:11px;color:#B57A1A}
.fw-st .t{font-weight:600;font-size:13px;color:#1A1814;margin:1px 0 3px}
.fw-st .v{font-family:'JetBrains Mono',monospace;font-size:11px;color:#0F8A82;line-height:1.4}
.fw-st .d{font-size:11px;color:#5F574B;line-height:1.35;margin-top:4px}
.fw-chip{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;padding:2px 8px;border-radius:10px;margin:2px 4px 2px 0;
 background:rgba(15,138,130,.10);color:#0F8A82;border:1px solid rgba(15,138,130,.25)}
.fw-chip.off{background:transparent;color:#5F574B;border-color:rgba(26,24,20,.15)}
.fw-card{background:#FFFCF4;border:1px solid rgba(26,24,20,.12);border-radius:6px;padding:12px 14px;margin-bottom:6px}
.fw-card.on{border-color:#0F8A82;box-shadow:inset 3px 0 0 #0F8A82}
.fw-card h4{margin:0 0 2px;font-size:15px;color:#1A1814}
.fw-card .meta{font-family:'JetBrains Mono',monospace;font-size:11px;color:#5F574B;margin-bottom:6px}
.fw-card .st{color:#B57A1A}.fw-card .st.desc{color:#7A2E5A}
.fw-card dl{display:grid;grid-template-columns:92px 1fr;gap:3px 10px;margin:0;font-size:12.5px;line-height:1.4}
.fw-card dt{color:#5F574B;font-family:'JetBrains Mono',monospace;font-size:11px;padding-top:1px}
.fw-card dd{margin:0;color:#1A1814}
.fw-m{font-family:'JetBrains Mono',monospace;font-size:11.5px;margin-top:7px;padding-top:6px;border-top:1px dashed rgba(26,24,20,.15);color:#1A1814}
.fw-m .up{color:#0F8A82}.fw-m .down{color:#7A2E5A}
</style>"""


def try_cfg(c):
    """flip the top bar + settings to a catalog type's switches"""
    for k, wk in (("chunker", "w_chunker"), ("embedder", "w_embedder"), ("style", "w_style"), ("contextual", "w_ctx"), ("retriever", "w_retriever"),
                  ("reranker", "w_reranker"), ("labels", "w_labels"), ("spatial", "w_spatial"), ("check", "w_check")):
        st.session_state[wk] = c[k]


def delta(a, b, key2, name):
    va, vb = a.get(key2), b.get(key2)
    if va is None or vb is None:
        return ""
    d = vb - va
    cls = "up" if d > 0.005 else "down" if d < -0.005 else ""
    return f'{name} {va:.2f} → {vb:.2f} <span class="{cls}">({d:+.2f})</span>'


if view == "framework":
    md(FW_CSS)
    cur = {"chunker": chunker, "embedder": embedder, "style": style, "contextual": contextual, "retriever": retriever, "reranker": reranker,
           "labels": use_labels, "spatial": use_spatial, "check": check_mode}
    vals = {"chunker": CHUNKERS[chunker]["label"], "style": style, "embedder": embedder, "contextual": "+ page context" if contextual else "no context",
            "retriever": retriever, "reranker": reranker, "labels": "label positions" if use_labels else "no labels",
            "spatial": "spatial facts" if use_spatial else "", "check": f"check {check_mode} · {provider}"}
    active = framework.classify(cur)
    md('<div class="rw-eyebrow">the framework · rag is seven stages, every rag type changes one or more of them · your setup up top is shown in teal</div>')
    md('<div class="fw-pipe">' + "".join(
        f'<div class="fw-st"><div class="n">{i:02d}</div><div class="t">{esc(name)}</div><div class="v">{"<br>".join(esc(vals[s]) for s in sw if vals.get(s))}</div>'
        f'<div class="d">{esc(what)}</div></div>' for i, (sid, name, what, sw) in enumerate(framework.STAGES, 1)) + "</div>")
    md('<div class="rw-eyebrow" style="margin-top:8px">this setup is</div>'
       + "".join(f'<span class="fw-chip">{esc(t["name"])}</span>' for t in framework.CATALOG if t["id"] in active)
       + '<div class="rw-meta">change any switch up top (or in settings) and this updates. that is how you build one</div>')
    runs_all = [evals.load(kb, r) for r in evals.list_runs(kb)]
    steps = framework.story_runs(runs_all, provider)
    # catalog
    md('<div class="rw-eyebrow" style="margin-top:16px">catalog · every type described with the same fields'
       + (f' · measured numbers from story.py runs on {esc(provider)}' if steps else f' · run python story.py --provider {esc(provider)} to fill in measured numbers') + "</div>")
    g_pick = st.segmented_control("group", ["all"] + framework.GROUPS, default="all", key="fw_group", label_visibility="collapsed") or "all"
    types = [t for t in framework.CATALOG if g_pick == "all" or t["group"] == g_pick]
    cols = st.columns(2, gap="medium")
    for i, t in enumerate(types):
        with cols[i % 2]:
            m = framework.measured(t, steps)
            mline = ""
            if m:
                mline = (f'<div class="fw-m">measured ({esc(t["measured"][0])} → {esc(t["measured"][1])}): '
                         + " · ".join(x for x in (delta(m[0], m[1], "retrieval_pass", "retrieval"), delta(m[0], m[1], "answer_score", "answer"),
                                                  delta(m[0], m[1], "grounded", "grounded")) if x) + "</div>")
            stat_cls = "st" if t["status"] == "implemented" else "st desc"
            md(f'<div class="fw-card{" on" if t["id"] in active else ""}"><h4>{esc(t["name"])}</h4>'
               f'<div class="meta">{esc(t["group"])} · <span class="{stat_cls}">{esc(t["status"])}</span> · {esc(t["fp"])} · stages: {esc(", ".join(t["stages"]))}</div>'
               f'<dl><dt>problem</dt><dd>{esc(t["problem"])}</dd><dt>how</dt><dd>{esc(t["how"])}</dd><dt>cost</dt><dd>{esc(t["cost"])}</dd>'
               f'<dt>fails when</dt><dd>{esc(t["fails"])}</dd><dt>use when</dt><dd>{esc(t["when"])}</dd></dl>{mline}</div>')
            if t["config"]:
                need = ix_key(t["config"]["chunker"], t["config"]["embedder"], t["config"]["style"], t["config"]["contextual"])
                st.button(f"try {t['name'].lower()}", key=f"fw_try_{t['id']}", type="tertiary", on_click=try_cfg, args=(t["config"],),
                          help=f"sets the switches up top. needs the {need} index" + ("" if index.status(kb, need) else " (not built yet, build it in the documents panel)"))
    # the story
    md('<div class="rw-eyebrow" style="margin-top:16px">using the framework: one stage at a time on the same '
       f'{len(gold_all)} questions · {esc(provider)}</div>')
    have = [(n, stage, what) for n, stage, what in framework.STORY if n in steps]
    if not have:
        md(f'<div class="rw-empty">no story runs for {esc(provider)} yet. python story.py --provider {esc(provider)}</div>')
    else:
        rows = [{"step": n, "metric": lab, "value": steps[n]["summary"].get(k2)} for n, _, _ in have
                for lab, k2 in (("retrieval pass", "retrieval_pass"), ("answer score", "answer_score"), ("grounded", "grounded"))
                if steps[n]["summary"].get(k2) is not None]
        order = [n for n, _, _ in have]
        line = alt.Chart(pd.DataFrame(rows)).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("step:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format=".1f", tickCount=5)),
            color=alt.Color("metric:N", scale=alt.Scale(range=SERIES[:3]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["step", "metric", alt.Tooltip("value:Q", format=".2f")]).properties(height=240).configure_axis(
            labelColor="#5F574B", labelFont="JetBrains Mono", labelFontSize=11, gridColor="rgba(26, 24, 20, 0.08)", domainColor="rgba(26, 24, 20, 0.12)"
        ).configure_legend(labelFont="JetBrains Mono", labelColor="#5F574B").configure_view(strokeWidth=0).configure(background="#FFFCF4")
        a, b = st.columns([1.4, 1], gap="large")
        with a:
            st.altair_chart(line, width="stretch")
        with b:
            trs = "".join(f'<tr><td>{esc(n)}</td><td>{esc(stage)}</td><td>{pct(steps[n]["summary"].get("answer_score"))}</td>'
                          f'<td>{steps[n]["summary"].get("hallucinated", 0)}</td></tr>' for n, stage, _ in have)
            md(f'<table class="rw-tbl"><tr><th>step</th><th>stage changed</th><th>answer</th><th>halluc.</th></tr>{trs}</table>')
            md('<div class="rw-note">each step changes one stage. retrieval going up while answers go down (fp7) is what sent the next steps '
               'after the evidence instead of the search. numbers come from the llm judge, spot-check before quoting</div>')

if view == "evals":
    run_list = evals.list_runs(kb)
    if not run_list:
        md(f'<div class="rw-empty">no eval runs yet. {len(gold_all)} gold questions in corpus/{kb}/gold.json. hit run eval up top, or run a whole ladder: python cli.py ladder {kb} fp7</div>')
    else:
        runs = [evals.load(kb, r) for r in run_list]
        names = {r["file"]: f"{r['at']} · {evals.label(r)}" + (f" · {r['note']}" if r.get("note") else "") for r in runs}
        default = st.session_state.get("eval_pick")
        c1, c2 = st.columns([1.25, 1], gap="large")
        with c1:
            pick = st.selectbox("run", run_list, index=run_list.index(default) if default in run_list else 0, format_func=names.get)
            R = next(r for r in runs if r["file"] == pick)
            older = runs[run_list.index(pick) + 1:]
            PS = older[0]["summary"] if older else {}
            S = R["summary"]
            md('<div class="rw-eyebrow">this run' + (" vs the run before it" if older else "") + f' · split {R.get("split", "all")} · {S["questions"]} questions</div>')
            md('<div class="rw-tiles">'
               + tile("target", "retrieval pass", S["retrieval_pass"], PS.get("retrieval_pass"), SERIES[0], sub="all gold evidence sent")
               + tile("list", "recall@5", S.get("recall@5"), PS.get("recall@5"), SERIES[1], sub=f"mrr {pct(S.get('mrr'))}")
               + tile("check", "answer", S["answer_score"], PS.get("answer_score"), SERIES[2], sub=f"{S['correct']} ok · {S['partial']} part · {S['wrong']} wrong")
               + tile("shield", "grounding", S["grounded"], PS.get("grounded"), SERIES[0], sub=f"key facts {pct(S.get('key_facts'))}")
               + tile("link", "citation", S["citation_ok"], PS.get("citation_ok"), SERIES[1], sub="cites the expected doc")
               + tile("bolt", "latency (s)", S["avg_seconds"], PS.get("avg_seconds"), SERIES[2], "{:.1f}", True,
                      sub=f"our code {S.get('program_ms', '–')} ms · {S.get('llm_calls') or '–'} calls/q")
               + "</div>")
            notes = [f"{S['generation_failures']} generation failures (right evidence, wrong answer)", f"{S['retrieval_failures']} retrieval failures"]
            if S.get("false_abstain") or S.get("corrected"):
                notes.append(f"said 'not found' on {pct(S.get('false_abstain'))} of answerable · {S.get('corrected', 0)} answers corrected by the checker")
            if S.get("abstain_ok") is not None:
                notes.append(f"unanswerable handled {pct(S['abstain_ok'])} · {S.get('hallucinated', 0)} hallucinated")
            notes.append(f"judge agrees with you {pct(S['judge_agreement'])} on {S['checked']} checked" if S.get("checked") else "tick 'checked' on rows you graded by hand to measure the judge")
            md('<div class="rw-note">' + " · ".join(notes) + "</div>")
        with c2:
            md('<div class="rw-eyebrow">by question type · this run</div>')
            rows = [{"type": t, "col": m, "value": v[key2] if v[key2] is not None else 0, "n": v["n"]}
                    for t, v in S.get("by_type", {}).items() for m, key2 in [("retrieval", "retrieval_pass"), ("recall@5", "recall@5"), ("answer", "answer_score")]]
            if rows:
                st.altair_chart(type_heat(rows), width="stretch")
            md('<div class="rw-eyebrow" style="margin-top:8px">recent configs · same provider + split</div>')
            seen, recent = set(), []
            for r in runs:
                lab = evals.label(r)
                if r["provider"] == R["provider"] and r.get("split") == R.get("split") and lab not in seen:
                    seen.add(lab)
                    recent.append(r)
            recent = recent[:6]
            best = max(recent, key=lambda r: tuple(r["summary"].get(x) or 0 for x in ("answer_score", "retrieval_pass", "recall@5")))
            trs = "".join(f'<tr class="{"best" if r is best and len(recent) > 1 else ""}"><td>{esc(evals.label(r).rsplit(" · ", 1)[0])}</td>'
                          + "".join(f'<td>{pct(r["summary"].get(x))}</td>' for x in ("retrieval_pass", "recall@5", "answer_score"))
                          + f'<td>{r["summary"]["avg_seconds"]:.1f}</td></tr>' for r in recent)
            md(f'<table class="rw-tbl"><tr><th>config</th><th>retr</th><th>r@5</th><th>answer</th><th>sec</th></tr>{trs}</table>')

        md('<div class="rw-eyebrow" style="margin-top:14px">per question · grade, grounded and checked are editable</div>')
        df = pd.DataFrame([{"id": i["id"], "type": i.get("type", ""), "retrieval": "PASS" if i["retrieval_pass"] else "FAIL",
                            "r@5": i.get("recall@5"), "mrr": i.get("mrr"), "grade": i.get("grade") or None, "grounded": i.get("grounded") or None,
                            "judge": i.get("judge_grade") or "", "checked": bool(i.get("checked")), "diagnosis": i.get("diagnosis", ""),
                            "conf": i.get("confidence"), "calls": i.get("llm_calls"), "sec": i.get("seconds", 0), "question": i["question"]} for i in R["items"]])
        edited = st.data_editor(df, hide_index=True, width="stretch", disabled=[c for c in df.columns if c not in ("grade", "grounded", "checked")],
                                column_config={"grade": st.column_config.SelectboxColumn(options=["correct", "partial", "wrong"]),
                                               "grounded": st.column_config.SelectboxColumn(options=["yes", "no"]),
                                               "checked": st.column_config.CheckboxColumn(help="you checked this grade by hand")}, key=f"ed_{pick}")
        if st.button("save grades", type="secondary"):
            evals.regrade(kb, pick, {r2["id"]: {"grade": r2["grade"] or "", "grounded": r2["grounded"] or "", "checked": bool(r2["checked"])} for r2 in edited.to_dict("records")})
            st.rerun()
        item = st.selectbox("question detail", R["items"], format_func=lambda i: f"{i['id']} · {i['question']}")
        if item.get("error"):
            md(f'<div class="rw-err">{esc(item["error"])}</div>')
        else:
            a, b = st.columns(2, gap="large")
            with a:
                claims_html = "".join(f'<div class="rw-claim {"ok" if c["supported"] else "bad"}"><b>{"✓" if c["supported"] else "✗"}</b> {esc(c["claim"])}</div>'
                                      for c in (item.get("claims") or []))
                md(f'<div class="rw-card"><div class="rw-eyebrow">expected</div><div class="rw-answer">{esc(item["expected_answer"])}</div></div>'
                   f'<div class="rw-card"><div class="rw-eyebrow">generated · {badge(item["diagnosis"])}</div><div class="rw-answer">{cite_marks(item["answer"])}</div>'
                   + (f'<div class="rw-why">{cite_marks(item["why"])}</div>' if item.get("why") else "")
                   + (f'<div class="rw-why">before the fix: {cite_marks(item["original_answer"])}</div>' if item.get("original_answer") else "")
                   + claims_html
                   + (f'<div class="rw-meta">judge: {esc(item.get("reason", ""))}</div>' if item.get("reason") else "") + "</div>")
                g = next((x for x in gold_all if x["id"] == item["id"]), None)
                if g and g["evidence"]:
                    ev = g["evidence"][0]
                    pg_img = kbs.data_dir(kb) / "pages" / f"{ev['doc']}_p{ev.get('page', 1)}.png"
                    if pg_img.exists():
                        boxes = [(tuple(e["box"]), "gold", "#0F8A82") for e in g["evidence"] if e.get("box") and e["doc"] == ev["doc"]]
                        boxes += [(index.box_of(h), f"[{h['n']}]", "#B57A1A") for h in item["sent"] if h["doc"] == ev["doc"] and h.get("box")]
                        st.image(draw_boxes(pg_img, boxes, 900), caption=f"{ev['doc']} p{ev.get('page', 1)} · teal = gold evidence, amber = what was sent", width="stretch")
            with b:
                md(hit_cards(item["hits"]))

if view == "retrieval":
    if not last:
        md('<div class="rw-empty">ask a question first. this shows the exact prompt and raw output behind the last answer</div>')
    else:
        a, b = st.columns(2, gap="large")
        with a:
            md(f'<div class="rw-eyebrow">exact prompt sent to {esc(last["model"])}</div>')
            st.code(last["prompt"], language=None, wrap_lines=True, height=420)
        with b:
            md('<div class="rw-eyebrow">raw model output</div>')
            st.code(last["raw"], language=None, wrap_lines=True)
            imgs = ([last["page_image"]] if last["page_image"] else []) + [h["crop"] for h in last["sent"] if h["kind"] == "visual"]
            if imgs:
                md('<div class="rw-eyebrow">images attached, in order</div>')
                cols = st.columns(min(4, len(imgs)))
                for c, p in zip(cols, imgs):
                    c.image(pv(ROOT / p), width="stretch")

if view == "chunks":
    if not stat:
        md(f'<div class="rw-empty">build the {esc(key)} index to see its chunks</div>')
    else:
        chunks = index.all_chunks(kb, key)
        f1, f2, f3 = st.columns([2, 1, 1])
        doc_names = sorted({c["meta"]["doc"] for c in chunks}, key=lambda n: (not any(c["meta"]["doc"] == n and c["meta"].get("page_image") for c in chunks), n))
        doc_f = f1.selectbox("document", doc_names)
        pages_f = sorted({c["meta"]["page"] for c in chunks if c["meta"]["doc"] == doc_f})
        page_f = f2.selectbox("page", pages_f)
        kind_f = f3.selectbox("kind", ["all", "visual", "text"])
        rows = [c for c in chunks if c["meta"]["doc"] == doc_f and c["meta"]["page"] == page_f and (kind_f == "all" or c["meta"]["kind"] == kind_f)]
        rows.sort(key=lambda c: (c["meta"]["kind"] != "visual", c["meta"]["chunk"], c["id"]))
        a, b = st.columns([1.1, 1], gap="large")
        with a:
            img_path = ROOT / rows[0]["meta"]["page_image"] if rows and rows[0]["meta"].get("page_image") else None
            if img_path and img_path.exists():
                boxes = [(index.box_of(c["meta"]), c["meta"].get("label") or c["id"].split("|")[-1],
                          BOX_COLOR["table" if c["meta"].get("sub") == "table" else c["meta"]["kind"]]) for c in rows]
                st.image(draw_boxes(img_path, boxes), caption=f"{doc_f} p{page_f} · {key} · teal = text, amber = table, plum = visual", width="stretch")
            else:
                st.caption("text file, no page image. chunks are listed on the right")
        with b:
            st.caption(f"{len(rows)} chunks on this page · {len(chunks)} in the index · text is what got embedded (context line included)")
            st.dataframe(pd.DataFrame([{"id": c["id"].split("|", 2)[-1], "kind": c["meta"]["kind"], "label": c["meta"].get("label", ""),
                                        "chars": len(c["text"]), "heading": c["meta"].get("heading", "")} for c in rows]), hide_index=True, width="stretch", height=220)
            if rows:
                sel = st.selectbox("inspect chunk", rows, format_func=lambda c: c["id"])
                if sel["meta"]["crop"] and sel["meta"]["kind"] == "visual":
                    st.image(pv(ROOT / sel["meta"]["crop"], 640), caption=sel["meta"].get("where", ""), width=320)
                st.code(sel["text"], language=None, wrap_lines=True, height=260)

if view == "test bed":
    md(f'<div class="rw-eyebrow">gold-standard questions · corpus/{esc(kb)}/gold.json · {len(gold_all)} total · '
       + " · ".join(f"{s2} {sum(g['split'] == s2 for g in gold_all)}" for s2 in ("dev", "test")) + "</div>")
    raw_gold = [evals.normalize(g) for g in kbs.gold(kb)]
    if raw_gold:
        types = pd.Series([g["type"] for g in raw_gold]).value_counts()
        st.caption("by type: " + " · ".join(f"{t} {n}" for t, n in types.items()))
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        opts = ["+ new question"] + [g["id"] for g in raw_gold]
        pick_id = st.selectbox("edit", opts, format_func=lambda i: i if i.startswith("+") else f"{i} · {next(g['question'] for g in raw_gold if g['id'] == i)}")
        cur = next((g for g in raw_gold if g["id"] == pick_id), None) or {"id": testbed.next_id(raw_gold, "q"), "type": "text_fact", "split": "dev",
                                                                          "question": "", "expected_answer": "", "key_facts": [], "answerable": True, "evidence": []}
        ev_key = f"ev_{kb}_{cur['id']}"
        st.session_state.setdefault(ev_key, list(cur["evidence"]))
        evs = st.session_state[ev_key]
        with st.form(f"gold_{cur['id']}"):
            c1, c2, c3 = st.columns([1, 1.4, 0.8])
            gid = c1.text_input("id", cur["id"])
            gtype = c2.selectbox("type", testbed.TYPES, index=testbed.TYPES.index(cur["type"]) if cur["type"] in testbed.TYPES else 0)
            gsplit = c3.selectbox("split", ["dev", "test"], index=0 if cur["split"] == "dev" else 1)
            gq = st.text_input("question", cur["question"])
            gans = st.text_area("expected answer", cur["expected_answer"], height=80)
            gfacts = st.text_area("key facts (one per line, short)", "\n".join(cur["key_facts"]), height=80)
            gok = st.checkbox("answerable from the docs", value=cur["answerable"])
            if st.form_submit_button("save question", type="primary"):
                new = {"id": gid.strip(), "type": gtype, "split": gsplit, "question": gq.strip(), "expected_answer": gans.strip(),
                       "key_facts": [f.strip() for f in gfacts.splitlines() if f.strip()], "answerable": gok, "evidence": evs if gok else []}
                items = [g for g in raw_gold if g["id"] not in (cur["id"], new["id"])] + [new]
                testbed.save_gold(kb, sorted(items, key=lambda g: g["id"]))
                st.success(f"saved {new['id']}")
                st.rerun()
        if pick_id != "+ new question" and st.button("delete this question", type="tertiary"):
            testbed.save_gold(kb, [g for g in raw_gold if g["id"] != cur["id"]])
            st.rerun()
    with b:
        md('<div class="rw-eyebrow">evidence · where the answer is</div>')
        for i, e in enumerate(evs):
            c1, c2 = st.columns([5, 1])
            c1.caption(f"{e['doc']} p{e.get('page', '?')} · " + (f"quote: “{e['quote'][:60]}”" if e.get("quote") else f"box {e['box']}" if e.get("box") else "whole page"))
            if c2.button("remove", key=f"rm_{ev_key}_{i}", type="tertiary"):
                evs.pop(i)
                st.rerun()
        visual_docs = [d for d in docs]
        if visual_docs:
            d_pick = st.selectbox("doc", visual_docs, format_func=lambda p: p.name, key=f"evdoc_{ev_key}")
            pages_ev = kbs.pages(kb, d_pick)
            p_pick = st.number_input("page", 1, len(pages_ev), 1, key=f"evpg_{ev_key}") if len(pages_ev) > 1 else 1
            pg = pages_ev[p_pick - 1]
            md('<div class="rw-eyebrow">area on the page (3x3 cells) or an exact quote</div>')
            cells = []
            for row in (["nw", "n", "ne"], ["w", "c", "e"], ["sw", "s", "se"]):
                cs = st.columns(3)
                cells += [n for c, n in zip(cs, row) if c.checkbox(n, key=f"cell_{ev_key}_{n}")]
            quote = st.text_input("exact quote from the text (best for text questions)", key=f"q_{ev_key}")
            if st.button("add evidence", type="secondary"):
                e = {"doc": d_pick.name, "page": p_pick}
                if quote.strip():
                    e["quote"] = quote.strip()
                elif cells:
                    e["box"] = testbed.cells_box(cells)
                evs.append(e)
                st.rerun()
            if pg["image"]:
                boxes = [(tuple(e["box"]), "gold", "#0F8A82") for e in evs if e.get("box") and e["doc"] == d_pick.name and e.get("page") == p_pick]
                if cells:
                    boxes.append((tuple(testbed.cells_box(cells)), "new", "#B57A1A"))
                st.image(draw_boxes(pg["image"], boxes, 800), width="stretch")
            with st.expander("draft questions for this page with ai"):
                st.caption("the model reads this page and suggests questions. check every one against the page before adding it")
                if st.button("draft 3 questions", type="secondary", disabled=not has_key(provider)):
                    with st.spinner("drafting"):
                        st.session_state[f"drafts_{kb}"] = testbed.draft(kb, provider, d_pick, p_pick)
                for j, dr in enumerate(st.session_state.get(f"drafts_{kb}", [])):
                    ev0 = dr["evidence"][0]
                    md(f'<div class="rw-card"><div class="rw-eyebrow">{esc(dr["type"])} · {esc(ev0["doc"])} p{ev0["page"]} · '
                       + ("quote" if ev0.get("quote") else f"box {ev0.get('box')}" if ev0.get("box") else "page") + f'</div><b>{esc(dr["question"])}</b>'
                       f'<div class="rw-why">{esc(dr["expected_answer"])}</div><div class="rw-meta">key facts: {esc("; ".join(dr["key_facts"]))}</div></div>')
                    if st.button("add to gold", key=f"add_draft_{j}", type="secondary"):
                        items = list(raw_gold) + [{"id": testbed.next_id(raw_gold, "q"), "split": "dev", **dr}]
                        testbed.save_gold(kb, items)
                        st.session_state[f"drafts_{kb}"].pop(j)
                        st.rerun()

if view == "compare":
    run_list = evals.list_runs(kb)
    if not run_list:
        md('<div class="rw-empty">run some evals first. then compare chunkers, embedders, retrievers and rerankers here</div>')
    else:
        runs = [evals.load(kb, r) for r in run_list]
        tbl = pd.DataFrame([{"when": r["at"], "note": r.get("note", ""), "chunker": r.get("chunker", r.get("strategy")), "embedder": r.get("embedder", "minilm"),
                             "desc": r.get("style", "prose"), "ctx": r.get("contextual", False), "retriever": r["settings"]["retriever"],
                             "rerank": r["settings"]["reranker"], "check": r["settings"].get("check", "none"), "labels": r["settings"].get("labels", False),
                             "+page": r["settings"]["with_page"], "provider": r["provider"], "split": r.get("split", "all"),
                             "retrieval": r["summary"]["retrieval_pass"], "r@5": r["summary"].get("recall@5"), "mrr": r["summary"].get("mrr"),
                             "answer": r["summary"]["answer_score"], "grounded": r["summary"]["grounded"],
                             "halluc.": r["summary"].get("hallucinated"), "calls/q": r["summary"].get("llm_calls"), "sec": r["summary"]["avg_seconds"],
                             "file": r["file"]} for r in runs])
        st.dataframe(tbl, hide_index=True, width="stretch", height=280,
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("retrieval", "r@5", "mrr", "answer", "grounded")})
        names = {r["file"]: f"{r['at'][5:]} · {evals.label(r)}" for r in runs}
        pick = st.multiselect("compare up to 4 runs", run_list, default=run_list[:min(3, len(run_list))], max_selections=4, format_func=names.get)
        if pick:
            series = [names[f] for f in pick]
            rows = [{"metric": m, "run": names[r["file"]], "value": r["summary"].get(key2)} for r in runs if r["file"] in pick
                    for m, key2 in METRICS if r["summary"].get(key2) is not None]
            st.altair_chart(chart(rows, series, 300), width="stretch")
            heat = [{"type": t, "col": names[r["file"]][:60], "value": v["answer_score"] if v["answer_score"] is not None else 0, "n": v["n"]}
                    for r in runs if r["file"] in pick for t, v in r["summary"].get("by_type", {}).items()]
            if heat:
                md('<div class="rw-eyebrow">answer score by question type</div>')
                st.altair_chart(type_heat(heat), width="stretch")
        md('<div class="rw-note">only measured numbers here. answer and grounding come from the llm judge unless you overrode them, so spot-check before quoting. compare runs on the same split.</div>')
