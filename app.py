"""
app.py — Application Streamlit Analyse de Sentiment Tunisien
============================================================
Upload un fichier CSV/JSON → nettoyage → MARBERT → dashboard interactif
"""

import os, re, time, io, json, base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse Sentiment Tunisien",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    'positif': '#1e8e3e',
    'neutre':  '#e37400',
    'negatif': '#d93025',
}
ID2LABEL = {0: 'negatif', 1: 'neutre', 2: 'positif'}

# ── CSS minimal ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background:#f8f9fa; border-radius:12px; padding:20px 16px;
    text-align:center; border:1px solid #e0e0e0;
}
.kpi-val  { font-size:2.2rem; font-weight:700; margin:0; }
.kpi-lbl  { font-size:0.85rem; color:#666; margin:0; }
.badge-pos{ background:#e8f5e9; color:#1e8e3e; padding:2px 8px;
            border-radius:999px; font-size:0.8rem; font-weight:600; }
.badge-neg{ background:#fce8e6; color:#d93025; padding:2px 8px;
            border-radius:999px; font-size:0.8rem; font-weight:600; }
.badge-neu{ background:#fef3e2; color:#e37400; padding:2px 8px;
            border-radius:999px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MODELE (mis en cache)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Chargement du modèle MARBERT...")
def load_model():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from langdetect import detect, LangDetectException
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()

    MODEL_PATH = './models/MARBERT-balanced-after-gen-neut'
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()   # test rapide pour détecter un contexte CUDA corrompu
            device = 'cuda'
        except RuntimeError:
            device = 'cpu'
    else:
        device = 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=3,
        id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    ).to(device)
    mdl.eval()
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    return tok, mdl, device, client, F, torch, detect, LangDetectException


# ══════════════════════════════════════════════════════════════════════════════
# NETTOYAGE AVANCE (mêmes filtres que nettoyage_avance.ipynb)
# ══════════════════════════════════════════════════════════════════════════════
ARABIC_RE = re.compile(r'[؀-ۿ]')
LATIN_RE  = re.compile(r'[a-zA-Z]')

def _no_content(t):
    return not ARABIC_RE.search(t) and not LATIN_RE.search(t)

def _gibberish_latin(t):
    return bool(re.search(r'(?i)[bcdfghjklmnpqrstvwxyz]{7,}', t))

def _char_repeat(t, seuil=0.28):
    s = re.sub(r'\s', '', t.lower())
    if not s: return True
    return any(s.count(c)/len(s) > seuil for c in set(s) if c.isalpha())

def _word_repeat(t, seuil=0.45):
    w = t.lower().split()
    if len(w) < 5: return False
    return Counter(w).most_common(1)[0][1] / len(w) > seuil

def _gibberish_random(t):
    for w in t.split():
        wc = re.sub(r'[^a-zA-Z؀-ۿ]', '', w)
        if len(wc) >= 15:
            if not re.findall(r'[aeiouAEIOUَ-ِاوي]', wc):
                return True
    return False

def nettoyer_texte(t: str):
    t = str(t).strip()
    if not t or t == 'nan':               return None
    if _no_content(t):                    return None
    if _gibberish_latin(t):               return None
    if _gibberish_random(t):              return None
    if _char_repeat(t):                   return None
    if _word_repeat(t):                   return None
    return t


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION LANGUE
# ══════════════════════════════════════════════════════════════════════════════
_TUNISIAN = {
    'ya','ma','fi','eli','elli','mel','ken','bch','mta3','weld','bent',
    'yji','nhar','kol','fel','barcha','bahi','taw','shkoun','wesh',
    'sahih','famma','aslema','marhba','nchallah','inchallah','mabrouk',
    'saha','brabi','mouch','barra','kima','chnowa','winou','fein','leh',
    'barcha','9dar','3leh','sahbi','mriguel','kifech','kifach','chbik',
}

def detect_language(text, detect_fn, LangDetectException):
    t = text.strip()
    if not t: return 'arabizi'
    ar = len(re.findall(r'[؀-ۿ]', t))
    total = max(len(re.findall(r'\w', t)), 1)
    if ar / total > 0.25: return 'arabic'
    score  = len(re.findall(r'[379]', t)) * 2
    score += len(re.findall(r'[48]',  t))
    mots   = set(re.findall(r'[a-zA-Z]+', t.lower()))
    score += len(mots & _TUNISIAN) * 3
    if score >= 3: return 'arabizi'
    try:
        lang = detect_fn(t)
        if lang == 'fr': return 'french'
        if lang == 'en': return 'english'
    except LangDetectException:
        pass
    return 'arabizi'


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT FICHIER UPLOADE
# ══════════════════════════════════════════════════════════════════════════════
def charger_fichier(uploaded_file):
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()

    if name.endswith('.json') or name.endswith('.jsonl'):
        try:
            data = json.loads(raw.decode('utf-8'))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.read_json(io.BytesIO(raw), lines=True)
        except Exception:
            df = pd.read_json(io.BytesIO(raw), lines=True)
    else:
        for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                continue

    # Trouver colonne text
    col_text = None
    for c in df.columns:
        if c.lower() in ('text', 'texte', 'comment', 'commentaire', 'message', 'content'):
            col_text = c
            break
    if col_text is None:
        str_cols = df.select_dtypes(include='object').columns.tolist()
        if str_cols:
            col_text = str_cols[-1]

    if col_text is None:
        st.error("Impossible de trouver une colonne texte dans ce fichier.")
        return None

    df = df.rename(columns={col_text: 'text'})
    df['text'] = df['text'].astype(str).str.strip()
    return df[['text']]


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════
_PROMPT_TRAD = (
    "Traduis ce texte en arabe. Réponds uniquement avec la traduction, sans explication.\n"
    "Texte : {text}\nTraduction arabe :"
)

def lancer_analyse(df_raw, tok, mdl, device, groq_client, F, torch, detect_fn, LDE):
    texts_raw = df_raw['text'].tolist()
    n_total   = len(texts_raw)

    st.info(f"**{n_total} textes** chargés — pipeline en cours...")
    prog = st.progress(0, text="Nettoyage...")

    # ── 1. Nettoyage
    textes_propres = [(i, nettoyer_texte(t)) for i, t in enumerate(texts_raw)]
    textes_propres = [(i, t) for i, t in textes_propres if t]
    n_clean = len(textes_propres)
    prog.progress(0.10, text=f"Nettoyage terminé — {n_clean}/{n_total} textes valides")

    # ── 2. Détection langue
    prog.progress(0.20, text="Détection des langues...")
    results_lang = [(i, t, detect_language(t, detect_fn, LDE)) for i, t in textes_propres]

    to_translate = [(i, t, l) for i, t, l in results_lang if l in ('french', 'english')]
    prog.progress(0.30, text=f"Traduction de {len(to_translate)} textes (fr/en) via Groq...")

    # ── 3. Traductions Groq
    translated_map = {}
    for idx, (i, t, l) in enumerate(to_translate):
        try:
            resp = groq_client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': _PROMPT_TRAD.format(text=t)}],
                max_tokens=256, temperature=0,
            )
            translated_map[i] = resp.choices[0].message.content.strip()
        except Exception:
            translated_map[i] = t
        time.sleep(0.25)
        if (idx + 1) % 10 == 0:
            pct = 0.30 + (idx / max(len(to_translate), 1)) * 0.20
            prog.progress(pct, text=f"Traductions : {idx+1}/{len(to_translate)}")

    prog.progress(0.50, text="Prédictions MARBERT...")

    # ── 4. Prédictions MARBERT
    indices   = [i    for i, t, l in results_lang]
    orig_txts = [t    for i, t, l in results_lang]
    langs     = [l    for i, t, l in results_lang]
    mar_txts  = [translated_map.get(i, t) for i, t, l in results_lang]

    BATCH = 32
    all_probs = []
    effective_device = device
    for b in range(0, len(mar_txts), BATCH):
        batch = mar_txts[b:b+BATCH]
        try:
            enc = tok(batch, return_tensors='pt', padding=True,
                      truncation=True, max_length=128).to(effective_device)
            with torch.no_grad():
                logits = mdl(**enc).logits
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        except RuntimeError as cuda_err:
            if 'CUDA' in str(cuda_err) and effective_device == 'cuda':
                st.warning("CUDA indisponible — basculement sur CPU (plus lent).")
                effective_device = 'cpu'
                mdl.to('cpu')
                enc = tok(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
                with torch.no_grad():
                    logits = mdl(**enc).logits
                all_probs.append(F.softmax(logits, dim=-1).numpy())
            else:
                raise
        pct = 0.50 + (min(b + BATCH, len(mar_txts)) / len(mar_txts)) * 0.45
        prog.progress(pct, text=f"MARBERT : {min(b+BATCH, len(mar_txts))}/{len(mar_txts)}")

    all_probs = np.vstack(all_probs)
    prog.progress(1.0, text="Analyse terminée ✅")

    # ── 5. Construire DataFrame résultats
    rows = []
    for k, (i, ot, l, mt, p) in enumerate(zip(indices, orig_txts, langs, mar_txts, all_probs)):
        pid  = int(np.argmax(p))
        rows.append({
            'text':          ot,
            'langue':        l,
            'texte_marbert': mt if mt != ot else '',
            'sentiment':     ID2LABEL[pid],
            'confiance':     round(float(p[pid]), 4),
            'prob_neg':      round(float(p[0]), 4),
            'prob_neu':      round(float(p[1]), 4),
            'prob_pos':      round(float(p[2]), 4),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE DES MOTS
# ══════════════════════════════════════════════════════════════════════════════
_STOPWORDS = {
    # français
    'de','la','le','les','un','une','des','et','est','je','tu','il','elle',
    'nous','vous','ils','pas','ne','que','qui','dans','sur','avec','pour',
    'mais','ou','donc','ni','car','au','aux','du','se','on','en','par','plus',
    'si','tout','bien','aussi','comme','alors','même','très','avoir','être',
    'cest','jai','jai','cela','ca','ça','ya','nan','cela',
    # arabizi communs
    'w','ya','ma','fi','el','al','mel','men','min','3la','3al','ki','bch',
    'mta3','mte3','rah','haka','la','walla','wallah','ana','inti','ahna',
    'wella','wla','taw','fel','fil','lel','li','lli','eli','elli','ken',
    'barra','howa','heya','houma','kol','nhar','w9et','famma','walla',
    'inchallah','nchallah','brabi','baraka','wint','winta','sahbi',
    # arabe commun
    'من','في','على','إلى','هذا','هذه','التي','الذي','أن','كان','مع',
    'عن','قد','ما','لا','كل','هو','هي','هم','أنا','انا','نحن','هذي',
    'الله','والله','wallah','yrab','يرب','ربي',
}

def top_mots(df_sent, sentiment, n=15):
    textes = df_sent[df_sent.sentiment == sentiment]['text'].tolist()
    words  = []
    for t in textes:
        for w in re.findall(r'[a-zA-Z؀-ۿ]{3,}', t.lower()):
            if w not in _STOPWORDS and len(w) >= 3:
                words.append(w)
    counts = Counter(words).most_common(n)
    return pd.DataFrame(counts, columns=['mot', 'freq'])


# ══════════════════════════════════════════════════════════════════════════════
# ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
# ══════════════════════════════════════════════════════════════════════════════
_ABSA_ASPECTS   = ['connexion','débit','service_client','prix',
                   'coupures','installation','application','equipement']
_ABSA_SENTIMENTS = {'positif','negatif','neutre'}

_ABSA_PROMPT = """\
Tu analyses des avis clients pour une entreprise de télécommunications tunisienne.
Pour chaque commentaire numéroté, identifie les aspects mentionnés et leur sentiment.

Aspects autorisés : connexion, débit, service_client, prix, coupures, installation, application, equipement

Définitions :
- connexion     : qualité/stabilité de la connexion internet
- débit         : vitesse, bandwidth, rapidité
- service_client: accueil, support, conseillers, hotline
- prix          : tarif, abonnement, rapport qualité/prix, facture
- coupures      : pannes, interruptions, coupures de service
- installation  : mise en place, intervention technicien
- application   : app mobile, espace client web
- equipement    : box, modem, routeur, câble

Commentaires :
{entries}

Réponds UNIQUEMENT avec ce JSON compact (aucun texte autour) :
{{"r":[{{"i":1,"a":[{{"n":"connexion","s":"negatif"}}]}},{{"i":2,"a":[]}}]}}

Règles strictes :
- "n" = exactement l'un des aspects autorisés
- "s" = exactement "positif", "negatif" ou "neutre"
- Si aucun aspect clair → "a":[]
- Chaque commentaire doit apparaître dans "r\""""


def _absa_parse(content, batch_size):
    results = [[] for _ in range(batch_size)]
    try:
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if not m:
            return results
        data = json.loads(m.group())
        for item in data.get('r', []):
            idx = int(item.get('i', 0)) - 1
            if 0 <= idx < batch_size:
                for asp in item.get('a', []):
                    n = str(asp.get('n', '')).strip().lower()
                    s = str(asp.get('s', '')).strip().lower()
                    if n in _ABSA_ASPECTS and s in _ABSA_SENTIMENTS:
                        results[idx].append({'aspect': n, 'aspect_sentiment': s})
    except Exception:
        pass
    return results


def run_absa_on_df(df, groq_client, batch_size=5, delay=0.4):
    """Lance ABSA sur df (colonnes: text, sentiment, langue).
    Retourne un DataFrame long-format : une ligne par aspect par commentaire."""
    texts  = df['text'].astype(str).tolist()
    rows   = []
    total  = len(texts)
    prog   = st.progress(0, text="ABSA — analyse des aspects...")

    for start in range(0, total, batch_size):
        chunk = texts[start:start + batch_size]
        entries = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(chunk))
        try:
            resp = groq_client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user',
                           'content': _ABSA_PROMPT.format(entries=entries)}],
                max_tokens=400,
                temperature=0,
            )
            content = resp.choices[0].message.content.strip()
        except Exception:
            content = '{"r":[]}'

        parsed = _absa_parse(content, len(chunk))

        for local_i, row_idx in enumerate(range(start, min(start + batch_size, total))):
            base_row = df.iloc[row_idx]
            aspects  = parsed[local_i]
            base = {
                'text'            : base_row.get('text', ''),
                'langue'          : base_row.get('langue', ''),
                'sentiment_global': base_row.get('sentiment', ''),
                'confiance'       : base_row.get('confiance', None),
            }
            if aspects:
                for asp in aspects:
                    rows.append({**base, **asp})
            else:
                rows.append({**base,
                             'aspect'          : 'autre',
                             'aspect_sentiment': base_row.get('sentiment', 'neutre')})

        time.sleep(delay)
        done = min(start + batch_size, total)
        prog.progress(done / total, text=f"ABSA : {done}/{total} commentaires")

    prog.progress(1.0, text="Analyse ABSA terminée ✅")
    return pd.DataFrame(rows)


def afficher_absa(df_absa):
    """Affiche les visualisations ABSA."""
    df_real = df_absa[df_absa.aspect != 'autre'].copy()
    if df_real.empty:
        st.warning("Aucun aspect détecté dans les commentaires filtrés.")
        return

    COLORS_S = {'positif': '#1e8e3e', 'negatif': '#d93025', 'neutre': '#e37400'}

    # ── KPIs ──────────────────────────────────────────────────────────────────
    n_asp    = df_real.groupby('text').ngroups
    n_neg    = (df_real.aspect_sentiment == 'negatif').sum()
    top_prob = df_real[df_real.aspect_sentiment == 'negatif']['aspect'].value_counts()
    top_neg_asp = top_prob.index[0] if not top_prob.empty else '—'

    c1, c2, c3 = st.columns(3)
    c1.metric("Commentaires avec aspects détectés", f"{n_asp:,}")
    c2.metric("Mentions négatives d'aspects", f"{int(n_neg):,}")
    c3.metric("Aspect le plus problématique", top_neg_asp.replace('_', ' ').title())
    st.markdown("---")

    # ── Chart 1 : Stacked bar aspects × sentiment ─────────────────────────────
    st.subheader("Répartition des sentiments par aspect")
    pivot = (df_real.groupby(['aspect','aspect_sentiment'])
             .size().unstack(fill_value=0).reset_index())
    for col in ['positif','negatif','neutre']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot['total'] = pivot['positif'] + pivot['negatif'] + pivot['neutre']
    pivot['nss']   = ((pivot['positif'] - pivot['negatif']) / pivot['total'] * 100).round(1)
    pivot = pivot.sort_values('negatif', ascending=False)

    # Normaliser en %
    pivot_pct = pivot.copy()
    for s in ['positif','negatif','neutre']:
        pivot_pct[s] = (pivot[s] / pivot['total'] * 100).round(1)

    fig = go.Figure()
    for sent, color in [('negatif','#d93025'),('neutre','#e37400'),('positif','#1e8e3e')]:
        fig.add_trace(go.Bar(
            name=sent.capitalize(),
            x=pivot_pct['aspect'].str.replace('_',' ').str.title(),
            y=pivot_pct[sent],
            marker_color=color,
            text=pivot_pct[sent].apply(lambda v: f'{v:.0f}%'),
            textposition='inside',
        ))
    fig.update_layout(
        barmode='stack', height=380,
        yaxis=dict(title='%', range=[0,100]),
        xaxis_title='', legend_title='Sentiment',
        margin=dict(t=10, b=0, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2 : NSS par aspect (bullet chart horizontal) ────────────────────
    st.subheader("NSS par aspect — Net Sentiment Score")
    st.caption("NSS = (% positifs − % négatifs). Vert = satisfait, Rouge = problème.")
    pivot_sorted = pivot.sort_values('nss')
    colors_nss = ['#1e8e3e' if v >= 0 else '#d93025' for v in pivot_sorted['nss']]
    fig2 = go.Figure(go.Bar(
        x=pivot_sorted['nss'],
        y=pivot_sorted['aspect'].str.replace('_',' ').str.title(),
        orientation='h',
        marker_color=colors_nss,
        text=pivot_sorted['nss'].apply(lambda v: f'{v:+.1f}'),
        textposition='outside',
    ))
    fig2.add_vline(x=0, line_dash='dash', line_color='#888', line_width=1.5)
    fig2.update_layout(
        height=320, margin=dict(t=10, b=0, l=0, r=60),
        xaxis_title='NSS (pts)', yaxis_title='',
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3 : Bubble — volume × NSS ──────────────────────────────────────
    st.subheader("Volume de mentions vs NSS")
    st.caption("La taille de la bulle = nombre de mentions. Position X = NSS.")
    fig3 = px.scatter(
        pivot,
        x='nss', y='aspect',
        size='total', color='nss',
        color_continuous_scale=['#d93025','#e37400','#1e8e3e'],
        color_continuous_midpoint=0,
        text=pivot['aspect'].str.replace('_',' ').str.title(),
        size_max=60,
        hover_data={'total': True, 'nss': True},
    )
    fig3.update_traces(textposition='middle right')
    fig3.add_vline(x=0, line_dash='dash', line_color='#888')
    fig3.update_layout(
        height=380, margin=dict(t=10, b=0, l=0, r=0),
        xaxis_title='NSS (pts)', yaxis_title='',
        yaxis=dict(showticklabels=False),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Drill-down par aspect ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Drill-down par aspect")
    asp_choice = st.selectbox(
        "Choisir un aspect",
        options=_ABSA_ASPECTS,
        format_func=lambda x: x.replace('_', ' ').title(),
    )
    df_asp = df_real[df_real.aspect == asp_choice]
    if df_asp.empty:
        st.info("Aucun commentaire détecté pour cet aspect.")
        return

    ca, cb, cc = st.columns(3)
    ca.metric("Positifs", int((df_asp.aspect_sentiment == 'positif').sum()))
    cb.metric("Négatifs", int((df_asp.aspect_sentiment == 'negatif').sum()))
    cc.metric("Neutres",  int((df_asp.aspect_sentiment == 'neutre').sum()))

    for sent in ['negatif', 'positif', 'neutre']:
        sub = df_asp[df_asp.aspect_sentiment == sent].head(4)
        if sub.empty:
            continue
        color = COLORS_S[sent]
        st.markdown(f"**{sent.upper()}**")
        for _, r in sub.iterrows():
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:9px 14px;'
                f'margin:4px 0;background:{color}18;border-radius:6px;color:inherit;">'
                f'<small style="opacity:0.6">{r.langue}</small><br>{r.text}</div>',
                unsafe_allow_html=True)

    st.markdown("")
    csv_asp = df_asp.to_csv(index=False)
    st.download_button(f"⬇️ Export CSV — {asp_choice}",
                       csv_asp, f"absa_{asp_choice}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# GENERATEUR DE RAPPORT HTML
# ══════════════════════════════════════════════════════════════════════════════
def generate_report_html(df, company_name, analyst_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
    n     = len(df)
    n_pos = int((df.sentiment == 'positif').sum())
    n_neg = int((df.sentiment == 'negatif').sum())
    n_neu = int((df.sentiment == 'neutre').sum())
    nss   = round(((n_pos - n_neg) / max(n, 1)) * 100, 1)
    conf_moy  = df.confiance.mean() if n else 0
    high_conf = int((df.confiance >= 0.90).sum())
    low_conf  = int((df.confiance <  0.70).sum())

    def fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{data}"

    # ── Chart 1 : Donut sentiments ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.8, 4))
    _, _, autotexts = ax.pie(
        [n_pos, n_neg, n_neu],
        labels=['Positif', 'Négatif', 'Neutre'],
        colors=['#1e8e3e', '#d93025', '#e37400'],
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.62, edgecolor='white', linewidth=2),
        pctdistance=0.77,
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
    ax.set_title('Répartition sentiments', fontsize=13, fontweight='bold', pad=12)
    pie_b64 = fig_to_b64(fig)

    # ── Chart 2 : Langues ─────────────────────────────────────────────────────
    lang_counts = df.langue.value_counts()
    cmap = {'arabic': '#4285f4', 'arabizi': '#fbbc04', 'french': '#34a853', 'english': '#ea4335'}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bars = ax.bar(lang_counts.index, lang_counts.values,
                  color=[cmap.get(l, '#9e9e9e') for l in lang_counts.index],
                  edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, lang_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, lang_counts.max() * 1.18)
    ax.set_title('Distribution des langues', fontsize=13, fontweight='bold')
    ax.set_ylabel('Commentaires')
    lang_b64 = fig_to_b64(fig)

    # ── Chart 3 : Confiance ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(df.confiance, bins=20, color='#1a73e8', edgecolor='white', alpha=0.85)
    ax.axvline(0.90, color='#34a853', linestyle='--', linewidth=1.5, label='90%')
    ax.axvline(0.70, color='#ea4335', linestyle='--', linewidth=1.5, label='70%')
    ax.set_title('Distribution de la confiance', fontsize=13, fontweight='bold')
    ax.set_xlabel('Score'); ax.set_ylabel('Commentaires'); ax.legend(fontsize=9)
    conf_b64 = fig_to_b64(fig)

    # ── Charts 4-6 : Top mots ─────────────────────────────────────────────────
    word_imgs = {}
    for sent, color in [('positif', '#1e8e3e'), ('negatif', '#d93025'), ('neutre', '#e37400')]:
        wdf = top_mots(df, sent, n=10)
        if wdf.empty:
            continue
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        yp = list(range(len(wdf)))
        ax.barh(yp, wdf.freq.tolist(), color=color, alpha=0.85, edgecolor='white')
        ax.set_yticks(yp)
        ax.set_yticklabels(wdf.mot.tolist(), fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f'Mots fréquents — {sent.upper()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Occurrences')
        word_imgs[sent] = fig_to_b64(fig)

    # ── Exemples représentatifs ───────────────────────────────────────────────
    examples_html = ''
    for sent, color, emoji in [
        ('positif', '#1e8e3e', '😊'), ('negatif', '#d93025', '😠'), ('neutre', '#e37400', '😐')
    ]:
        sub = df[df.sentiment == sent].nlargest(3, 'confiance')
        if sub.empty:
            continue
        rows = ''.join(
            f'<tr><td style="padding:8px 12px;border-bottom:1px solid #eee;max-width:420px">{r.text}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center">{r.langue}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;font-weight:700">{r.confiance:.1%}</td></tr>'
            for _, r in sub.iterrows()
        )
        examples_html += f'''
        <div style="margin-bottom:28px">
          <h4 style="color:{color};margin:0 0 10px 0">{emoji} {sent.upper()}</h4>
          <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:{color}15">
              <th style="padding:8px 12px;text-align:left;border-bottom:2px solid {color}40">Commentaire</th>
              <th style="padding:8px 12px;text-align:center;border-bottom:2px solid {color}40">Langue</th>
              <th style="padding:8px 12px;text-align:center;border-bottom:2px solid {color}40">Confiance</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>'''

    wc_html = ''.join(
        f'<div style="flex:1;min-width:200px"><img src="{word_imgs[s]}" style="width:100%;border-radius:8px;border:1px solid #e0e0e0"></div>'
        for s in ['positif', 'negatif', 'neutre'] if s in word_imgs
    )
    nss_color = '#1e8e3e' if nss >= 10 else ('#e37400' if nss >= 0 else '#d93025')
    nss_label = 'Positif' if nss >= 10 else ('Neutre' if nss >= 0 else 'Négatif')

    return f'''<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="UTF-8">
<title>Rapport Sentiment — {company_name}</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0;color:#212121;background:#fff}}
  .cover{{background:linear-gradient(135deg,#0d47a1 0%,#1a73e8 60%,#42a5f5 100%);color:white;padding:56px 72px}}
  .cover h1{{font-size:2rem;margin:0 0 8px}}
  .cover h2{{font-size:1rem;font-weight:400;opacity:.85;margin:0 0 28px}}
  .cover-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;border-top:1px solid rgba(255,255,255,.25);padding-top:22px}}
  .cover-item label{{display:block;font-size:10px;text-transform:uppercase;letter-spacing:1px;opacity:.6;margin-bottom:4px}}
  .cover-item span{{font-size:.95rem;font-weight:600}}
  .sec{{padding:36px 72px}}
  .sec h2{{font-size:1.3rem;color:#1a73e8;border-bottom:2px solid #e8f0fe;padding-bottom:8px;margin-bottom:22px}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px}}
  .kpi{{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:10px;padding:18px 10px;text-align:center}}
  .kpi .v{{font-size:1.8rem;font-weight:700;margin:0}}
  .kpi .l{{font-size:.78rem;color:#666;margin:4px 0 0}}
  .row{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:24px}}
  .row>div{{flex:1;min-width:180px}}
  .row img{{width:100%;border-radius:8px;border:1px solid #e0e0e0}}
  .meta{{background:#f8f9fa;border-radius:8px;padding:12px 18px;font-size:13px;color:#444}}
  .footer{{background:#f8f9fa;border-top:1px solid #e0e0e0;padding:14px 72px;text-align:center;font-size:11px;color:#888}}
  @media print{{.sec{{page-break-inside:avoid}}}}
</style>
</head><body>

<div class="cover">
  <div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;opacity:.65;margin-bottom:14px">Rapport d'analyse de sentiment</div>
  <h1>Analyse des Commentaires Clients</h1>
  <h2>{company_name} — Rapport automatisé · MARBERT + LIME</h2>
  <div class="cover-grid">
    <div class="cover-item"><label>Client</label><span>{company_name}</span></div>
    <div class="cover-item"><label>Analyste</label><span>{analyst_name or "—"}</span></div>
    <div class="cover-item"><label>Date</label><span>{date_str}</span></div>
  </div>
</div>

<div class="sec">
  <h2>Résumé exécutif</h2>
  <div class="kpi-grid">
    <div class="kpi"><p class="v" style="color:#1a73e8">{n:,}</p><p class="l">Commentaires</p></div>
    <div class="kpi"><p class="v" style="color:#1e8e3e">{n_pos/max(n,1)*100:.1f}%</p><p class="l">😊 Positifs</p></div>
    <div class="kpi"><p class="v" style="color:#d93025">{n_neg/max(n,1)*100:.1f}%</p><p class="l">😠 Négatifs</p></div>
    <div class="kpi"><p class="v" style="color:#e37400">{n_neu/max(n,1)*100:.1f}%</p><p class="l">😐 Neutres</p></div>
    <div class="kpi"><p class="v" style="color:{nss_color}">{nss:+.1f}</p><p class="l">NSS — {nss_label}</p></div>
  </div>
  <div class="meta">
    <strong>Confiance moyenne :</strong> {conf_moy:.1%} &nbsp;·&nbsp;
    <strong>≥ 90% :</strong> {high_conf} textes ({high_conf/max(n,1)*100:.0f}%) &nbsp;·&nbsp;
    <strong>&lt; 70% :</strong> {low_conf} textes ({low_conf/max(n,1)*100:.0f}%)
  </div>
</div>

<div class="sec" style="padding-top:0">
  <h2>Visualisations</h2>
  <div class="row">
    <div><img src="{pie_b64}"></div>
    <div><img src="{lang_b64}"></div>
    <div><img src="{conf_b64}"></div>
  </div>
</div>

<div class="sec" style="padding-top:0">
  <h2>Mots récurrents par sentiment</h2>
  <p style="color:#666;font-size:13px;margin-bottom:18px">Les mots les plus fréquents par catégorie révèlent les sujets qui génèrent satisfaction ou insatisfaction.</p>
  <div class="row">{wc_html}</div>
</div>

<div class="sec" style="padding-top:0">
  <h2>Exemples représentatifs</h2>
  {examples_html}
</div>

<div class="footer">
  Rapport généré le {date_str} &nbsp;·&nbsp; {company_name} &nbsp;·&nbsp; Propulsé par MARBERT · Groq LLaMA-3.3 · LIME
</div>
</body></html>'''


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def badge(s):
    return f'<span class="badge-{s}">{s}</span>'

def afficher_dashboard(df, seuil_conf, filtres_sent, filtres_lang):
    df_f = df.copy()
    if filtres_sent:
        df_f = df_f[df_f.sentiment.isin(filtres_sent)]
    if filtres_lang:
        df_f = df_f[df_f.langue.isin(filtres_lang)]
    df_f = df_f[df_f.confiance >= seuil_conf]

    page = st.radio("", ["📈 Vue Globale", "🎯 Aspects", "🔍 Analyse Détaillée", "🔬 LIME Explorer", "📁 Données"],
                    horizontal=True, label_visibility='collapsed')

    # ── PAGE 1 : Vue Globale ──────────────────────────────────────────────────
    if page == "📈 Vue Globale":
        n     = len(df_f)
        np_   = (df_f.sentiment == 'positif').sum()
        nn_   = (df_f.sentiment == 'negatif').sum()
        nu_   = (df_f.sentiment == 'neutre').sum()
        nss   = round(((np_ - nn_) / max(n, 1)) * 100, 1)
        conf_moy = df_f.confiance.mean() if n else 0

        # ── KPI cards
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, lbl, color in [
            (c1, f"{n:,}",                   "Commentaires", "#1a73e8"),
            (c2, f"{np_/max(n,1)*100:.1f}%", "😊 Positif",   COLORS['positif']),
            (c3, f"{nn_/max(n,1)*100:.1f}%", "😠 Négatif",   COLORS['negatif']),
            (c4, f"{nu_/max(n,1)*100:.1f}%", "😐 Neutre",    COLORS['neutre']),
            (c5, f"{nss:+.1f} pts",          "NSS",          "#1a73e8" if nss >= 0 else COLORS['negatif']),
        ]:
            col.markdown(
                f'<div class="kpi-card"><p class="kpi-val" style="color:{color}">{val}</p>'
                f'<p class="kpi-lbl">{lbl}</p></div>', unsafe_allow_html=True)

        st.caption(f"Confiance moyenne du modèle : **{conf_moy:.1%}**  |  "
                   f"NSS = (positifs - négatifs) / total × 100")
        st.markdown("---")

        # ── Ligne 1 : Donut + Langues + Confiance
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure(go.Pie(
                labels=['Positif','Négatif','Neutre'],
                values=[np_, nn_, nu_],
                hole=0.55,
                marker_colors=[COLORS['positif'], COLORS['negatif'], COLORS['neutre']],
                textinfo='percent+label', textfont_size=13,
            ))
            fig.update_layout(title="Répartition des sentiments",
                              showlegend=False, height=300,
                              margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            lang_counts = df_f.langue.value_counts().reset_index()
            lang_counts.columns = ['langue','count']
            fig3 = px.bar(lang_counts, x='langue', y='count',
                          title="Langues détectées", color='langue',
                          color_discrete_map={
                              'arabic':'#4285f4','arabizi':'#fbbc04',
                              'french':'#34a853','english':'#ea4335'})
            fig3.update_layout(height=300, margin=dict(t=40,b=0,l=0,r=0),
                               showlegend=False, yaxis_title="Nombre")
            st.plotly_chart(fig3, use_container_width=True)

        with col3:
            fig2 = px.histogram(df_f, x='confiance', nbins=20,
                                title="Distribution de la confiance",
                                color_discrete_sequence=['#1a73e8'])
            fig2.update_layout(height=300, margin=dict(t=40,b=0,l=0,r=0),
                               xaxis_title="Confiance", yaxis_title="Nombre")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("🔑 Mots récurrents par sentiment")
        st.caption("Quels mots reviennent le plus souvent dans chaque catégorie ? "
                   "Cela révèle les sujets qui génèrent satisfaction ou insatisfaction.")

        # ── Ligne 2 : Top mots par sentiment
        col4, col5, col6 = st.columns(3)
        for col, sent in zip([col4, col5, col6], ['negatif', 'positif', 'neutre']):
            with col:
                wdf = top_mots(df_f, sent, n=15)
                if wdf.empty:
                    st.info(f"Pas de données pour {sent}")
                    continue
                fig_w = px.bar(wdf, x='freq', y='mot', orientation='h',
                               title=f"Top mots — {sent.upper()}",
                               color_discrete_sequence=[COLORS[sent]])
                fig_w.update_layout(height=400, margin=dict(t=40,b=0,l=0,r=10),
                                    yaxis=dict(autorange='reversed'),
                                    xaxis_title="Occurrences", yaxis_title="")
                st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("---")
        st.subheader("⚔️ Mots distinctifs : Positif vs Négatif")
        st.caption("Les mots qui apparaissent beaucoup plus dans les commentaires négatifs "
                   "que positifs (et inversement) — ce sont les signaux forts.")

        df_neg = top_mots(df_f, 'negatif', 20)
        df_pos = top_mots(df_f, 'positif', 20)
        df_neg['sentiment'] = 'negatif'
        df_pos['sentiment'] = 'positif'

        # Mots exclusifs à chaque sentiment (pas dans l'autre)
        mots_neg = set(df_neg.mot)
        mots_pos = set(df_pos.mot)
        excl_neg = df_neg[df_neg.mot.isin(mots_neg - mots_pos)].head(8)
        excl_pos = df_pos[df_pos.mot.isin(mots_pos - mots_neg)].head(8)

        combined = pd.concat([
            excl_neg.assign(freq=-excl_neg.freq),
            excl_pos
        ])

        if not combined.empty:
            fig_comp = px.bar(
                combined, x='freq', y='mot', orientation='h',
                color='sentiment',
                color_discrete_map={'negatif': COLORS['negatif'], 'positif': COLORS['positif']},
                title="Mots caractéristiques (négatif ← | → positif)",
            )
            fig_comp.update_layout(height=380, margin=dict(t=40,b=0,l=0,r=0),
                                   yaxis=dict(autorange='reversed'),
                                   xaxis_title="← Négatif  |  Positif →",
                                   yaxis_title="", showlegend=False,
                                   xaxis=dict(tickformat='d'))
            st.plotly_chart(fig_comp, use_container_width=True)

    # ── PAGE 2 : Aspects (ABSA) ───────────────────────────────────────────────
    elif page == "🎯 Aspects":
        st.subheader("Analyse par Aspect — Que pensent les clients de chaque sujet ?")
        st.caption(
            "ABSA (Aspect-Based Sentiment Analysis) : au lieu de dire \"ce commentaire est négatif\", "
            "on identifie **quel sujet** est négatif. Propulsé par Groq LLaMA-3.3."
        )

        absa_key = 'df_absa'

        if absa_key not in st.session_state:
            st.info(
                "L'analyse ABSA n'a pas encore été lancée. "
                "Choisissez un mode et cliquez sur **Analyser**."
            )
            mode_absa = st.radio(
                "Mode d'analyse",
                ["Échantillon rapide (100 commentaires, ~1 min)",
                 "Dataset complet (tous les commentaires, ~5-10 min)"],
                index=0,
            )
            if st.button("🎯 Lancer l'analyse ABSA", type="primary"):
                _, _, _, groq_client, _, _, _, _ = load_model()
                sample = df_f.head(100) if "100" in mode_absa else df_f
                with st.spinner(f"Analyse ABSA en cours sur {len(sample)} commentaires..."):
                    df_absa = run_absa_on_df(sample, groq_client)
                st.session_state[absa_key] = df_absa
                st.rerun()
        else:
            df_absa = st.session_state[absa_key]
            # Afficher infos + option reset
            col_info, col_reset = st.columns([4, 1])
            col_info.caption(
                f"ABSA calculé sur **{df_absa['text'].nunique()}** commentaires — "
                f"**{len(df_absa[df_absa.aspect != 'autre'])}** aspects détectés."
            )
            if col_reset.button("🔄 Relancer"):
                del st.session_state[absa_key]
                st.rerun()
            afficher_absa(df_absa)

    # ── PAGE 3 : Analyse Détaillée ────────────────────────────────────────────
    elif page == "🔍 Analyse Détaillée":
        st.subheader("Top exemples par sentiment")
        for sent in ['positif', 'negatif', 'neutre']:
            sub = df_f[df_f.sentiment == sent].nlargest(5, 'confiance')
            if sub.empty: continue
            color = COLORS[sent]
            st.markdown(f"**{badge(sent)}**", unsafe_allow_html=True)
            for _, r in sub.iterrows():
                st.markdown(
                    f'<div style="border-left:4px solid {color};padding:10px 14px;'
                    f'margin:5px 0;background:{color}22;border-radius:6px;color:inherit;">'
                    f'<small style="opacity:0.65;font-weight:600">'
                    f'{r.langue} — confiance : {r.confiance:.1%}</small><br>'
                    f'{r.text}</div>',
                    unsafe_allow_html=True)
            st.markdown("")

        st.markdown("---")
        st.subheader("Tableau complet")
        show_cols = ['text', 'langue', 'sentiment', 'confiance']
        st.dataframe(df_f[show_cols].reset_index(drop=True), use_container_width=True, height=400)

        csv = df_f.to_csv(index=False, encoding='utf-8')
        st.download_button("⬇️ Télécharger les résultats (CSV)",
                           csv, "resultats_sentiment.csv", "text/csv")

    # ── PAGE 3 : LIME Explorer ────────────────────────────────────────────────
    elif page == "🔬 LIME Explorer":
        st.subheader("Explications LIME — pourquoi cette prédiction ?")
        st.caption("LIME montre quels mots ont le plus influencé la décision du modèle.")

        mode = st.radio("Source du texte", ["Sélectionner depuis les résultats", "Entrer un texte libre"])
        if mode == "Sélectionner depuis les résultats":
            opts = df_f.apply(lambda r: f"[{r.sentiment}] {r.text[:80]}", axis=1).tolist()
            if not opts:
                st.warning("Aucun résultat avec les filtres actuels.")
                return
            idx  = st.selectbox("Choisir un commentaire", range(len(opts)), format_func=lambda i: opts[i])
            row  = df_f.iloc[idx]
            text_lime = row.texte_marbert if row.texte_marbert else row.text
            st.info(f"**Texte original :** {row.text}")
            if row.texte_marbert:
                st.info(f"**Envoyé au modèle :** {row.texte_marbert}")
        else:
            text_libre = st.text_area("Entrez un texte à analyser")
            if not text_libre.strip():
                st.stop()
            tok_, mdl_, device_, groq_, F_, torch_, detect_fn, LDE = load_model()
            lang = detect_language(text_libre, detect_fn, LDE)
            text_lime = text_libre
            if lang in ('french', 'english'):
                try:
                    resp = groq_.chat.completions.create(
                        model='llama-3.3-70b-versatile',
                        messages=[{'role':'user','content':_PROMPT_TRAD.format(text=text_libre)}],
                        max_tokens=256, temperature=0)
                    text_lime = resp.choices[0].message.content.strip()
                    st.info(f"Traduit en arabe : {text_lime}")
                except Exception:
                    pass

        if st.button("🔬 Lancer LIME (~20 secondes)"):
            tok_, mdl_, device_, _, F_, torch_, _, _ = load_model()
            from lime.lime_text import LimeTextExplainer

            def predict_proba_lime(texts):
                enc = tok_(list(texts), return_tensors='pt', padding=True,
                           truncation=True, max_length=128).to(device_)
                with torch_.no_grad():
                    logits = mdl_(**enc).logits
                return F_.softmax(logits, dim=-1).cpu().numpy()

            with st.spinner("LIME en cours..."):
                exp_obj = LimeTextExplainer(
                    class_names=['negatif','neutre','positif'], bow=False
                ).explain_instance(text_lime, predict_proba_lime,
                                   num_features=10, num_samples=500, labels=[0,1,2])

            probs_lime = predict_proba_lime([text_lime])[0]
            pred_id    = int(np.argmax(probs_lime))
            pred_lbl   = ID2LABEL[pred_id]

            st.markdown(f"### Prédiction : {badge(pred_lbl)} ({probs_lime[pred_id]:.0%})",
                        unsafe_allow_html=True)

            cols = st.columns(3)
            for col, (cid, clbl) in zip(cols, [(0,'negatif'),(1,'neutre'),(2,'positif')]):
                weights = exp_obj.as_list(label=cid)
                if not weights: continue
                weights.sort(key=lambda x: -abs(x[1]))
                words  = [w[0] for w in weights]
                values = [w[1] for w in weights]
                bar_c  = [COLORS[clbl] if v > 0 else '#9e9e9e' for v in values]
                fig = go.Figure(go.Bar(x=values, y=words, orientation='h',
                                       marker_color=bar_c))
                marker = " ← PRÉDIT" if cid == pred_id else ""
                fig.update_layout(title=f"{clbl.upper()}{marker}",
                                  height=320, margin=dict(t=30,b=0,l=0,r=0),
                                  yaxis=dict(autorange='reversed'))
                col.plotly_chart(fig, use_container_width=True)

    # ── PAGE 4 : Données ──────────────────────────────────────────────────────
    elif page == "📁 Données":
        st.subheader(f"Dataset complet — {len(df_f)} entrées")
        st.dataframe(df_f.reset_index(drop=True), use_container_width=True, height=500)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ CSV", df_f.to_csv(index=False),
                           "predictions.csv", "text/csv")
        c2.download_button("⬇️ JSON", df_f.to_json(orient='records', force_ascii=False),
                           "predictions.json", "application/json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎯 Sentiment Tunisien")
        st.caption("Pipeline NLP — MARBERT + LIME")
        st.markdown("---")

        if st.session_state.get('analyse_done'):
            st.markdown("### Filtres")
            seuil = st.slider("Confiance minimale", 0.0, 1.0, 0.0, 0.05,
                              format="%.2f")
            sentiments = st.multiselect("Sentiments",
                                        ['positif', 'negatif', 'neutre'],
                                        default=['positif', 'negatif', 'neutre'])
            langues = st.multiselect("Langues",
                                     ['arabic', 'arabizi', 'french', 'english'],
                                     default=['arabic', 'arabizi', 'french', 'english'])
            st.markdown("---")
            if st.button("🔄 Nouvelle analyse"):
                for k in ['analyse_done', 'df_results', 'report_html', 'report_filename', 'df_absa']:
                    st.session_state.pop(k, None)
                st.rerun()

            st.markdown("---")
            st.markdown("### 📄 Rapport client")
            company = st.text_input("Nom du client", value="TOPNET", key="report_company")
            analyst = st.text_input("Analyste", value="",
                                    placeholder="Votre nom (optionnel)", key="report_analyst")

            if st.button("📊 Générer le rapport", use_container_width=True):
                df_res = st.session_state['df_results']
                df_f2 = df_res.copy()
                if sentiments:
                    df_f2 = df_f2[df_f2.sentiment.isin(sentiments)]
                if langues:
                    df_f2 = df_f2[df_f2.langue.isin(langues)]
                df_f2 = df_f2[df_f2.confiance >= seuil]
                with st.spinner("Génération en cours..."):
                    html_report = generate_report_html(df_f2, company, analyst)
                st.session_state['report_html'] = html_report
                fname = f"rapport_{company.lower().replace(' ', '_')}.html"
                st.session_state['report_filename'] = fname

            if 'report_html' in st.session_state:
                st.download_button(
                    "⬇️ Télécharger le rapport",
                    data=st.session_state['report_html'],
                    file_name=st.session_state.get('report_filename', 'rapport.html'),
                    mime="text/html",
                    use_container_width=True,
                    key="dl_report",
                )
                st.caption("Chrome → Ctrl+P → Enregistrer en PDF")
        else:
            seuil, sentiments, langues = 0.0, [], []

        st.markdown("---")
        if st.button("🔁 Réinitialiser le modèle", help="Utile si une erreur CUDA survient"):
            load_model.clear()
            st.rerun()
        st.caption("MARBERT fine-tuné | Groq Llama-3.3 | LIME")

    # ── Zone principale ───────────────────────────────────────────────────────
    if not st.session_state.get('analyse_done'):
        st.title("Analyse de Sentiment — Commentaires Clients Tunisiens")
        st.markdown(
            "Uploadez un fichier **CSV** ou **JSON** contenant vos commentaires clients. "
            "Le pipeline détecte la langue, traduit si nécessaire, puis prédit le sentiment "
            "avec **MARBERT** et explique les résultats avec **LIME**."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded = st.file_uploader(
                "📂 Chargez votre fichier (CSV ou JSON)",
                type=['csv', 'json', 'jsonl'],
                help="Le fichier doit contenir une colonne 'text' ou 'comment'"
            )
        with col2:
            st.markdown("**Format attendu :**")
            st.code("id,text\n1,barcha bati2 el connexion\n2,merci topnet mriguel", language='text')

        if uploaded:
            df_raw = charger_fichier(uploaded)
            if df_raw is not None:
                st.success(f"✅ {len(df_raw)} textes chargés depuis **{uploaded.name}**")
                st.dataframe(df_raw.head(5), use_container_width=True)

                if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
                    tok, mdl, device, groq_client, F, torch, detect_fn, LDE = load_model()
                    df_res = lancer_analyse(df_raw, tok, mdl, device,
                                           groq_client, F, torch, detect_fn, LDE)
                    st.session_state['df_results']  = df_res
                    st.session_state['analyse_done'] = True
                    st.rerun()
    else:
        df_res = st.session_state['df_results']
        afficher_dashboard(df_res, seuil, sentiments, langues)


if __name__ == '__main__':
    main()
