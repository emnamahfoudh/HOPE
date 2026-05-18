"""
app.py — Application Streamlit Analyse de Sentiment Tunisien
============================================================
Upload un fichier CSV/JSON → nettoyage → MARBERT → dashboard interactif
"""

import os, re, time, io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    for b in range(0, len(mar_txts), BATCH):
        batch = mar_txts[b:b+BATCH]
        enc   = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits
        all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
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
    """Retourne les n mots les plus fréquents pour un sentiment donné."""
    textes = df_sent[df_sent.sentiment == sentiment]['text'].tolist()
    words  = []
    for t in textes:
        for w in re.findall(r'[a-zA-Z؀-ۿ]{3,}', t.lower()):
            if w not in _STOPWORDS and len(w) >= 3:
                words.append(w)
    counts = Counter(words).most_common(n)
    return pd.DataFrame(counts, columns=['mot', 'freq'])


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

    page = st.radio("", ["📈 Vue Globale", "🔍 Analyse Détaillée", "🔬 LIME Explorer", "📁 Données"],
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

    # ── PAGE 2 : Analyse Détaillée ────────────────────────────────────────────
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
                for k in ['analyse_done', 'df_results']:
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            seuil, sentiments, langues = 0.0, [], []

        st.markdown("---")
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
