"""
run_predictions.py
------------------
Lance le pipeline MARBERT sur un CSV contenant une colonne 'text'.
Produit un CSV avec les colonnes de sentiment + probabilités.

Usage :
    python run_predictions.py --input  "DATA TOPNET/topnet_all_clean.csv"
                              --output "DATA TOPNET/topnet_all_predictions.csv"
"""

import os, re, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = './models/MARBERT-balanced-after-gen-neut'
ID2LABEL    = {0: 'negatif', 1: 'neutre', 2: 'positif'}
BATCH_SIZE  = 32          # textes envoyés à MARBERT en une fois
SAVE_EVERY  = 200         # sauvegarde intermédiaire tous les N textes
GROQ_DELAY  = 0.3         # secondes entre appels Groq (rate limit)

_TUNISIAN = {
    'ya','ma','fi','eli','elli','mel','ken','bch','mta3','weld','bent',
    'yji','nhar','kol','fel','barcha','bahi','taw','chy','shkoun','wesh',
    'sahih','famma','mafamma','aslema','marhba','yasser','ahna','entom',
    'howa','heya','heka','hedhi','houma','nchallah','inchallah','taberaka',
    'mabrouk','saha','brabi','zeda','mouch','barra','haki','kima','chnowa',
    'chniya','winou','fein','leh','sayb','tfeh','barcha','9dar','3leh',
}

_PROMPT = (
    "Traduis ce texte en arabe. "
    "Reponds uniquement avec la traduction arabe, sans explication.\n\n"
    "Texte : {text}\nTraduction arabe :"
)


# ── Chargement modèle ─────────────────────────────────────────────────────────
def load_model():
    print(f"Chargement MARBERT depuis {MODEL_PATH}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=3, id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()}
    ).to(device)
    mdl.eval()
    print(f"  Device : {device.upper()} | torch {torch.__version__}")
    return tok, mdl, device


# ── Détection langue ──────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    text = text.strip()
    if not text:
        return 'arabizi'
    arabic_chars = len(re.findall(r'[؀-ۿ]', text))
    total_alpha  = max(len(re.findall(r'\w', text)), 1)
    if arabic_chars / total_alpha > 0.25:
        return 'arabic'
    score  = len(re.findall(r'[379]', text)) * 2
    score += len(re.findall(r'[48]',  text))
    mots   = set(re.findall(r'[a-zA-Z]+', text.lower()))
    score += len(mots & _TUNISIAN) * 3
    if score >= 3:
        return 'arabizi'
    try:
        lang = detect(text)
        if lang == 'fr': return 'french'
        if lang == 'en': return 'english'
    except LangDetectException:
        pass
    return 'arabizi'


# ── Traduction Groq ───────────────────────────────────────────────────────────
def translate_to_arabic(text: str, client: Groq) -> str:
    try:
        resp = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': _PROMPT.format(text=text)}],
            max_tokens=256,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Traduction échouée : {e}]")
        return text


# ── Prédiction batch ──────────────────────────────────────────────────────────
def predict_batch(texts: list, tok, mdl, device) -> np.ndarray:
    enc = tok(
        texts, return_tensors='pt',
        padding=True, truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        logits = mdl(**enc).logits
    return F.softmax(logits, dim=-1).cpu().numpy()


# ── Pipeline principal ────────────────────────────────────────────────────────
def run(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv, encoding='utf-8')
    if 'text' not in df.columns:
        col = df.select_dtypes(include='object').columns[-1]
        df = df.rename(columns={col: 'text'})
    texts = df['text'].astype(str).str.strip().tolist()
    n = len(texts)
    print(f"\nDataset : {n} textes -> {output_csv}")

    # Reprendre là où on s'est arrêté (checkpoint)
    start_idx = 0
    results = []
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv, encoding='utf-8')
        start_idx = len(existing)
        results = existing.to_dict('records')
        print(f"  Checkpoint trouvé : reprise à partir de l'index {start_idx}")

    tok, mdl, device = load_model()
    groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    # Séparer les textes restants par langue
    pending = texts[start_idx:]
    langs   = [detect_language(t) for t in pending]

    to_translate_idx = [i for i, l in enumerate(langs) if l in ('french', 'english')]
    print(f"  Détection langue terminée.")
    print(f"  -> {langs.count('arabic')} arabes, {langs.count('arabizi')} arabizi, "
          f"{langs.count('french')} français, {langs.count('english')} anglais")
    print(f"  -> {len(to_translate_idx)} textes à traduire via Groq\n")

    # Traductions Groq
    translated = list(pending)
    for i, idx in enumerate(to_translate_idx):
        translated[idx] = translate_to_arabic(pending[idx], groq_client)
        time.sleep(GROQ_DELAY)
        if (i + 1) % 10 == 0:
            print(f"  Traductions : {i+1}/{len(to_translate_idx)}")

    # Prédictions MARBERT en batches
    print(f"\nPrédictions MARBERT (batch={BATCH_SIZE})...")
    all_probs = []
    for i in range(0, len(translated), BATCH_SIZE):
        batch = translated[i:i + BATCH_SIZE]
        probs = predict_batch(batch, tok, mdl, device)
        all_probs.append(probs)
        done = min(i + BATCH_SIZE, len(translated))
        pct  = done / len(translated) * 100
        print(f"  {done:4d}/{len(translated)} ({pct:.0f}%)", end='\r', flush=True)

    print(f"  {len(translated)}/{len(translated)} (100%) — terminé.     ")
    all_probs = np.vstack(all_probs)

    # Construire les résultats
    for i, (text, lang, trans, probs) in enumerate(
        zip(pending, langs, translated, all_probs)
    ):
        pred_id   = int(np.argmax(probs))
        sentiment = ID2LABEL[pred_id]
        confiance = float(probs[pred_id])
        results.append({
            'id'          : start_idx + i + 1,
            'text'        : text,
            'langue'      : lang,
            'texte_marbert': trans if trans != text else '',
            'sentiment'   : sentiment,
            'confiance'   : round(confiance, 4),
            'prob_neg'    : round(float(probs[0]), 4),
            'prob_neu'    : round(float(probs[1]), 4),
            'prob_pos'    : round(float(probs[2]), 4),
        })

        # Sauvegarde intermédiaire
        if (i + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8')
            print(f"  [Checkpoint] {start_idx + i + 1}/{n} sauvegardés")

    # Sauvegarde finale
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding='utf-8')

    # Résumé
    print(f"\n{'='*50}")
    print(f"Fichier sauvegardé : {output_csv}")
    print(f"Total              : {len(df_out)} prédictions")
    dist = df_out['sentiment'].value_counts()
    for label, count in dist.items():
        pct = count / len(df_out) * 100
        print(f"  {label:10s} : {count:4d}  ({pct:.1f}%)")
    print(f"Confiance moyenne  : {df_out['confiance'].mean():.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='DATA TOPNET/topnet_all_clean.csv')
    parser.add_argument('--output', default='DATA TOPNET/topnet_all_predictions.csv')
    args = parser.parse_args()
    run(args.input, args.output)
