"""
run_absa.py
-----------
Extrait les aspects et leur sentiment (ABSA) via Groq LLaMA-3.3.
Traite 5 commentaires par appel API (batch) pour réduire les coûts.

Usage :
    python run_absa.py --input "DATA TOPNET/topnet_all_predictions.csv"
                       --output "DATA TOPNET/topnet_all_absa.csv"
"""

import os, re, json, time, argparse
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

#os pour lire les variables d'environnement (la clé API)
#re pour expression reguliere pour extraire le JSON de la réponse
#json pour convertir texte json en dictionnaire python 
#time pour faire time.sleep() entre les appels API
#argparse pour gérer les arguments de ligne de commande
#pandas pour lire et écrire les fichiers CSV
#groq pour interagir avec l'API Groq biblio officielle 
#load_dotenv() → lit le fichier .env qui contient GROQ_API_KEY=... pour ne pas écrire la clé directement dans le code
load_dotenv()

ASPECTS = [
    'connexion', 'débit', 'service_client', 'prix',
    'coupures', 'installation', 'application', 'equipement',
]
SENTIMENTS = {'positif', 'negatif', 'neutre'}

BATCH_SIZE  = 5  #envoie 5 comm en un seul appel API pour réduire les coûts (LLaMA-3.3-70b est facturé à la token, faire un appel par comm serait plus cher)
SAVE_EVERY  = 100
GROQ_DELAY  = 0.5   # secondes entre appels pour ne pas depasser la liùite de l'api (rate limiting)

_PROMPT = """\
Tu analyses des avis clients pour une entreprise de télécommunications tunisienne.
Pour chaque commentaire numéroté, identifie les aspects mentionnés et leur sentiment.

Aspects autorisés : connexion, débit, service_client, prix, coupures, installation, application, equipement

Définitions :
- connexion    : qualité/stabilité de la connexion internet
- débit        : vitesse, bandwidth, rapidité
- service_client : accueil, support, conseillers, hotline
- prix         : tarif, abonnement, rapport qualité/prix, facture
- coupures     : pannes, interruptions, coupures de service
- installation : mise en place, intervention technicien
- application  : app mobile, espace client web
- equipement   : box, modem, routeur, câble

Commentaires :
{entries}

Réponds UNIQUEMENT avec ce JSON compact (aucun texte autour) :
{{"r":[{{"i":1,"a":[{{"n":"connexion","s":"negatif"}}]}},{{"i":2,"a":[]}}]}}

Règles :
- "n" doit être exactement l'un des aspects autorisés
- "s" doit être exactement "positif", "negatif" ou "neutre"
- Si aucun aspect clair, mets "a":[]
- Chaque commentaire doit avoir une entrée dans "r", même si "a":[]"""


def _build_prompt(texts):
    entries = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(texts))
    return _PROMPT.format(entries=entries)


def _parse_response(content, batch_size):
    """Extrait les aspects d'une réponse Groq. Retourne liste de listes."""
    results = [[] for _ in range(batch_size)]
    try:
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if not m:
            return results
        data = json.loads(m.group())
        for item in data.get('r', []):
            idx = item.get('i', 0) - 1
            if 0 <= idx < batch_size:
                for asp in item.get('a', []):
                    n = str(asp.get('n', '')).strip().lower()
                    s = str(asp.get('s', '')).strip().lower()
                    if n in ASPECTS and s in SENTIMENTS:
                        results[idx].append({'aspect': n, 'aspect_sentiment': s})
    except Exception:
        pass
    return results


def run(input_csv, output_csv):
    df = pd.read_csv(input_csv, encoding='utf-8')
    if 'id' not in df.columns:
        df.insert(0, 'id', range(1, len(df) + 1))
    n = len(df)
    print(f"\nDataset : {n} commentaires → {output_csv}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    rows_out = []
    done_ids = set()
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv, encoding='utf-8')
        rows_out = existing.to_dict('records')
        done_ids = set(existing['id'].tolist())
        print(f"  Checkpoint : {len(done_ids)} déjà traités, reprise...")

    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    pending = df[~df['id'].isin(done_ids)].reset_index(drop=True)
    print(f"  {len(pending)} commentaires à traiter\n")

    for start in range(0, len(pending), BATCH_SIZE):
        batch_df = pending.iloc[start:start + BATCH_SIZE]
        batch_texts = batch_df['text'].astype(str).tolist()

        # Appel Groq
        try:
            resp = client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': _build_prompt(batch_texts)}],
                max_tokens=400,
                temperature=0,
            )
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Erreur API] {e} — batch ignoré")
            content = '{"r":[]}'

        aspects_by_pos = _parse_response(content, len(batch_texts))

        # Construire les lignes de sortie
        for local_i, (_, row) in enumerate(batch_df.iterrows()):
            aspects = aspects_by_pos[local_i]
            base = {
                'id'               : row['id'],
                'text'             : row.get('text', ''),
                'langue'           : row.get('langue', ''),
                'sentiment_global' : row.get('sentiment', ''),
                'confiance'        : row.get('confiance', ''),
            }
            if aspects:
                for asp in aspects:
                    rows_out.append({**base, **asp})
            else:
                rows_out.append({**base, 'aspect': 'autre', 'aspect_sentiment': row.get('sentiment', 'neutre')})

        time.sleep(GROQ_DELAY)

        done_count = start + len(batch_texts)
        pct = done_count / len(pending) * 100
        print(f"  {done_count:4d}/{len(pending)} ({pct:.0f}%)", end='\r', flush=True)

        # Sauvegarde intermédiaire
        if done_count % SAVE_EVERY < BATCH_SIZE:
            pd.DataFrame(rows_out).to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n  [Checkpoint] {done_count}/{len(pending)} sauvegardés")

    # Sauvegarde finale
    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"\n\n{'='*50}")
    print(f"Sauvegardé : {output_csv}")
    print(f"Lignes      : {len(df_out)}  (long format — une ligne par aspect)")
    print(f"\nTop aspects :")
    pivot = df_out[df_out.aspect != 'autre'].groupby(['aspect','aspect_sentiment']).size().unstack(fill_value=0)
    print(pivot.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='DATA TOPNET/topnet_all_predictions.csv')
    parser.add_argument('--output', default='DATA TOPNET/topnet_all_absa.csv')
    args = parser.parse_args()
    run(args.input, args.output)
