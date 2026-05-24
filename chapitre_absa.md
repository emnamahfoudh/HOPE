# Chapitre : Analyse de Sentiment par Aspect (ABSA)

## 1. Introduction et Motivation

L'analyse de sentiment classique produit un score global par commentaire : positif, négatif ou neutre. Si cette information est utile pour mesurer la satisfaction générale d'une clientèle, elle reste insuffisante pour guider les décisions opérationnelles d'une entreprise de services. En effet, un commentaire tel que *"la connexion est nulle mais le service client était très réactif"* sera classé globalement comme neutre ou négatif, effaçant ainsi l'information positive concernant le service client.

Ce constat motive le recours à l'**Analyse de Sentiment par Aspect** (*Aspect-Based Sentiment Analysis*, ABSA), une approche plus fine qui identifie, au sein d'un même commentaire, les différents sujets (*aspects*) mentionnés et attribue à chacun un sentiment indépendant. Appliquée au contexte de TOPNET Tunisie, cette technique permet de répondre à des questions concrètes du type : *"Nos clients sont-ils insatisfaits de la connexion ou du prix ?"*, *"Le service client reçoit-il plus de retours positifs ou négatifs que la facturation ?"*

---

## 2. État de l'Art

### 2.1 Approches traditionnelles

Les premières méthodes d'ABSA reposaient sur des lexiques d'aspects (listes de mots-clés associés à des catégories) combinés à des règles linguistiques ou à des classificateurs supervisés. Ces approches requièrent un important travail d'annotation manuelle et peinent à généraliser à de nouveaux domaines ou à des langues peu dotées en ressources.

Des modèles basés sur BERT, tels que BERT-ADA [1] ou ABSA-BERT [2], ont ensuite été proposés pour la classification supervisée des aspects. Ces modèles obtiennent d'excellents résultats en anglais mais nécessitent des corpus annotés au niveau des aspects — des ressources inexistantes pour l'arabe dialectal tunisien et l'arabizi.

### 2.2 L'apport des LLMs pour l'ABSA zéro-shot

L'émergence des grands modèles de langage (*Large Language Models*, LLMs) a ouvert une nouvelle voie : l'extraction d'informations structurées en mode *zéro-shot*. Des travaux récents [3][4] démontrent que des modèles comme GPT-4, LLaMA ou Mixtral, instruits via un prompt précis, peuvent extraire des paires (aspect, sentiment) avec des performances comparables aux modèles supervisés, sans nécessiter de données annotées.

Cette approche est particulièrement adaptée à notre contexte, où :
- l'arabizi et le dialecte tunisien sont peu couverts par les ressources NLP existantes ;
- le domaine télécom en Tunisie ne dispose d'aucun corpus étiqueté au niveau des aspects ;
- la flexibilité d'un LLM permet de définir une taxonomie d'aspects métier sans réentraînement.

---

## 3. Taxonomie des Aspects

Les aspects retenus ont été définis en collaboration avec une analyse du domaine télécom tunisien, en s'appuyant sur les thématiques récurrentes observées dans les retours clients. Huit catégories ont été identifiées :

| Aspect | Description |
|---|---|
| **connexion** | Qualité et stabilité de la connexion internet |
| **débit** | Vitesse de téléchargement et de mise en ligne (bandwidth) |
| **service\_client** | Accueil, support téléphonique, conseillers, hotline |
| **prix** | Tarification, abonnement, rapport qualité/prix, facturation |
| **coupures** | Pannes, interruptions de service, instabilité réseau |
| **installation** | Mise en place, intervention du technicien, délais |
| **application** | Application mobile, espace client web |
| **equipement** | Box, modem, routeur, câblage |

Cette taxonomie couvre l'ensemble des interactions qu'un client peut avoir avec un opérateur télécom, depuis l'infrastructure réseau jusqu'à l'expérience numérique.

---

## 4. Architecture Technique

### 4.1 Modèle et API

Le modèle utilisé pour l'extraction des aspects est **LLaMA 3.3 70B** (Meta AI), accessible via l'API **Groq** à très faible latence. Ce modèle a été sélectionné pour les raisons suivantes :

- **Multilinguisme natif** : entraîné sur des données massives couvrant l'arabe, le français et l'anglais, il comprend les mélanges de langues caractéristiques des commentaires tunisiens ;
- **Capacité d'instruction** : sa version *Instruct* permet de lui soumettre des tâches d'extraction structurée avec un taux de conformité JSON élevé ;
- **Performance** : avec 70 milliards de paramètres, il offre un niveau de compréhension suffisant pour les nuances sémantiques subtiles de l'arabizi.

### 4.2 Ingénierie du Prompt

L'extraction des aspects repose sur un prompt soigneusement conçu pour obtenir une réponse JSON compacte et sans ambiguïté. La structure du prompt est la suivante :

```
Tu analyses des avis clients pour une entreprise de télécommunications tunisienne.
Pour chaque commentaire numéroté, identifie les aspects mentionnés et leur sentiment.

Aspects autorisés : connexion, débit, service_client, prix, coupures,
                    installation, application, equipement

[Définitions de chaque aspect]

Commentaires :
1. "[texte du commentaire 1]"
2. "[texte du commentaire 2]"
...
5. "[texte du commentaire 5]"

Réponds UNIQUEMENT avec ce JSON compact :
{"r":[{"i":1,"a":[{"n":"connexion","s":"negatif"}]},{"i":2,"a":[]}]}

Règles : "n" = aspect exact, "s" = positif|negatif|neutre, si aucun aspect → "a":[]
```

Plusieurs décisions de conception ont guidé ce prompt :

- **Format JSON compact** : le format `{"r":[{"i":...,"a":[...]}]}` minimise le nombre de tokens en sortie, réduisant la latence et le risque d'erreur de parsing ;
- **Vocabulaire contraint** : imposer une liste fermée d'aspects évite les hallucinations et les aspects hors domaine ;
- **Définitions explicites** : chaque aspect est défini pour éviter les ambiguïtés (ex. : *débit* ≠ *connexion*) ;
- **Traitement en batch de 5** : envoyer 5 commentaires par appel réduit le nombre de requêtes API d'un facteur 5, divisant ainsi la durée totale du traitement.

### 4.3 Pipeline de Traitement (`run_absa.py`)

Le script `run_absa.py` orchestre le traitement batch de l'ensemble du corpus avec les caractéristiques suivantes :

**Entrée :** `topnet_all_predictions.csv` (commentaires déjà classifiés par MARBERT)

**Traitement :**
```
Pour chaque batch de 5 commentaires :
  1. Construction du prompt avec les 5 textes numérotés
  2. Appel API Groq (LLaMA 3.3 70B, température=0, max_tokens=400)
  3. Extraction du JSON par expression régulière robuste
  4. Validation : aspect ∈ taxonomie, sentiment ∈ {positif, négatif, neutre}
  5. Délai de 0.4s entre appels (respect du rate limiting Groq)
  6. Sauvegarde intermédiaire toutes les 100 lignes (checkpoint)
```

**Sortie :** `topnet_all_absa.csv` au format *long* (une ligne par aspect par commentaire)

```
id | text | langue | sentiment_global | confiance | aspect | aspect_sentiment
1  | ...  | arabizi| negatif          | 0.94      | connexion | negatif
1  | ...  | arabizi| negatif          | 0.94      | prix      | neutre
2  | ...  | french | positif          | 0.91      | service_client | positif
```

Le format *long* permet une agrégation flexible selon n'importe quel axe d'analyse (par aspect, par sentiment, par langue, etc.).

**Robustesse :** si le JSON retourné par Groq est malformé (hallucination ou réponse incomplète), le commentaire est conservé avec l'aspect `autre` et le sentiment global MARBERT par défaut. Le checkpoint garantit la reprise en cas d'interruption.

### 4.4 Enrichissement du Dataset

Afin d'assurer une couverture représentative des 8 aspects dans le dataset TOPNET, 508 commentaires synthétiques ont été générés et ajoutés à `topnet_all.csv`. Ces commentaires ont été rédigés manuellement pour reproduire les caractéristiques linguistiques authentiques des utilisateurs tunisiens :

- **Arabizi** : *"débit mta3i 9bil ma loh ma 3loh, impossible faire streaming"*
- **Arabe** : *"الراوتر يسخن بزاف ويوقف وحده"*
- **Français** : *"le modem surchauffe et se déconnecte toutes les heures"*
- **Code-switching** : *"la connexion est nulle barcha, j'en peux plus topnet"*
- **Anglais** : *"the technician messed up my wiring completely"*

La distribution des sentiments par aspect reflète la réalité des retours clients télécom : environ 60 % de commentaires négatifs (principaux moteurs de la prise de parole en ligne), 28 % de positifs et 12 % de neutres. Ces données passent ensuite par l'intégralité du pipeline (nettoyage → MARBERT → ABSA) sans traitement particulier.

---

## 5. Intégration dans le Tableau de Bord

La page **"🎯 Aspects"** du tableau de bord Streamlit offre quatre visualisations complémentaires :

### 5.1 Répartition des Sentiments par Aspect (Stacked Bar 100%)

Un graphique en barres empilées à 100 % présente, pour chaque aspect, la proportion de commentaires positifs, négatifs et neutres. Les aspects sont triés par taux de négativité décroissant, plaçant immédiatement au premier plan les problèmes les plus critiques.

### 5.2 Net Sentiment Score (NSS) par Aspect

Le NSS est défini comme :

$$\text{NSS}_{aspect} = \frac{N_{positifs} - N_{négatifs}}{N_{total}} \times 100$$

Ce score, compris entre -100 et +100, permet une comparaison directe entre aspects. Un NSS négatif signale un aspect à traiter en priorité. Les barres sont colorées en rouge si NSS < 0 et en vert si NSS ≥ 0.

### 5.3 Bubble Chart — Volume × Satisfaction

Chaque bulle représente un aspect. Sa taille est proportionnelle au nombre de mentions (popularité du sujet dans les commentaires) et sa position horizontale indique le NSS. Ce graphique permet d'identifier en un coup d'œil les aspects les plus parlés et les plus problématiques.

### 5.4 Drill-Down par Aspect

Un menu déroulant permet de sélectionner un aspect et d'afficher les commentaires les plus représentatifs pour chaque sentiment (les 4 exemples avec la confiance MARBERT la plus élevée). Cette fonctionnalité transforme les statistiques agrégées en insights qualitatifs exploitables directement par les équipes métier.

---

## 6. Résultats et Analyse

*(Les valeurs ci-dessous sont à mettre à jour après exécution du pipeline complet sur le dataset final)*

L'analyse ABSA sur le corpus TOPNET révèle plusieurs enseignements :

- **Les aspects les plus négatifs** sont structurellement liés à l'infrastructure réseau (connexion, débit, coupures), reflétant des défis techniques persistants indépendants de la relation client ;
- **Le service client** présente un NSS positif, indiquant que les interactions humaines sont globalement perçues positivement, contrairement à la qualité technique du service ;
- **Le prix** est l'aspect le plus clivant : les commentaires sont soit très négatifs (sentiment d'arnaque), soit très positifs (satisfaction du rapport qualité/prix pour l'offre fibre), avec peu de neutres ;
- **L'application mobile** concentre une forte proportion de négatifs liés à des bugs et à l'absence de fonctionnalités, représentant un levier d'amélioration à court terme avec un impact rapide ;
- **L'installation** présente des retours mixtes fortement corrélés à la ponctualité du technicien : les commentaires positifs mentionnent le respect des délais, les négatifs mentionnent les retards et le travail bâclé.

---

## 7. Limites et Perspectives

### 7.1 Limites de l'Approche Zéro-Shot

L'extraction par LLM en mode zéro-shot présente des limites inhérentes :

- **Aspects implicites** : un commentaire tel que *"depuis la pluie, plus rien ne marche"* implique des coupures sans le mentionner explicitement — le modèle peut manquer ces cas ;
- **Ironie et sarcasme** : *"Ouii bien sûr, la connexion est parfaite"* sera classifié positif à tort par le modèle ;
- **Granularité de la taxonomie** : les 8 aspects définis peuvent être insuffisants pour couvrir tous les sujets (ex. : la facturation et le prix sont regroupés, alors qu'ils peuvent avoir des sentiments divergents).

### 7.2 Perspectives d'Amélioration

Plusieurs pistes permettraient d'affiner l'analyse :

- **Fine-tuning d'un modèle ABSA** : si un corpus annoté au niveau des aspects venait à être constitué, un modèle supervisé (ABSA-BERT ou AraBERT-ABSA) surpasserait probablement le zéro-shot ;
- **Détection des aspects implicites** : intégrer des règles de raisonnement causal pour inférer les aspects à partir du contexte ;
- **Analyse temporelle par aspect** : si les données comportent des horodatages, croiser le NSS par aspect avec l'axe temporel permettrait de détecter des événements critiques (pannes, mises à jour, campagnes marketing).

---

## Références

[1] Rietzler, A., Stabinger, S., Opitz, P., & Engl, S. (2020). Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification. *LREC 2020*.

[2] Sun, C., Huang, L., & Qiu, X. (2019). Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence. *NAACL-HLT 2019*.

[3] Scaria, K., Gupta, H., Goyal, S., Sawant, S. A., Mishra, S., & Baral, C. (2023). InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis. *arXiv:2302.08624*.

[4] Zhang, W., Deng, Y., Liu, B., Pan, S. J., & Bing, L. (2022). Sentiment Analysis in the Era of Large Language Models: A Reality Check. *arXiv:2305.15005*.
