"""
add_aspect_comments.py
----------------------
1. Revert topnet_all_predictions.csv et topnet_all_clean.csv
   à leurs tailles originales (avant le script précédent).
2. Supprime topnet_all_absa.csv s'il existe.
3. Ajoute 500 commentaires bruts UNIQUEMENT dans topnet_all.csv.
   -> Le pipeline complet (cleaning -> MARBERT -> ABSA) les traitera.
"""

import os
import pandas as pd

BASE        = "DATA TOPNET"
ALL_CSV     = os.path.join(BASE, "topnet_all.csv")
CLEAN_CSV   = os.path.join(BASE, "topnet_all_clean.csv")
PRED_CSV    = os.path.join(BASE, "topnet_all_predictions.csv")
ABSA_CSV    = os.path.join(BASE, "topnet_all_absa.csv")

ORIG_ALL   = 2596   # taille originale topnet_all.csv
ORIG_CLEAN = 2581   # taille originale topnet_all_clean.csv
ORIG_PRED  = 2581   # taille originale topnet_all_predictions.csv

# ── 500 commentaires bruts couvrant les 8 aspects ─────────────────────────────
# Arabizi · arabe · français · anglais · code-switching
# Chaque commentaire mentionne explicitement un aspect télécom TOPNET.

COMMENTS = [
    # ══════════════════════════════ CONNEXION ══════════════════════════════
    "la connexion topnet taaba barcha, kol yom mchkla w mchi normal",
    "internet mta3i yqta3 kol sa3a, connexion instable maaich",
    "connexion tet3awwar kol layla, fi lyom wella fi lil bla fark",
    "wifi mta3 topnet yqalib 3la ro7ou, yji wyrou7 bla sabab",
    "ya topnet connexion dyalek 3aychek mchi fiha 3andi drop kol ma3aya",
    "internet mta3i m3adedch, kol ma n7awel ntfarraj 3al youtube yqta3",
    "connexion instable depuis une semaine entière, je contacte topnet mais rien",
    "la connexion est tellement mauvaise que je ne peux même pas envoyer un mail",
    "honestly topnet connection is so bad i cant even load a simple webpage",
    "internet drops every 20 minutes topnet please fix your infrastructure",
    "الاتصال عندي ما يستقرش، يقطع ويرجع كل ما شوية بدون سبب",
    "الإنترنت عندي ضعيف جداً ومش مستقر، وقف التوبنت بالكامل",
    "connexion mta3 topnet ki el barq, tji wtroh bla wa9t moharrab",
    "fi marra zadt 5 fois f nhar connexion tqta3, mchi ya3mel",
    "signaux WiFi topnet faibles même à côté du routeur, c'est inadmissible",
    "topnet connexion drops every night around 9pm, suspicious",
    "la connexion est si instable que le télétravail est impossible chez moi",
    "internet bta3 topnet yjib l3ar, connexion mafiha 7aja bahi",
    "connexion mta3na ta7at barka men 3omrha, mchi bahi hedha",
    "le réseau topnet est complètement saturé dans ma zone, connexion nulle",
    "الاتصال يقطع ويرجع ويقطع، مللت من هذا الوضع مع توبنت",
    "9abl ma nkammil ay 7aja 3al internet connexion twelli",
    "topnet réseau nul dans ma région, pas de connexion fiable depuis des mois",
    "connexion mta3i ma ta3malch fin nchghel online, taaba bezzaf",
    "internet yqta3 tawa rbe3 sa3a, topnet wakef m3aya llyoum",
    "impossible de faire du streaming en HD avec topnet connexion trop instable",
    "la qualité de la connexion topnet s'est dégradée depuis 2 mois",
    "connexion bahi w stable barcha, ma3andich ay mchkla m3a topnet",
    "topnet stable 24/7 even during peak hours, very impressed",
    "الاتصال مستقر وممتاز، ما عندنيش مشكلة مع توبنت",
    "connexion mta3na top, streaming w gaming bla ay mchkla",
    "depuis que j'ai topnet ma connexion est toujours stable, je recommande",
    "topnet best connection i've had in years, never drops",
    "connexion stable même le soir quand tout le monde est connecté",
    "wifi topnet ykhiddem bahi w stable, ma3andich chkwa",
    "connexion mta3 topnet solide kol wa9t mchi 3andha 7aja",
    "chkon 3andou info 3al stabilité connexion topnet fi zone 7ammamet?",
    "est-ce que la qualité de connexion topnet varie selon la zone géographique?",
    "quelqu'un peut m'expliquer comment améliorer la stabilité connexion topnet?",

    # ══════════════════════════════ DÉBIT ══════════════════════════════
    "débit mta3i 2mbps ghi, s3art fi pakej el plus cher topnet 50mbps",
    "speed test gives me 1mbps on a 30mbps plan, this is fraud topnet",
    "le débit est catastrophique, 5mbps au lieu de 50mbps annoncés",
    "débit 9bil ma loh ma 3loh, impossible de faire une visio en HD",
    "الديبي عندي ضعيف بزاف، الفيديو ما يحملش بدون انقطاع",
    "kol layali yibdi débit ydha3af barcha, fi lyom ok w lil mchi",
    "vitesse trop lente même pour envoyer un simple email topnet",
    "bandwidth mta3i ma yossel la yji la yrou7, download lbesa",
    "débit lent barcha, impossible shghel online wella streaming",
    "topnet speed is embarrassingly slow for the price i'm paying",
    "la vitesse descend à 2mbps le soir, c'est vraiment inacceptable",
    "3andna débit bata9a fi l3aylet, kol wa7ed ychghel yqal el bqi",
    "download speed mta3i 3mbps w3na fi 2025, shu hal khedma topnet",
    "impossible de télécharger un fichier de 1Go sans que ça prenne des heures",
    "الانترنت بطيء جداً، ما ينجمش تشتغل عليه",
    "speed drops to zero during evening hours topnet should be ashamed",
    "débit réel ne correspond jamais au débit contractuel chez topnet",
    "mta3 sa3ten bch n7ammel update ta3 jeu, débit kbira 3al wraq w ches3 fi lwaki3",
    "video calls impossible avec topnet débit trop faible et instable",
    "الديبي اللي وعدوني بيه والديبي الحقيقي فرق كبير جداً",
    "fibre topnet speed incredible, 300mbps même le soir el peak",
    "débit mriguel, download speed vachement bien depuis installation fibre",
    "الفيبر عملها واو، الديبي مريول وما في مشكلة",
    "excellent débit je streame en 4K sans aucun buffering avec topnet",
    "topnet fiber is fast and consistent, best internet i've ever had",
    "vitesse top mta3na, gaming w streaming bla lag w bla mchkla",
    "débit annoncé respecté, testé plusieurs fois avec speedtest",
    "topnet fibre delivered exactly what they promised speed wise",
    "chkon 3andou plan 50mbps, kifech el débit réel 3andou belwa9t?",
    "est-ce que topnet garantit le débit annoncé dans le contrat signé?",
    "quelqu'un a testé le débit topnet fibre 100mbps, el wa9i3 kifech?",

    # ══════════════════════════════ SERVICE CLIENT ══════════════════════════════
    "hotline topnet m3adedch, bch tossel mta3 sa3ten w khallouni f attente",
    "le service client est incompétent, j'explique mon problème 5 fois rien",
    "rang hotline 4 fois, ma7adch radd 3liya, shu hal khedma topnet",
    "the support team has no idea what they're doing, completely useless",
    "خدمة الحرفاء ما تنجم تطلبش، ساعة في الانتظار وبعدين يقطعوا عليك",
    "3amelt plainte w ma3adch 7add tassaleh fiya, khedma ta3bana barca",
    "conseiller impoli et n'a pas résolu mon problème après 1h d'attente",
    "support topnet yqoulek nwajehouk w ma7aja taythi, words w basta",
    "i've called 4 times this week and still no solution whatsoever topnet",
    "المستشار ما فهمش مشكلتي وما حلهاش، مريت ساعة بلاش",
    "khedma mta3 topnet zift, yraddou 3lik w ygooulek nwajehouk w sa7etk",
    "impossible de joindre le service client topnet, toujours occupé",
    "le délai de réponse du SAV topnet est inacceptable, 3 jours d'attente",
    "topnet customer care doesnt care at all, very frustrating experience",
    "فريق الدعم ما يحلوش المشاكل، يعطوك مواعيد ويكذبوا عليك",
    "3mel ticket w bech tjaweb 3lik topnet kol ma sabt a9allek yoma",
    "le service après-vente est désorganisé et ne tient pas ses promesses",
    "rang topnet 3 fois today w 3 fois ywajehouni le meme service",
    "support ne comprend pas le dialecte tunisien, communication difficile",
    "hotline topnet inaccessible pendant des heures, c'est inadmissible",
    "l'agent topnet kheddem professionnel, 7all el problème f 10 minutes",
    "service client sympa et efficace, merci pour votre aide vraiment",
    "topnet SAV mriguel, raddou 3liya fi wqet w 7aloulaha l mchkla",
    "خدمة الحرفاء عندهم عيشوا، حلوا المشكلة بسرعة وبإحترافية",
    "excellent customer service, problem solved in one call topnet",
    "agent était patient et compétent, a tout expliqué clairement",
    "topnet support very helpful resolved my issue within the hour",
    "l'équipe topnet est réactive et professionnelle, très satisfait",
    "khedma bahi w rapide, 7all el mchkla w 3mel followup ba3d",
    "9addart nwassel service client fi a9al men 5 da9aye9 w 7aloulaha",
    "les techniciens topnet sont très compétents, bravo à toute l'équipe",
    "fein el numéro hotline topnet pour signaler un problème urgent?",
    "c'est quoi les horaires du service client topnet, min wa9teh lwa9teh?",
    "est-ce que topnet a un chat en ligne pour le support client?",

    # ══════════════════════════════ PRIX ══════════════════════════════
    "prix mta3 topnet ghali barcha w khidma mta3ou mchi taille b3id",
    "l'abonnement coûte trop cher pour une connexion aussi médiocre topnet",
    "السعر غالي بزاف وما يستاهلش على الإطلاق مقارنة بالخدمة",
    "topnet cher et mauvais rapport qualité prix franchement déçu",
    "facture mta3i zadt b 20dt bla ma m3arrefnich, shu hedhi ya topnet",
    "prix mte3hom zyad 3la khidmethom, mchi ma3qoul hedha",
    "je paye 90dt par mois pour du 5mbps réel, c'est du vol",
    "the price is way too high for what you actually get with topnet",
    "ثمن الاشتراك غالي جداً مقارنة بالخدمة اللي نتحصل عليها",
    "tarif augmenté sans prévenir, j'ai découvert ça sur ma facture",
    "topnet raise prices every 6 months without improving anything",
    "prix mta3 topnet ghali w débit bata9a, mchi 3adel hedha",
    "l'augmentation des tarifs n'est pas justifiée vu la qualité du service",
    "اشتركت بسعر واحد وبعد شهرين لقيت الفاتورة زادت بلا ما يخبروني",
    "rapport qualité prix mta3 topnet zift, mchi ystahel el flous",
    "facture ghalia barcha w connection machi bahi, mchi 3adil",
    "prix augmente chaque année mais la qualité reste identique",
    "too expensive for what it offers, there are better deals elsewhere",
    "le coût de l'abonnement topnet n'est pas compétitif sur le marché",
    "pourquoi payer premium pour une connexion low-cost? topnet arnaque",
    "pakej mta3 topnet bech, prix ma3qoul w débit bahi, mriguel",
    "bon rapport qualité/prix je recommande topnet sans hésitation",
    "الباقة بسعر معقول وفيها كلش، راضي على الخدمة مع توبنت",
    "great value for money topnet offers the best deals in my area",
    "par rapport à la concurrence topnet reste le moins cher pour la fibre",
    "prix juste pour la qualité offerte, je suis satisfait de topnet",
    "topnet gives you the best speed per dinar in my opinion",
    "rapport qualité prix excellent chez topnet, je recommande",
    "chkon 3andou info 3al tarif pakej fibre 100mbps topnet?",
    "quels sont les différents forfaits disponibles chez topnet maintenant?",
    "est-ce qu'il y a des promotions chez topnet pour les nouveaux abonnés?",

    # ══════════════════════════════ COUPURES ══════════════════════════════
    "coupure kol yom f topnet, mchi normal w mchi ma3qoul absolument",
    "ya7koum el internet mta3i 5 fois par jour, chnowa hedha topnet",
    "انقطاع متكرر مع توبنت، كل ليلة نفس المشكل بدون أي حل",
    "interruptions tous les soirs entre 20h et 22h, j'en peux plus topnet",
    "internet mta3 topnet y9ta3 bla 7seb kol yom, mchi ma3qoul",
    "3 jours sans internet topnet ne répond pas et fait rien pour résoudre",
    "panne depuis hier soir, le technicien ne vient toujours pas topnet",
    "fi coupure kol ma yji mtar wella rih, materiel bata9a w zift",
    "الانقطاعات المتكررة خسرتني في شغلي، مش طبيعي هذا كل يوم",
    "network goes down every weekend without fail topnet unacceptable",
    "on coupe l'accès sans prévenir ni communiquer chez topnet",
    "internet yqta3 kol layla mn 11 lel 12, mahech normal hedha",
    "daily outages with topnet and no explanation given to customers",
    "je ne peux pas compter sur topnet pour le télétravail tellement ça coupe",
    "كل يوم نفس المشكلة، الإنترنت يوقف ومنبعثش، ما يصلحش هذا",
    "topnet cuts off precisely during online exams and meetings kima mza3za3",
    "une semaine entière de coupures intermittentes sans solution topnet",
    "internet tqta3 sa3ten w tramme9 wella traje3 bsa3a, haka kol yom",
    "les coupures topnet deviennent de plus en plus fréquentes et longues",
    "outage since 3 days and topnet customer service says tomorrow tomorrow",
    "zid 6 months m3a topnet, w9ta mchi fi coupure wa7da, 7amdellah",
    "service stable, rarement des coupures depuis l'installation fibre",
    "topnet reliable, ma3andich mchkla fil continuité khedma top",
    "never had an outage in over a year with topnet, very reliable",
    "منذ اشتركت في توبنت ما شفتش انقطاع، خدمة موثوقة جداً",
    "topnet fibre très stable, même lors des orages jamais de coupure",
    "uptime excellent depuis installation, ma3andich chkwa 3a coupures",
    "topnet has been incredibly reliable for my home office needs",
    "fi coupure 3andi depuis 2h, 3addi wella mchkla zone mta3na?",
    "comment signaler une coupure internet à topnet efficacement?",
    "est-ce qu'il y a des travaux maintenance topnet dans la région sfax?",

    # ══════════════════════════════ INSTALLATION ══════════════════════════════
    "technicien mta3 topnet mja khayeb, khalla les fils 3al 7it bla tarthib",
    "l'installation a pris 3 semaines au lieu de 3 jours comme promis",
    "التركيب تأخر بزاف، وعدوني بأسبوع وجاء بعد شهر كامل",
    "installateur ji tardive barcha w khidma machi professionnel 3lih",
    "the technician messed up my wiring completely, unprofessional work",
    "technicien ma7alfech el mwa3id, 3 rendez-vous ratés topnet",
    "installation bâclée, les câbles traînent partout c'est un chantier",
    "التقني جاء بدون معدات كافية واضطر يرجع مرة ثانية تضييع وقت",
    "waited all day for the topnet technician who never showed up",
    "le délai d'attente pour l'installation est de 3 semaines c'est long",
    "technicien ji w 9al ma3ndouch el materiel, rajje3 ba3d jomma",
    "installation haphazard, the cables are exposed and look dangerous",
    "3mel installation w khalla el connexion machi stable, rajje3 mouch",
    "il faut insister des dizaines de fois pour avoir un technicien topnet",
    "التقني ما احترافيش في شغله، التركيب كان فوضى كاملة",
    "installation rapide w propre, technicien professionnel mriguel barka",
    "technicien ji fi wa9tou w 3mel installation parfaite kol shay behi",
    "التركيب كان سريع ومنظم، التقني محترف ومبادر، يعيشوا توبنت",
    "très satisfait de l'installation topnet, technicien compétent et ponctuel",
    "installation done in 2 hours clean professional work very happy",
    "technicien sympa w kheddem behi, 7ass bel khidma el bahi",
    "l'équipe d'installation topnet était très professionnelle et rapide",
    "topnet installer was on time, friendly, and did excellent work",
    "installation mta3 topnet bahi w tech professionnels w mhendza",
    "khedma installation top, walu tarab rana moustarifin",
    "installation mta3 topnet tnajjem taakhod 9addech wqet fi 3adeh?",
    "quel est le délai moyen pour l'installation topnet dans la région?",
    "est-ce que topnet installe aussi les câbles à l'intérieur de l'appartement?",

    # ══════════════════════════════ APPLICATION ══════════════════════════════
    "app mta3 topnet ta7at kol ma nftah, bla fayda w mchi normal 3andi",
    "l'application plante constamment, impossible de consulter ma facture",
    "التطبيق ما يخدمش، يعلق دايماً ويغلق وحده بدون سبب",
    "topnet app is the worst app ever, crashes every 2 seconds literally",
    "espace client web mta3hom lent barcha, impossible tkhiddem fiha",
    "l'app ne fonctionne plus du tout depuis la dernière mise à jour",
    "التطبيق بطيء وما عندوش الخصائص الضرورية، ناقصه أشياء كثيرة",
    "login doesn't work password reset broken, site completely unusable",
    "application inutile, même voir la consommation est impossible",
    "topnet app keeps logging me out every time, frustrating experience",
    "l'interface de l'espace client est dépassée et pleine de bugs topnet",
    "app mta3 topnet tk7el wella t7i9 3lik bla ma ta3rfek chnowa",
    "site web topnet tombe souvent en maintenance pile quand j'en ai besoin",
    "cannot pay my bill online because the payment page always errors out",
    "التطبيق ما يوفرش كل الخدمات، يضطر تروح لوكالة",
    "3 fois 7awalat nsedded facture 3al app, kol marra error w basta",
    "l'application topnet ne permet pas de changer son forfait en ligne",
    "notifications ne fonctionnent pas sur l'app topnet android",
    "topnet website is down more than it's up, terrible web infrastructure",
    "app mta3 topnet fi ios ma ta3malch b9addech, bug everywhere",
    "app mta3 topnet mriguela, easy to navigate w tout est clair fiha",
    "l'application est bien faite, je gère tout en ligne facilement",
    "تطبيق توبنت سهل ومريح، كل شيء في مكانه ويخدم",
    "topnet app is clean and simple, easy to pay bills and check usage",
    "app bahi ynajem tbeddel pakej w tsedded facture kol fi kol",
    "le site web topnet est intuitif et rapide, très bonne expérience",
    "topnet improved their app a lot recently, much better now",
    "mfich option fil app bch nbeddel pakej mta3i wella lazem nrang?",
    "comment activer les notifications de facture sur l'app topnet?",
    "est-ce qu'on peut suspendre son abonnement via l'application topnet?",

    # ══════════════════════════════ EQUIPEMENT ══════════════════════════════
    "box mta3 topnet ta7an barcha, w7ed hit tamsekha bla ma twalli",
    "le modem surchauffe et se déconnecte toutes les heures topnet",
    "الراوتر يسخن بزاف ويوقف وحده، جودة الأجهزة رديئة جداً",
    "routeur mta3hom ta7an fi 3 mois barka, materia ma3andouch toul",
    "topnet box broke after 6 months quality is absolutely terrible",
    "le routeur fourni est de mauvaise qualité signal ne couvre pas l'appart",
    "الموديم عطل في أقل من سنة، جودة موش مقبولة على الإطلاق",
    "box resets itself every night at 3am without explanation topnet",
    "modem sakhon, w wifi mta3ou ki ma yossel 3 mètres barcha",
    "le câble fourni par topnet est de mauvaise qualité, s'abîme vite",
    "box mta3 topnet ta9der tkhaleha fil frigo 7ata tbarred, ta7na",
    "the router provided overheats constantly and needs to be restarted daily",
    "matériel topnet de mauvaise qualité, modem hors service en 8 mois",
    "الجهاز اللي عطوني إياه قديم وما يدعمش الويفي السريع",
    "wi-fi range of the topnet box is terrible, signal drops 3 meters away",
    "box tqallib 3la ro7ha barcha, mchi fiha 7aja bahi materiel zift",
    "antennes du routeur topnet trop faibles pour un appartement normal",
    "router provided by topnet cant handle multiple devices simultaneously",
    "جهاز توبنت دافى على الدوام وأخاف يشتعل، مش آمن",
    "box mta3 topnet solide, fi 3 snin wala mchkla wa7da 7amdellah",
    "le routeur est de bonne qualité, WiFi excellent dans toute la maison",
    "الراوتر ممتاز، الواي فاي يوصل لكل الغرف بدون مشكل",
    "topnet modem is solid, no issues whatsoever in 2 years of use",
    "box bahi w wifi ykhiddem bien même f ber el dar, mriguel",
    "la box topnet est moderne et performante, très content du matériel",
    "topnet equipment is high quality, router has excellent range",
    "matériel topnet top, box fibre stable et signal wifi puissant",
    "box mta3 topnet compatible m3a kol les appareils wella lazem routeur?",
    "peut-on utiliser son propre routeur avec la connexion topnet fibre?",
    "quelle est la durée de garantie de la box fournie par topnet?",

    # ══════ CONNEXION — suite ══════
    "connexion mta3 topnet mchi fiha, drop rate barcha w mchi ma3qoul",
    "wifi signal weak even next to the router, topnet do something please",
    "la connexion internet topnet est catastrophique dans toute la ville",
    "yom connexion ok yom bata9a, mchi stabilité bta3 topnet zift",
    "internet mta3i ki el thlab, fi w mafish, topnet 7all el mchkla",
    "connexion fibre mta3i sakhna barcha ma3adedch, bla sabab wa7ed",
    "drop every evening at peak hours, topnet needs better infrastructure",
    "منذ شهر وأنا أعاني من انقطاعات في الاتصال مع توبنت بدون حل",
    "impossible de faire confiance à topnet pour une connexion stable",
    "connexion mta3 topnet ki el berd, yjik w yrou7, ma3endouch stability",
    "réseau topnet saturé dans mon quartier, connexion inutilisable le soir",
    "fi wa9t el 3acha connexion topnet ta3yayet, shu hedha ya topnet",
    "the connection is unreliable and topnet refuses to acknowledge the issue",
    "connexion de haute qualité depuis l'installation topnet, très content",
    "topnet connexion stable bahia, même pendant les heures de pointe",
    "الاتصال عندي ممتاز مع توبنت، ما عنديش أي مشكلة",
    "j'ai changé pour topnet et la connexion est tellement meilleure",
    "connexion stable kol wa9t, gaming w streaming bla ay interruption",
    "topnet internet is rock solid, best decision to switch providers",
    "مش عارف ليش الناس يشكوا من توبنت، عندي الاتصال ممتاز",

    # ══════ DÉBIT — suite ══════
    "débit mta3i yedrib fi 1mbps kol ma ndrablek 3la topnet ygooulek normal",
    "vitesse de connexion topnet trop lente pour le travail à distance",
    "la nuit le débit est 10 fois meilleur, la journée topnet inutilisable",
    "speed test montre 3mbps sur un abonnement 50mbps, arnaques topnet",
    "le débit varie tellement qu'il est impossible de planifier quoi que ce soit",
    "débit zift fi wa9t el nachaa, youtube buffer 4K mchi possible",
    "internet speed degrades significantly after 8pm with topnet daily",
    "ما نقدرش نشتغل من البيت بسبب سرعة الإنترنت الضعيفة مع توبنت",
    "le débit topnet plafonne à 10mbps même avec l'offre 100mbps",
    "download speed ok upload speed zero, asymmetric problem topnet",
    "j'obtiens 200mbps comme promis, topnet fibre est au top vraiment",
    "débit constant et élevé, jamais de dégradation avec topnet fibre",
    "ايجابي، الديبي سريع ومستقر، ما شفتش مشكلة مع توبنت ابدا",
    "very fast download speeds consistently, topnet delivered on promise",
    "débit mriguel kol wa9t même fi wa9t el nachaa, topnet top",
    "ما توقعتش يكون الديبي بهيك مستوى، توبنت فاجئني بالإيجاب",

    # ══════ SERVICE CLIENT — suite ══════
    "hotline topnet toujours occupée, impossible de joindre quelqu'un",
    "j'attends un rappel depuis 3 jours, le service client disparaît",
    "khidma bta3 topnet ki el 3ma, ma trach ma tjich, bla fayda",
    "l'agent m'a raccroché au nez sans résoudre mon problème topnet",
    "ما عندهمش احترافية في التعامل مع المشاكل، يرفعوا في وجهك",
    "service client jamais disponible, on est laissés sans aide topnet",
    "support dit que c'est résolu mais le problème persiste encore",
    "topnet agents give copy paste answers without reading your complaint",
    "مستشار توبنت ما يفهمش المشكلة ويعطيك إجابات جاهزة ماعلاش",
    "on me répond toujours que le problème va être résolu, jamais fait",
    "agent topnet réactif et sympathique, problème résolu en 5 minutes",
    "service client topnet toujours disponible et très professionnel",
    "كل مرة اتصلت بخدمة الحرفاء لقيت ناس محترفة تحل المشكلة",
    "hotline topnet réactive, attente courte et solutions rapides",
    "topnet support team went above and beyond to help me resolve issue",
    "très bonne expérience avec le service client topnet, je recommande",

    # ══════ PRIX — suite ══════
    "l'abonnement topnet augmente chaque année sans amélioration réelle",
    "factura mta3 topnet zyada 3al normal, mchi bahi hedha",
    "السعر مش عادل مقارنة بما يقدمه توبنت للمشترك اليوم",
    "topnet pricing is unreasonable given the actual quality delivered",
    "paye trop cher chaque mois pour une qualité médiocre chez topnet",
    "زادوا السعر وما زادوش في جودة الخدمة، موش معقول",
    "ils augmentent les prix sans améliorer l'infrastructure topnet",
    "rapport qualité-prix mta3 topnet catastrophique fi l'wa9t el 7ali",
    "trop cher pour ce qu'on reçoit réellement avec topnet honnêtement",
    "topnet raised my bill by 15dt without any notification, unacceptable",
    "topnet reste la meilleure offre du marché pour la fibre vraiment",
    "prix abordable et service de qualité, je renouvelle chez topnet",
    "السعر معقول جداً مقارنة بالمنافسين، راضي على توبنت",
    "best price to performance ratio in the market definitely topnet",
    "rapport qualité prix imbattable avec l'offre fibre topnet actuelle",
    "topnet offers competitive pricing, well worth the subscription fee",

    # ══════ COUPURES — suite ══════
    "el internet y9ta3 kol cha3a b les 5 da9aye9, topnet shu hedha",
    "coupures quotidiennes topnet, je ne peux plus travailler de chez moi",
    "الانقطاع يصير في الأوقات المهمة دائماً، مواعيد زوم وامتحانات",
    "les pannes topnet durent plusieurs heures sans communication",
    "internet qta3 w mawjeech technicien lel 3 jem3at, topnet el7a9",
    "je perds mes réunions importantes à cause des coupures topnet",
    "ما قدرتش نسلم مشروعي بسبب الانقطاع المستمر مع توبنت",
    "topnet outages during football matches are the worst timing",
    "fi coupure kol marra nshghel fil online bcha3, topnet 9addek",
    "les coupures topnet causent des pertes financières pour mon business",
    "mafia coupure 3andi depuis un an m3a topnet, khidma top",
    "service très stable, aucune coupure notable en plusieurs mois",
    "توبنت موثوق جداً في منطقتي، ما عندي انقطاعات أبداً",
    "uptime topnet excellent, rarely any issues in residential area",
    "stable service without interruptions, topnet delivers what promised",
    "depuis que j'ai topnet je n'ai plus de coupures intempestives",

    # ══════ INSTALLATION — suite ══════
    "technicien topnet est arrivé avec 2 jours de retard sans explication",
    "installation mal faite, les câbles passent sous les portes partout",
    "التقني ما أنهاش التركيب صح، لازم يرجع مرة ثانية",
    "3 rendez-vous ratés pour l'installation topnet, inadmissible",
    "le technicien a abîmé le mur pour passer les câbles topnet",
    "installation bâclée en 20 minutes, la connexion n'est pas stable",
    "التركيب استغرق 3 أسابيع بينما وعدونا بـ3 أيام فقط",
    "technicien compétent et rapide, installation nickel du premier coup",
    "l'équipe d'installation topnet était ponctuelle et très propre",
    "ركّبوا كل شيء بشكل احترافي وسريع، يعيشوا تقنيو توبنت",
    "installation parfaite, technicien a expliqué tout le fonctionnement",
    "topnet technician installed everything perfectly within 2 hours",
    "installation propre et rapide, technicien professionnel et aimable",
    "التقني جاء في وقته وركّب كل شيء بجودة عالية وسرعة",

    # ══════ APPLICATION — suite ══════
    "l'application topnet est inutilisable depuis la nouvelle version",
    "app mta3 topnet mchi sahl khdamtou, interface 3wisa barcha",
    "التطبيق ما يخدمش على الهاتف القديم، يقع كل ما نفتحه",
    "cannot view my usage history on topnet app, always shows error",
    "le site web topnet en maintenance depuis des heures maintenant",
    "app plante systématiquement quand j'essaie de payer ma facture",
    "topnet app missing basic features that competitors have since years",
    "المنطقة الشخصية كثيراً ما تعطل ولا تقدر تدخل عليها",
    "espace client jamais à jour, les informations affichées sont fausses",
    "l'application topnet s'améliore vraiment, nouvelle version top",
    "app intuitive et rapide, je gère tout mon abonnement depuis le phone",
    "التطبيق سهل الاستعمال وفيه كل الخدمات اللي نحتاجها",
    "topnet app works great, can manage everything from my smartphone",
    "très pratique pour suivre sa consommation et payer en ligne topnet",

    # ══════ EQUIPEMENT — suite ══════
    "box topnet ta7an barcha, lazem nwaffi el mrawwa7 janbha",
    "le routeur topnet a rendu l'âme au bout de 8 mois seulement",
    "الراوتر تعطّل فجأة وأنا في منتصف اجتماع مهم جداً",
    "modem mta3 topnet bta9a barcha, wifi mchi tassel 3 mitres",
    "la qualité du matériel topnet est vraiment décevante en général",
    "box ytsakhon w ywaqqef, lazem nfa99iha kol yom mne el barrak",
    "routeur topnet fourni de mauvaise qualité, j'ai dû acheter le mien",
    "الجهاز ما يدعمش wifi 6 وهذا مشكل للأجهزة الحديثة",
    "topnet provided router barely covers my small apartment poor signal",
    "le câble ethernet fourni est court et de mauvaise qualité topnet",
    "box mta3 topnet solide, fi 4 snin wala mchkla wa7da 7amdellah",
    "excellent matériel fourni, box moderne et routeur très performant",
    "الجهاز ممتاز، الواي فاي يغطي كامل الشقة وحتى الحوش",
    "topnet equipment is top notch, router handles many devices well",
    "très content de la box topnet, moderne et facile à configurer",
    "modem topnet de qualité professionnelle, très satisfait du matériel",

    # ══════ CONNEXION — encore ══════
    "connexion topnet mchi stable, yt3awwar w ytbe3 w ytbe3",
    "fi marra 3 jours bla connexion, topnet ma3adch 7all el mchkla",
    "internet mta3i ki el yo-yo, fi w mafish, ta3yayet men topnet",
    "la connexion coupe dès qu'il y a un orage ou du vent fort topnet",
    "j'ai dû acheter une 4G en backup tellement topnet est instable",
    "the wifi drops whenever i need it most, topnet please stabilize",
    "connexion topnet instable depuis l'installation, regret total",
    "connexion fiable barcha depuis 2 ans m3a topnet, satisfaction",
    "tellement content de topnet, connexion jamais tombée en 18 mois",
    "the network is solid and reliable, topnet is worth every dinar",
    "ما بدلتش من توبنت منذ سنوات، الاتصال ممتاز دائماً",

    # ══════ DÉBIT — encore ══════
    "débit mta3i ki el 56k modem 3am 2000, mchi normal fi 2025 topnet",
    "speed catastrophique le soir, impossible de regarder une série",
    "le débit est tellement lent que les pages web mettent 30s à charger",
    "10x slower than what i pay for, topnet needs to fix this urgently",
    "débit mriguel, speedtest donne exactement ce qui est promis",
    "الديبي ممتاز وثابت، ما عندي مشكلة مع سرعة توبنت أبداً",
    "topnet fibre gives me 500mbps consistently, excellent service",
    "vitesse parfaite pour le gaming et le streaming simultané topnet",
    "débit constant et fiable, je n'ai jamais eu à me plaindre topnet",

    # ══════ SERVICE CLIENT — encore ══════
    "attente de 45 minutes pour finalement avoir quelqu'un topnet",
    "le service client ferme trop tôt, pas accessible le weekend topnet",
    "technicien topnet envoyé après 10 jours d'attente, inacceptable",
    "تركوني في الانتظار ساعة ونص وبعدين قطعوا عليا الخط",
    "support took 3 days to reply to my ticket and still no fix topnet",
    "equipe topnet très professionnelle, m'a aidé à configurer tout",
    "service impeccable, problème résolu dès le premier appel topnet",
    "خدمة احترافية جداً وسريعة، حلوا مشكلتي في نفس اليوم",
    "topnet team was incredibly helpful, exceeded my expectations",

    # ══════ PRIX — encore ══════
    "augmentation de tarif chaque année, on ne peut plus faire confiance",
    "les offres topnet sont trop chères par rapport à la concurrence",
    "facture surprise avec des frais non annoncés, topnet mchi bahi",
    "الاشتراك غالي ومش يستاهل، بديت نفكر في تغيير المزود",
    "topnet keeps charging hidden fees that were never mentioned",
    "meilleur rapport qualité-prix du marché, merci topnet vraiment",
    "prix raisonnable pour une connexion fibre de très bonne qualité",
    "السعر مقبول ويستاهل مقارنة بجودة الخدمة المقدمة",
    "best value internet provider in tunisia without a doubt topnet",

    # ══════ COUPURES — encore ══════
    "3 coupures en une journée, topnet c'est vraiment trop instable",
    "panne totale depuis 5 jours, aucune communication de topnet",
    "الانقطاع صار وقت عرض مهم للزبائن، خسرت فرصة بسبب توبنت",
    "internet cuts at the worst possible times every single day topnet",
    "10 months without a single outage, topnet is very reliable here",
    "ما شفتش انقطاع واحد منذ بداية الاشتراك، خدمة موثوقة",
    "never had an issue with connectivity, topnet is excellent here",

    # ══════ INSTALLATION — encore ══════
    "technicien topnet a cassé un carreau en passant les câbles chez moi",
    "installation prévue dans 3 jours faite après 3 semaines topnet",
    "التقني جاء وقال ما عندوش المعدات، رجع بعد أسبوع ثاني",
    "installation soignée et rapide, technicien très professionnel topnet",
    "le technicien a tout expliqué clairement et installé proprement",
    "التركيب كان ممتازاً، التقني شرح كل شيء وشغّل كل شيء",
    "installation done properly first time, no issues at all topnet",

    # ══════ APPLICATION — encore ══════
    "l'app topnet ne charge pas les factures, bug depuis des semaines",
    "impossible de me connecter à l'espace client topnet depuis hier",
    "التطبيق ما يحفظش كلمة السر، لازم تدخلها كل مرة",
    "topnet app crashes every time i open the payment section always",
    "l'application topnet est enfin fluide après la mise à jour",
    "تطبيق ممتاز ومحدّث، يخدم بسرعة وما عندوش مشاكل",
    "best telecom app in Tunisia, topnet done it right this time",

    # ══════ EQUIPEMENT — encore ══════
    "la box topnet est bruyante la nuit, ventilateur trop fort",
    "موديم توبنت سخن بزاف وخفت يشعل، خطر حقيقي",
    "routeur topnet antenna broken after 4 months poor build quality",
    "le câble fourni est trop court pour mon installation topnet",
    "box topnet robuste et fiable, aucun problème depuis 2 ans",
    "الجهاز ممتاز وعمره طويل، ما عندي أي مشكلة مع راوتر توبنت",
    "excellent build quality on topnet router, signal great everywhere",
    "la box est moderne et le signal wifi couvre parfaitement tout l'appartement",

    # ══════ BATCH FINAL — tous aspects ══════
    "connexion topnet mchi ta3et, tet3awwar kol 3acher da9aye9",
    "j'ai changé de box mais la connexion topnet reste instable encore",
    "الإنترنت عندي يوقف كل ساعة تقريباً، توبنت ما يخدمش",
    "topnet stable and fast, no complaints after 8 months of usage",
    "connexion parfaite pour le gaming online, latence excellente topnet",
    "débit réel trop éloigné du débit contractuel topnet, arnaque",
    "speed test shows half the speed promised, disappointing topnet",
    "الديبي اللي وعدوني بيه ما تحصلتش عليه في الواقع أبداً",
    "fibre topnet bluffante, jamais eu un débit aussi bon de ma vie",
    "vitesse excellente même avec 10 appareils connectés simultanément",
    "rang service client topnet 5 fois, jamais obtenu une vraie solution",
    "le temps d'attente hotline topnet est insupportable chaque fois",
    "تواصلت مع خدمة الحرفاء وما حلوا مشكلتي، أضاعوا وقتي",
    "agent topnet très compétent, m'a guidé pas à pas pour résoudre",
    "topnet support proactive, they called me before i even complained",
    "l'abonnement topnet coûte trop cher pour ce service médiocre",
    "prix exorbitant sans contrepartie qualité, topnet déçoit vraiment",
    "الفاتورة زادت بدون سبب واضح، يجب إعادة النظر في التسعير",
    "excellent rapport qualité-prix, je renouvelle sans hésitation topnet",
    "topnet pricing is fair and transparent, no hidden fees ever",
    "coupure nhar el 3id, topnet ki ykhsar mwalid ki tkhou mwajed",
    "les pannes sont de plus en plus fréquentes chez topnet ces mois",
    "انقطاع دائم وبدون إشعار مسبق، توبنت لازم يحسّن خدمته",
    "zero outages in 14 months, topnet most reliable in my area",
    "topnet service is consistent, never had an interruption at home",
    "installation technicien topnet ratée, a dû recommencer 2 fois",
    "المواعيد مع التقني دايما تتأجل، مريت أسبوعين في الانتظار",
    "installation parfaite du premier coup, technicien topnet excellent",
    "فريق التركيب جاء في الوقت المحدد وأنجز الشغل بشكل رائع",
    "l'application topnet ne permet pas de voir l'historique de consommation",
    "app buggée, impossible de changer mon mot de passe depuis des semaines",
    "التطبيق تحسّن كثيراً في النسخة الأخيرة، سريع وعملي جداً",
    "topnet app is now smooth and reliable after the latest update",
    "la box topnet chauffe tellement qu'elle s'éteint automatiquement",
    "routeur fourni trop ancien par rapport aux standards actuels topnet",
    "الجهاز ممتاز وما عندوش مشكلة، راضي على جودة المعدات",
    "topnet router handles my whole family's devices without any issue",
    "connexion souvent lente le week-end, topnet saturé vraiment",
    "internet bata9a fi 3tla, mchi ta3et w mchi zat3a topnet wakef",
    "الديبي سريع وما عندي مشكلة في جودة الاتصال مع توبنت",
    "service client topnet inaccessible le soir et le weekend complet",
    "كل ما اتصلت بخدمة الحرفاء وجدت انتظار طويل وخدمة ضعيفة",
    "prix mta3 topnet tala3 w khidma machi kamel, mchi 3adel hedha",
    "internet yqta3 kol yom exactly at 8pm, topnet wakef 3ala nerves",
    "technicien mriguel, installation propre w rapide bla ta3ab",
    "app mta3 topnet mchi sahl, lazem t3awdha kol ma tkhdem fiha",
    "box topnet robuste, fi sba3 snin wala mchkla wa7da absolument",
    "connexion fi lyom ok, fi lil mafish, topnet mchi bahi hedha",
    "débit bahi fi wa9t el s7i7, ki yji masaa yibda ydha3af barcha",
    "topnet customer service picks up fast and resolves issues well",
    "the subscription fee is reasonable considering the quality offered",
    "fi coupure kol khamis exactly, topnet ydir maintenance bla ma y3allem",
    "installation tnajjem tatakhkhar 3 weeks, topnet lazem y7assan",
    "app tombe en panne pile quand j'ai besoin de payer la facture",
    "la box topnet est silencieuse et ne chauffe jamais, excellent",
    "connexion topnet top barcha depuis fibre, ma3andich ay chkwa",
    "الانترنت سريع وثابت، توبنت يستاهل الاشتراك فيه",
    "topnet is simply the best internet provider available here",
    "mriguel topnet, connexion stable, débit bahi, prix ma3qoul",
]


def run():
    # ── 1. Lire les fichiers actuels ──────────────────────────────────────────
    df_all   = pd.read_csv(ALL_CSV,   encoding='utf-8')
    df_clean = pd.read_csv(CLEAN_CSV, encoding='utf-8')
    df_pred  = pd.read_csv(PRED_CSV,  encoding='utf-8')

    # ── 2. Remettre les fichiers à leur taille originale ──────────────────────
    df_all   = df_all.head(ORIG_ALL)
    df_clean = df_clean.head(ORIG_CLEAN)
    df_pred  = df_pred.head(ORIG_PRED)

    # ── 3. Supprimer topnet_all_absa.csv si il existe ─────────────────────────
    if os.path.exists(ABSA_CSV):
        os.remove(ABSA_CSV)
        print("topnet_all_absa.csv supprimé.")

    # ── 4. Déduplication par rapport à topnet_all.csv ─────────────────────────
    existing = set(df_all['text'].astype(str).str.strip().str.lower().tolist())
    new_texts = []
    for t in COMMENTS:
        t = t.strip()
        if t.lower() not in existing:
            existing.add(t.lower())
            new_texts.append(t)

    print(f"\n{len(new_texts)} nouveaux commentaires à ajouter (sur {len(COMMENTS)} total).")

    # ── 5. Ajouter UNIQUEMENT dans topnet_all.csv ─────────────────────────────
    col_text = 'text' if 'text' in df_all.columns else df_all.columns[0]
    new_rows = pd.DataFrame({col_text: new_texts})
    df_all_new = pd.concat([df_all, new_rows], ignore_index=True)

    # ── 6. Sauvegarder ────────────────────────────────────────────────────────
    df_all_new.to_csv(ALL_CSV,   index=False, encoding='utf-8')
    df_clean.to_csv(CLEAN_CSV,   index=False, encoding='utf-8')
    df_pred.to_csv(PRED_CSV,     index=False, encoding='utf-8')

    print(f"\nRésultat :")
    print(f"  topnet_all.csv        : {len(df_all_new):,} lignes  (+{len(new_texts)})")
    print(f"  topnet_all_clean.csv  : {len(df_clean):,} lignes  (inchangé)")
    print(f"  topnet_all_predictions: {len(df_pred):,} lignes  (inchangé)")
    print(f"\nProchain étape :")
    print(f"  1. Lancer nettoyage_avance.ipynb sur topnet_all.csv")
    print(f"  2. Lancer run_predictions.py -> topnet_all_clean.csv")
    print(f"  3. Lancer run_absa.py -> topnet_all_absa.csv")
    print(f"  OU : uploader topnet_all_clean.csv directement dans l'app")


if __name__ == '__main__':
    run()
