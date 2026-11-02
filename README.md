# Maintenance prédictive des pompes — démonstration Streamlit

Étude de cas 5, Groupe 04. Application Streamlit qui charge les **modèles sauvegardés** des quatre jalons (`models/*.keras`),
applique **les mêmes prétraitements** que les notebooks et affiche la prédiction avec une interprétation :
durée de vie restante (RUL), alerte de remplacement, mode de défaillance et indicateur de santé.

**Application en ligne :** https://maintenance-predective-appgit-wh8cemq3nvahzmdjxvy8sq.streamlit.app/

## Structure du dépôt

```
├── app/
│   ├── app.py               interface Streamlit (point d'entrée)
│   ├── demo_core.py         chargement des modèles et prétraitements (fonctions reprises des notebooks)
│   ├── verifier_modeles.py  recalcule les résultats de test de chaque modèle
│   └── exemples/            pompes de test à importer : A201, A205 (type A), B606 (type B), .csv + .npy
├── data/                    uc5_flotteA.npz, uc5_flotteB.npz (données simulées)
├── models/                  modèles entraînés (.keras)
└── requirements.txt
```

`data/` sert à calculer les statistiques de normalisation (cycles sains du train, comme dans les notebooks) et fournit
les pompes de test proposées dans l'application.

## Modèles utilisés

| Jalon | Fichier(s) | Entrée | Sortie |
|---|---|---|---|
| 1 | `jalon1_final.keras` | 5 derniers cycles aplatis (45 valeurs) | RUL |
| 2 | `jalon2_cnn_retenu.keras`, `jalon2_autoencodeur.keras`, `jalon2_mlp_final.keras` | spectrogramme 32 × 32 ; 5 cycles (auto-encodeur) ; 9 entrées + indicateur + 4 probabilités | mode, indicateur de santé, RUL |
| 3 | `jalon3_rul_B.keras`, `jalon3_cnn_mode_B.keras` (type B) ; `jalon3_multitache_A.keras` (type A) | dernier cycle normalisé avec les statistiques du type ; spectrogramme | RUL, mode (type B) |
| 4 | `jalon4_final.keras` | 60 derniers cycles, zéros masqués au début | RUL |

Prétraitements communs : capteurs centrés-réduits par régime de charge avec les cycles sains (RUL > 150) du train
(du type B pour les modèles B), spectrogramme / 255, sortie × 125. Une pompe est signalée si la RUL prédite est < 30.

## Lancement local

Python 3.11 recommandé.

```bash
pip install -r requirements.txt
streamlit run app/app.py           # ouvre http://localhost:8501
python app/verifier_modeles.py     # facultatif : contrôle des résultats de test
```

## Déploiement sur Streamlit Community Cloud

1. Pousser ce dépôt sur GitHub (dépôt public, ou privé relié à votre compte Streamlit).
2. Sur [share.streamlit.io](https://share.streamlit.io) : **Create app** → choisir le dépôt, la branche `main`
   et **Main file path : `app/app.py`**.
3. Dans **Advanced settings**, choisir **Python 3.11**, puis **Deploy**.

Les dépendances sont installées depuis `requirements.txt` (racine du dépôt). Le premier démarrage prend quelques
minutes (installation de TensorFlow, puis chargement des données et des modèles, mis en cache ensuite).

## Utilisation

1. Choisir le **jalon** (et le type de pompe au jalon 3).
2. Choisir une **pompe de test** dans la liste, ou importer un `.csv` (colonnes `c0`…`c7`, `charge`, une ligne par cycle ;
   `rul` facultative) et, pour les jalons 2 et 3 (type B), le `.npy` des spectrogrammes de forme (cycles, 32, 32).
3. Déplacer le curseur **cycle observé** : le modèle ne voit que l'historique jusqu'à ce cycle, comme en exploitation.
4. Lire la RUL prédite, l'alerte (< 30 : remplacement ; 30-60 : surveillance ; > 60 : aucune action), le mode et
   l'indicateur de santé (jalon 2). L'encadré « Vérifier le modèle » recalcule les métriques sur toutes les pompes de test.

## Résultats de contrôle (`verifier_modeles.py`)

| Modèle | RMSE test | Score | Anticipation | Fausses alertes | Mode (RUL ≤ 60) |
|---|---|---|---|---|---|
| J1 MLP k = 5, perte asymétrique | 21,04 | 347 | 0,818 | 0,00 | — |
| J2 CNN + MLP enrichi | 22,42 | 1 513 | 0,818 | 0,00 | 0,90 |
| J3 type B (transfert) | 30,15 | 2 358 | 0,714 | 0,00 | 1,00 |
| J3 type A (multitâche λ = 0) | 24,57 | 2 371 | 0,818 | 0,00 | — |
| J4 GRU W = 60, masquage | 23,79 | 1 452 | 0,818 | 0,00 | — |

Identiques aux résultats de test des notebooks. Le backend Keras de l'application est TensorFlow ; les modèles des
jalons 1 à 3 ont été entraînés avec le backend PyTorch, le format `.keras` est indépendant du backend.
