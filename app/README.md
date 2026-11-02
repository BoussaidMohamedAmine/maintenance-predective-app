# Démonstration interactive — Étude de cas 5, Groupe 04

Application Streamlit qui charge les **modèles sauvegardés** des quatre notebooks (`models/*.keras`),
applique **les mêmes prétraitements** que les notebooks et affiche la prédiction avec une interprétation.

## Contenu

| Fichier | Rôle |
|---|---|
| `app.py` | interface Streamlit : choix du jalon, de l'entrée, du cycle observé ; prédiction et interprétation |
| `demo_core.py` | chargement des modèles et prétraitements (fonctions reprises des notebooks) |
| `verifier_modeles.py` | recalcule les résultats de test de chaque modèle et les compare aux notebooks |
| `exemples/` | historiques de pompes de test à importer : `pompe_A201`, `pompe_A205` (type A), `pompe_B606` (type B) — capteurs `.csv` et spectrogrammes `.npy` |

## Modèles utilisés

| Jalon | Fichier(s) | Entrée | Sortie |
|---|---|---|---|
| 1 | `jalon1_final.keras` | 5 derniers cycles aplatis (45 valeurs) | RUL |
| 2 | `jalon2_cnn_retenu.keras`, `jalon2_autoencodeur.keras`, `jalon2_mlp_final.keras` | spectrogramme 32 × 32 ; 5 cycles (auto-encodeur) ; 9 entrées + indicateur + 4 probabilités | mode, indicateur de santé, RUL |
| 3 | `jalon3_rul_B.keras`, `jalon3_cnn_mode_B.keras` (type B) ; `jalon3_multitache_A.keras` (type A) | dernier cycle normalisé avec les statistiques du type ; spectrogramme | RUL, mode (type B) |
| 4 | `jalon4_final.keras` | 60 derniers cycles, zéros masqués au début | RUL |

Prétraitements communs : capteurs centrés-réduits par régime de charge avec les cycles sains (RUL > 150) du train
(du type B pour les modèles B), spectrogramme / 255, sortie × 125. Une pompe est signalée si la RUL prédite est < 30.

## Lancement

Prérequis : l'environnement des notebooks (Python 3.11, TensorFlow 2.21, Keras 3.15, pandas, matplotlib) et `streamlit`
(`pip install streamlit`). Les données doivent être dans `UC5/data/` et les modèles dans `UC5/models/`.

```bash
cd app
python verifier_modeles.py      # facultatif : contrôle que les modèles redonnent les résultats de test
streamlit run app.py            # ouvre http://localhost:8501
```

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

Identiques aux résultats de test des notebooks. Note : le backend Keras de l'application est TensorFlow ; les modèles des
jalons 1 à 3 ont été entraînés avec le backend PyTorch, le format `.keras` est indépendant du backend.
