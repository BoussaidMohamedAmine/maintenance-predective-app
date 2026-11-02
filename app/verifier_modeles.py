"""Vérifie que les modèles sauvegardés, avec les prétraitements de demo_core, redonnent les résultats de test des notebooks.
Lancement : python app/verifier_modeles.py (depuis la racine du dépôt)."""
import numpy as np
from demo_core import Demo, RUL_MAX, SEUIL, ZONE_MODE, NOMS, score_nasa

d = Demo()


def dernier_cycle(type_, f):
    """Prédiction au dernier cycle observé de chaque pompe de test, RUL réelle et mode réel."""
    yh, y, pm, vm = [], [], [], []
    for p in d.pompes_test(type_).pompe:
        df, S, rv, mode = d.historique(type_, p)
        r = f(df, S)
        yh.append(r["rul"][-1]); y.append(rv[-1]); vm.append(mode)
        pm.append(NOMS[int(r["proba"][-1].argmax())] if "proba" in r else None)
    return np.array(yh), np.array(y), np.array(pm), np.array(vm)


def rapport(nom, type_, f, attendu):
    yh, y, pm, vm = dernier_cycle(type_, f)
    yc = np.minimum(y, RUL_MAX)
    r = {"RMSE": np.sqrt(np.mean((yh - yc) ** 2)), "score": score_nasa(yc, yh).sum(),
         "anticipation": (yh[y <= 30] < SEUIL).mean(), "fausses alertes": (yh[y > 60] < SEUIL).mean()}
    if pm[0] is not None:
        z = y <= ZONE_MODE
        r["mode (RUL ≤ 60)"] = (pm[z] == vm[z]).mean()
    print(f"\n{nom}")
    for k, v in r.items():
        print(f"  {k:<18} {v:9.3f}   notebook : {attendu.get(k, '-')}")


rapport("Jalon 1 · MLP k = 5, perte asymétrique (type A)", "A", lambda df, S: d.jalon1(df),
        {"RMSE": 21.04, "score": 347, "anticipation": 0.818, "fausses alertes": 0.0})
rapport("Jalon 2 · CNN + MLP enrichi (type A)", "A", lambda df, S: d.jalon2(df, S),
        {"RMSE": 22.42, "score": 1513, "mode (RUL ≤ 60)": 0.90})
rapport("Jalon 3 · transfert vers le type B", "B", lambda df, S: d.jalon3(df, S, "B"),
        {"RMSE": 30.15, "mode (RUL ≤ 60)": 1.00})
rapport("Jalon 3 · multitâche λ = 0 (type A)", "A", lambda df, S: d.jalon3(df, S, "A"), {"RMSE": 24.57})
rapport("Jalon 4 · GRU W = 60, masquage (type A)", "A", lambda df, S: d.jalon4(df),
        {"RMSE": 23.79, "score": 1452, "anticipation": 0.818, "fausses alertes": 0.0})
