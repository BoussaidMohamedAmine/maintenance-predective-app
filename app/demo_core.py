"""Cœur de la démonstration : chargement des modèles sauvegardés et prétraitements identiques aux notebooks
Groupe04_UC5_Jalon1 à Jalon4 (mêmes fonctions, mêmes statistiques calculées sur le train)."""
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
from pathlib import Path
import numpy as np, pandas as pd
import keras
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent         # dossier du projet : contient app, data et models
DATA, MOD = ROOT / "data", ROOT / "models"
RUL_MAX, SEUIL, ZONE_MODE = 125, 30, 60                # plafond, seuil d'alerte, zone d'évaluation du mode
CAPTEURS = [f"c{i}" for i in range(8)]
FEAT = CAPTEURS + ["charge"]
NOMS = ["sain", "roulement", "désalignement", "cavitation"]
K_J1, K_AE, W_J4 = 5, 5, 60                            # fenêtres retenues (J1 k = 5, auto-encodeur k = 5, J4 W = 60)


# ---------- données (mêmes fonctions que les notebooks)
def vers_df(D, s):
    d = pd.DataFrame(D["capteurs_" + s], columns=CAPTEURS)
    d["pompe"], d["cycle"], d["charge"] = D["pompe_" + s], D["cycle_" + s], D["charge_" + s]
    if "rul_" + s in D.files:
        d["rul"], d["mode"] = D["rul_" + s], D["mode_" + s]
    return d


def stats_regime(df):   # cycles sains (RUL > 150) du train, par régime de charge
    sains = df[df.rul > 150]
    return sains.groupby("charge")[CAPTEURS].mean(), sains.groupby("charge")[CAPTEURS].std()


def normaliser(df, stats):
    mu, sd = stats
    out = df.copy()
    out[CAPTEURS] = (df[CAPTEURS].values - mu.loc[df.charge].values) / sd.loc[df.charge].values
    return out


def fenetres_aplaties(df, k):   # J1 et auto-encodeur du J2 : k derniers cycles aplatis, 1er cycle répété au début
    v = df[FEAT].values.astype("float32")
    v = np.vstack([np.repeat(v[:1], k - 1, axis=0), v])
    return np.stack([v[i:i + k].ravel() for i in range(len(df))])


def sequences_masquees(df, W):   # J4 : W derniers cycles, zéros (masqués) avant le début de l'historique
    v = df[FEAT].values.astype("float32")
    v = np.vstack([np.zeros((W - 1, v.shape[1]), "float32"), v])
    return sliding_window_view(v, W, axis=0).transpose(0, 2, 1)


def spectro(S):   # (n, 32, 32) uint8 -> (n, 32, 32, 1) dans [0, 1]
    return np.asarray(S)[..., None].astype("float32") / 255.0


class Demo:
    """Charge les données, les statistiques de prétraitement et les modèles sauvegardés (une seule fois)."""

    def __init__(self):
        self.A, self.B = np.load(DATA / "uc5_flotteA.npz"), np.load(DATA / "uc5_flotteB.npz")
        self.TA = {s: vers_df(self.A, s) for s in ("train", "val", "test")}
        self.TB = {s: vers_df(self.B, s) for s in ("train", "test")}
        self.statsA, self.statsB = stats_regime(self.TA["train"]), stats_regime(self.TB["train"])
        charger = lambda f: keras.models.load_model(MOD / f, compile=False)   # perte asymétrique du J1 inutile en prédiction
        self.m = {"j1": charger("jalon1_final.keras"), "j2_cnn": charger("jalon2_cnn_retenu.keras"),
                  "j2_ae": charger("jalon2_autoencodeur.keras"), "j2_mlp": charger("jalon2_mlp_final.keras"),
                  "j3_rul_B": charger("jalon3_rul_B.keras"), "j3_cnn_B": charger("jalon3_cnn_mode_B.keras"),
                  "j3_multi_A": charger("jalon3_multitache_A.keras"), "j4": charger("jalon4_final.keras")}
        # indicateur de santé du J2 : référence = erreurs de reconstruction des cycles sains du train
        NA_tr = normaliser(self.TA["train"], self.statsA)
        err = np.concatenate([self._err_ae(g) for _, g in NA_tr.groupby("pompe", sort=False)])
        sain = NA_tr.rul.values > 150
        log_ref = np.log(err[sain] + 1e-8)
        self.is_mu, self.is_sd = float(log_ref.mean()), float(log_ref.std())
        self.seuil_ae = float(np.percentile(err[sain], 99))   # 99e centile des cycles sains
        self.err_seuil_is = (np.log(self.seuil_ae + 1e-8) - self.is_mu) / self.is_sd

    # ---------- exemples de test
    def pompes_test(self, type_):
        D = self.A if type_ == "A" else self.B
        return pd.DataFrame({"pompe": D["pompe_id_test"], "rul_fin": D["rul_fin_test"],
                             "mode": [NOMS[1 + m] for m in D["mode_pompe_test"]]})

    def historique(self, type_, pompe):
        """Historique brut d'une pompe de test (capteurs, charge), ses spectrogrammes et sa RUL réelle par cycle."""
        D, T = (self.A, self.TA) if type_ == "A" else (self.B, self.TB)
        sel = T["test"].pompe.values == pompe
        df = T["test"][sel].reset_index(drop=True)
        rul_fin = int(pd.Series(D["rul_fin_test"], index=D["pompe_id_test"])[pompe])
        mode = int(pd.Series(D["mode_pompe_test"], index=D["pompe_id_test"])[pompe])
        rul_vraie = rul_fin + (df.cycle.values[-1] - df.cycle.values)   # RUL réelle de chaque cycle observé
        return df, D["spectro_test"][sel], rul_vraie, NOMS[1 + mode]

    # ---------- prédictions (un historique = cycles d'une pompe, dans l'ordre)
    def _err_ae(self, df_norm):
        W = fenetres_aplaties(df_norm, K_AE)
        return ((self.m["j2_ae"].predict(W, verbose=0) - W) ** 2).mean(1)

    def _rul(self, nom, X):
        out = self.m[nom].predict(X, verbose=0, batch_size=1024)
        return RUL_MAX * (out["rul"] if isinstance(out, dict) else out).ravel()

    def jalon1(self, df):
        """MLP, fenêtre de 5 cycles aplatie, perte asymétrique ; normalisation par régime (stats A)."""
        return {"rul": self._rul("j1", fenetres_aplaties(normaliser(df, self.statsA), K_J1))}

    def jalon2(self, df, S):
        """CNN (mode), auto-encodeur (indicateur de santé), MLP de RUL enrichi : 9 entrées + IS + 4 probabilités du CNN."""
        N = normaliser(df, self.statsA)
        proba = self.m["j2_cnn"].predict(spectro(S), verbose=0)
        err = self._err_ae(N)
        IS = ((np.log(err + 1e-8) - self.is_mu) / self.is_sd).astype("float32")
        X = np.hstack([N[FEAT].values.astype("float32"), IS[:, None], proba])
        alarme = np.convolve((err > self.seuil_ae).astype(int), np.ones(3, int), "full")[:len(err)] == 3   # 3 cycles consécutifs
        return {"rul": self._rul("j2_mlp", X), "proba": proba, "IS": IS, "alarme_IS": alarme}

    def jalon3(self, df, S, type_):
        """Type B : MLP A ajusté (2 couches gelées), normalisation propre à B ; CNN ajusté sur B.
        Type A : modèle multitâche retenu (λ = 0), tête RUL seulement."""
        if type_ == "B":
            N = normaliser(df, self.statsB)
            return {"rul": self._rul("j3_rul_B", N[FEAT].values.astype("float32")),
                    "proba": self.m["j3_cnn_B"].predict(spectro(S), verbose=0)}
        return {"rul": self._rul("j3_multi_A", normaliser(df, self.statsA)[FEAT].values.astype("float32"))}

    def jalon4(self, df):
        """GRU 32 unités, fenêtre de 60 cycles, historiques courts complétés par des zéros masqués."""
        return {"rul": self._rul("j4", sequences_masquees(normaliser(df, self.statsA), W_J4))}


def score_nasa(y, yhat):
    d = np.asarray(yhat, dtype=float) - np.asarray(y, dtype=float)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def interpreter_rul(r):
    if r < SEUIL:
        return "alerte", f"RUL prédite {r:.0f} < {SEUIL} cycles : pompe signalée, planifier le remplacement."
    if r <= ZONE_MODE:
        return "surveillance", f"RUL prédite {r:.0f} cycles (entre 30 et 60) : zone tampon, surveiller la pompe."
    return "ok", f"RUL prédite {r:.0f} cycles (> 60) : aucune action, pompe loin de la panne."
