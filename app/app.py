"""Démonstration interactive, Étude de cas 5, Groupe 04 : modèles retenus des quatre jalons.
Lancement : streamlit run app/app.py (depuis la racine du dépôt)."""
import io
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from demo_core import Demo, NOMS, CAPTEURS, FEAT, RUL_MAX, SEUIL, ZONE_MODE, score_nasa, interpreter_rul

NAVY, ORANGE, GRIS = "#1f2f6b", "#e8690c", "#8a8a8a"
st.set_page_config(page_title="UC5 · Maintenance prédictive · Groupe 04", layout="wide")

MODELES = {
    "Jalon 1 · MLP (RUL)": "MLP 45 → 64 → 32 → 1, fenêtre des 5 derniers cycles aplatie, perte asymétrique, normalisation par régime.",
    "Jalon 2 · CNN + auto-encodeur + MLP enrichi": "CNN sur le spectrogramme (mode, seuil d'étiquetage 90), auto-encodeur (indicateur de santé), MLP de RUL à 14 entrées.",
    "Jalon 3 · nouveau modèle de pompe": "Type B : MLP A ajusté (2 couches gelées) avec la normalisation propre à B, CNN ajusté sur B. Type A : multitâche retenu (λ = 0).",
    "Jalon 4 · GRU (histoire de la pompe)": "GRU de 32 unités, 60 derniers cycles, historiques courts complétés par des zéros masqués.",
}


@st.cache_resource(show_spinner="Chargement des données, des statistiques de normalisation et des modèles…")
def demo():
    return Demo()


d = demo()

# ---------------- barre latérale : modèle et entrée
st.sidebar.title("Paramètres")
jalon = st.sidebar.radio("Jalon / modèle", list(MODELES))
st.sidebar.caption(MODELES[jalon])
j = int(jalon.split()[1])
type_ = st.sidebar.radio("Type de pompe", ["B", "A"], horizontal=True) if j == 3 else "A"
besoin_spectro = j == 2 or (j == 3 and type_ == "B")
source = st.sidebar.radio("Entrée", ["pompe de test (exemples fournis)", "fichier importé"])

df = S = rul_vraie = mode_vrai = None
if source.startswith("pompe"):
    tab = d.pompes_test(type_)
    p = st.sidebar.selectbox("Pompe de test", tab.pompe,
                             format_func=lambda x: f"{x} · RUL finale {tab.set_index('pompe').rul_fin[x]}")
    df, S, rul_vraie, mode_vrai = d.historique(type_, int(p))
else:
    st.sidebar.markdown(f"CSV : une ligne par cycle, dans l'ordre, colonnes `{', '.join(FEAT)}` (valeurs brutes, charge 0.6 / 0.8 / 1.0)."
                        + (" Spectrogrammes : fichier `.npy` de forme (cycles, 32, 32), valeurs 0-255." if besoin_spectro else ""))
    f_csv = st.sidebar.file_uploader("Capteurs (.csv)", type="csv")
    f_npy = st.sidebar.file_uploader("Spectrogrammes (.npy)", type="npy") if besoin_spectro else None
    if f_csv is not None:
        df = pd.read_csv(f_csv)
        manq = [c for c in FEAT if c not in df.columns]
        if manq:
            st.error(f"Colonnes manquantes : {manq}"); st.stop()
        if not set(np.round(df.charge.unique(), 2)) <= {0.6, 0.8, 1.0}:
            st.error("La charge doit valoir 0.6, 0.8 ou 1.0 (régimes appris)."); st.stop()
        df["charge"] = df.charge.astype("float32")   # même type que dans les données : clé des statistiques par régime
        if "cycle" not in df:
            df["cycle"] = np.arange(1, len(df) + 1)
        if "rul" in df:
            rul_vraie = df.rul.values
    if besoin_spectro and f_npy is not None:
        S = np.load(io.BytesIO(f_npy.getvalue()))
        if df is not None and S.shape != (len(df), 32, 32):
            st.error(f"Spectrogrammes de forme {S.shape}, attendu ({len(df)}, 32, 32)."); st.stop()

st.title("Maintenance prédictive des pompes de lavage")
st.caption("Étude de cas 5 · Groupe 04 · modèles sauvegardés des notebooks Groupe04_UC5_Jalon1 à Jalon4, mêmes prétraitements")

if df is None or (besoin_spectro and S is None):
    st.info("Choisissez une pompe de test ou importez les fichiers d'entrée dans la barre latérale. "
            "Des exemples à importer sont dans le dossier `app/exemples`.")
    st.stop()

# ---------------- cycle observé : on ne donne au modèle que l'historique jusqu'à ce cycle
n = len(df)
t = st.slider("Cycle observé (historique disponible jusqu'à ce cycle)", 1, n, n,
              help="Simule le suivi au fil des jours : le modèle ne voit que les cycles 1 à t.")
dft, St = df.iloc[:t].reset_index(drop=True), (S[:t] if S is not None else None)

with st.spinner("Prédiction…"):
    if j == 1:
        r = d.jalon1(dft)
    elif j == 2:
        r = d.jalon2(dft, St)
    elif j == 3:
        r = d.jalon3(dft, St, type_)
    else:
        r = d.jalon4(dft)

rul = float(r["rul"][-1])
niveau, texte = interpreter_rul(rul)

# ---------------- résultats
c1, c2, c3 = st.columns(3)
c1.metric("RUL prédite", f"{rul:.0f} cycles")
if rul_vraie is not None:
    vraie = int(rul_vraie[t - 1])
    c2.metric("RUL réelle (plafonnée à 125)", f"{min(vraie, RUL_MAX)} cycles", f"écart {rul - min(vraie, RUL_MAX):+.0f}", delta_color="off")
    c3.metric("Coût de l'erreur (score asymétrique)", f"{float(score_nasa(min(vraie, RUL_MAX), rul)):.2f}",
              help="e^(−d/13) − 1 si trop tôt, e^(d/10) − 1 si trop tard (d = prédit − réel)")
{"alerte": st.error, "surveillance": st.warning, "ok": st.success}[niveau](texte)

g, dr = st.columns([3, 2])
with g:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    cyc = dft.cycle.values
    if rul_vraie is not None:
        ax.plot(df.cycle.values, np.minimum(rul_vraie, RUL_MAX), color=GRIS, ls="--", lw=1.2, label="RUL réelle (plafonnée)")
    ax.plot(cyc, r["rul"], color=ORANGE, lw=1.8, label="RUL prédite")
    ax.axhline(SEUIL, color="crimson", ls=":", lw=1.2, label="seuil d'alerte (30)")
    ax.axhline(ZONE_MODE, color=NAVY, ls=":", lw=1, label="60 cycles")
    ax.scatter([cyc[-1]], [rul], color=ORANGE, zorder=5)
    ax.set_xlabel("cycle"); ax.set_ylabel("RUL (cycles)"); ax.set_ylim(0, 140); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Prédiction à chaque cycle, avec l'historique disponible à ce cycle", fontsize=10)
    st.pyplot(fig, clear_figure=True)

with dr:
    if "proba" in r:
        proba = r["proba"][-1]; k = int(proba.argmax())
        fig, ax = plt.subplots(figsize=(4.8, 2.4))
        ax.barh(NOMS, proba, color=[ORANGE if i == k else NAVY for i in range(4)])
        ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("probabilité (CNN)"); ax.grid(axis="x", alpha=.3)
        st.pyplot(fig, clear_figure=True)
        if k == 0:
            st.markdown("**Mode prédit : sain.** Aucun défaut visible dans le spectrogramme. Le mode n'est fiable qu'à partir "
                        "d'environ 40 cycles avant la panne (jalon 2, Q2) : « sain » plus tôt est attendu.")
        else:
            st.markdown(f"**Mode prédit : {NOMS[k]}** (probabilité {proba[k]:.2f}). Préparer l'intervention adaptée à ce mode.")
        if mode_vrai is not None:
            st.caption(f"Mode réel de la pompe à la panne : {mode_vrai}. Le CNN est évalué sur les cycles à RUL ≤ 60.")
        fig, ax = plt.subplots(figsize=(4.8, 2.6))
        ax.imshow(St[-1], origin="lower", aspect="auto", cmap="Blues", extent=[0, 1, 0, 500])
        ax.set_xlabel("temps (s)"); ax.set_ylabel("fréquence (Hz)"); ax.set_title(f"spectrogramme du cycle {cyc[-1]}", fontsize=9)
        st.pyplot(fig, clear_figure=True)
    elif j == 3:
        st.info("Type A : le modèle multitâche retenu a λ = 0, sa tête « mode » n'est pas entraînée. "
                "Le mode est donné par le CNN (jalon 2).")
    else:
        st.markdown("**Lecture.** La courbe orange est ce que le système aurait affiché chaque jour. "
                    "Une pompe est signalée quand la RUL prédite passe sous 30 cycles ; entre 30 et 60, zone tampon.")

if j == 2:
    st.subheader("Indicateur de santé (auto-encodeur)")
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.plot(cyc, r["IS"], color=NAVY, lw=1.2, label="indicateur de santé (erreur de reconstruction, échelle des cycles sains)")
    ax.axhline(d.err_seuil_is, color="crimson", ls=":", label="seuil (99e centile des cycles sains)")
    if r["alarme_IS"].any():
        c0 = cyc[int(np.argmax(r["alarme_IS"]))]
        ax.axvline(c0, color=ORANGE, lw=1.5, label=f"détection (3 cycles consécutifs au-dessus) : cycle {c0}")
    ax.set_xlabel("cycle"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)
    st.caption("L'auto-encodeur a appris à reconstruire des fenêtres de 5 cycles sains : quand la pompe se dégrade, il reconstruit mal "
               "et l'indicateur monte. Sur la validation : détection 56 cycles avant la panne en médiane, 2 détections prématurées sur 15.")

ENTREES = {1: "Fenêtre des 5 derniers cycles aplatie (45 entrées), premier cycle répété au début de l'historique.",
           2: "Spectrogramme / 255 ; auto-encodeur sur 5 cycles ; MLP : 9 entrées + indicateur de santé + 4 probabilités du CNN.",
           3: "Dernier cycle (9 entrées) ; spectrogramme / 255 pour le CNN.",
           4: "Fenêtre des 60 derniers cycles (60 × 9) ; zéros masqués avant le début de l'historique."}
stats_txt = "B (statistiques propres à B)" if (j == 3 and type_ == "B") else "A"
with st.expander("Prétraitements appliqués (identiques aux notebooks)"):
    st.markdown(f"""
- Capteurs centrés-réduits **par régime de charge** avec la moyenne et l'écart-type des cycles sains (RUL > 150) du train
  du type {stats_txt} ; la charge reste une entrée.
- {ENTREES[j]}
- Sortie sigmoïde × 125 = RUL en cycles (plafonnée à 125). Alerte si RUL prédite < {SEUIL}.""")
    st.dataframe(dft.tail(5)[["cycle"] + FEAT], hide_index=True)

# ---------------- vérification sur tout le test
with st.expander("Vérifier le modèle sur toutes les pompes de test"):
    if st.button("Lancer la vérification"):
        lignes = []
        for p_ in d.pompes_test(type_).pompe:
            df_, S_, rv, mv = d.historique(type_, int(p_))
            rr = d.jalon1(df_) if j == 1 else d.jalon2(df_, S_) if j == 2 else d.jalon3(df_, S_, type_) if j == 3 else d.jalon4(df_)
            lignes.append(dict(pompe=int(p_), rul_reelle=int(rv[-1]), rul_predite=float(rr["rul"][-1]), mode_reel=mv,
                               mode_predit=NOMS[int(rr["proba"][-1].argmax())] if "proba" in rr else "—"))
        res = pd.DataFrame(lignes)
        y, yh = np.minimum(res.rul_reelle.values, RUL_MAX), res.rul_predite.values
        m = {"RMSE": np.sqrt(np.mean((yh - y) ** 2)), "score (somme)": score_nasa(y, yh).sum(),
             "anticipation": (yh[res.rul_reelle <= 30] < SEUIL).mean(), "fausses alertes": (yh[res.rul_reelle > 60] < SEUIL).mean()}
        if "proba" in r:
            z = res.rul_reelle <= ZONE_MODE
            m["mode (RUL ≤ 60)"] = (res.mode_predit[z] == res.mode_reel[z]).mean()
        st.dataframe(pd.DataFrame({"test": {k: round(float(v), 3) for k, v in m.items()}}).T)
        st.dataframe(res.round({"rul_predite": 1}), hide_index=True)
