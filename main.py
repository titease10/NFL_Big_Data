"""
Man-Beater Score - NFL Big Data Bowl (Paris Side Quest)
=======================================================

Question football : quels receveurs (WR / TE / RB) creent de la separation
surtout contre une defense homme-a-homme (Man), et lesquels sont meilleurs
contre la zone (Zone) ?

Methode (resumee) :
1. On ne garde que les actions de passe ou la couverture PFF est "Man" ou "Zone".
2. On ne garde que les receveurs : pff_role == "Pass Route" et poste WR/TE/RB.
3. A l'instant ou le QB lache le ballon (event == "pass_forward"), on mesure
   pour chaque receveur la distance euclidienne au defenseur le plus proche
   (joueur de l'equipe en defense present sur l'image). C'est la "separation".
4. On agrege par joueur : separation moyenne vs Man, vs Zone, et le nombre
   d'actions dans chaque cas.
5. Man-Beater Score (normalise par poste). La separation est presque toujours
   plus faible contre l'homme que contre la zone (un defenseur colle le
   receveur) : le score brut sep(Man) - sep(Zone) est donc negatif pour quasi
   tout le monde et ne distingue pas le talent. On normalise donc PAR POSTE :
     z_man  = (sep_man  - moyenne_man_du_poste)  / ecart-type_man_du_poste
     z_zone = (sep_zone - moyenne_zone_du_poste) / ecart-type_zone_du_poste
     Man-Beater Score = z_man - z_zone
   > 0  => se separe mieux que ses pairs du meme poste CONTRE L'HOMME
           qu'il ne le fait contre la zone (un vrai "man-beater").
   < 0  => profil plutot "zone-beater".
   On conserve aussi le score brut (sep_man - sep_zone) pour la transparence.

Sortie : outputs/receiver_separation.csv + 3 graphiques PNG.

Usage :
    python main.py                 # pipeline complet
    python main.py --reuse-cache   # reutilise le CSV intermediaire deja extrait

Dependances : voir requirements.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")  # backend sans fenetre, pour exporter des PNG
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(
    "nfl-big-data-bowl-regional-event-data-main", "data"
)
TRACKING_DIR = os.path.join(DATA_DIR, "tracking")
OUT_DIR = "outputs"
CACHE_PATH = os.path.join(OUT_DIR, "separation_raw.csv")   # 1 ligne = 1 receveur/action
AGG_PATH = os.path.join(OUT_DIR, "receiver_separation.csv")  # 1 ligne = 1 joueur

# Seuils d'echantillon (ajustables)
MIN_TOTAL = 30      # au moins 30 actions au total
MIN_PER_TYPE = 10   # au moins 10 actions dans Man ET dans Zone

TARGET_POSITIONS = {"WR", "TE", "RB"}


# --------------------------------------------------------------------------- #
# Etape 1 : tables de reference (plays, players, pff) -> petites tables
# --------------------------------------------------------------------------- #
def load_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge plays / players / pff en ne gardant que les colonnes utiles."""
    plays = pd.read_csv(
        os.path.join(DATA_DIR, "plays.csv"),
        usecols=[
            "gameId", "playId", "possessionTeam", "defensiveTeam",
            "passResult", "pff_passCoverage", "pff_passCoverageType",
        ],
    )
    # On ne garde que Man / Zone
    plays = plays[plays["pff_passCoverageType"].isin(["Man", "Zone"])].copy()

    players = pd.read_csv(
        os.path.join(DATA_DIR, "players.csv"),
        usecols=["nflId", "displayName", "officialPosition"],
    )

    pff = pd.read_csv(
        os.path.join(DATA_DIR, "pffScoutingData.csv"),
        usecols=["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"],
    )
    return plays, players, pff


def build_route_runners(plays, players, pff) -> pd.DataFrame:
    """Table des receveurs (coureurs de route WR/TE/RB) pour les actions Man/Zone."""
    receivers = pff[pff["pff_role"] == "Pass Route"].merge(
        players, on="nflId", how="left"
    )
    receivers = receivers[receivers["officialPosition"].isin(TARGET_POSITIONS)]
    # On restreint aux actions retenues (Man/Zone) en joignant la couverture
    receivers = receivers.merge(
        plays[["gameId", "playId", "pff_passCoverageType",
               "possessionTeam", "defensiveTeam"]],
        on=["gameId", "playId"], how="inner",
    )
    return receivers


# --------------------------------------------------------------------------- #
# Etape 2 : extraction des frames pass_forward + calcul de separation
# --------------------------------------------------------------------------- #
def separation_for_game(game_file: str, plays, receivers) -> pd.DataFrame:
    """Pour un fichier tracking, calcule la separation de chaque receveur
    au defenseur le plus proche, a l'image pass_forward."""
    cols = ["gameId", "playId", "nflId", "frameId", "team", "event", "x", "y"]
    trk = pd.read_csv(game_file, usecols=cols)

    # On ne garde que l'instant du lacher de balle
    trk = trk[trk["event"] == "pass_forward"]
    if trk.empty:
        return pd.DataFrame()

    # Jointure couverture + equipe en defense (restreint aux actions Man/Zone)
    trk = trk.merge(
        plays[["gameId", "playId", "defensiveTeam", "pff_passCoverageType"]],
        on=["gameId", "playId"], how="inner",
    )
    if trk.empty:
        return pd.DataFrame()

    out_rows = []
    rec_keys = receivers[["gameId", "playId", "nflId"]].drop_duplicates()
    rec_set = set(map(tuple, rec_keys.to_numpy()))

    # On traite action par action
    for (gid, pid), grp in trk.groupby(["gameId", "playId"]):
        defense_team = grp["defensiveTeam"].iloc[0]
        cov_type = grp["pff_passCoverageType"].iloc[0]

        defenders = grp[grp["team"] == defense_team]
        if defenders.empty:
            continue
        def_xy = defenders[["x", "y"]].to_numpy()

        # Receveurs = joueurs offensifs coureurs de route presents sur l'image
        for _, row in grp.iterrows():
            key = (gid, pid, row["nflId"])
            if key not in rec_set:
                continue
            rec_xy = np.array([[row["x"], row["y"]]])
            dist = cdist(rec_xy, def_xy).min()  # defenseur le plus proche
            out_rows.append(
                (gid, pid, int(row["nflId"]), cov_type, float(dist))
            )

    return pd.DataFrame(
        out_rows,
        columns=["gameId", "playId", "nflId", "coverageType", "separation"],
    )


def extract_all_separations(plays, receivers) -> pd.DataFrame:
    """Parcourt les 122 fichiers tracking et concatene les separations."""
    files = sorted(
        os.path.join(TRACKING_DIR, f)
        for f in os.listdir(TRACKING_DIR)
        if f.endswith(".csv")
    )
    frames = []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        df = separation_for_game(f, plays, receivers)
        if not df.empty:
            frames.append(df)
        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] fichiers traites "
                  f"({time.time() - t0:.0f}s)")
    raw = pd.concat(frames, ignore_index=True)
    return raw


# --------------------------------------------------------------------------- #
# Etape 3 : agregation par joueur + Man-Beater Score
# --------------------------------------------------------------------------- #
def aggregate_players(raw: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Agrege par joueur et calcule le Man-Beater Score."""
    g = raw.groupby(["nflId", "coverageType"])["separation"].agg(["mean", "count"])
    g = g.unstack("coverageType")

    agg = pd.DataFrame(index=g.index)
    agg["sep_man"] = g[("mean", "Man")]
    agg["sep_zone"] = g[("mean", "Zone")]
    agg["n_man"] = g[("count", "Man")].fillna(0).astype(int)
    agg["n_zone"] = g[("count", "Zone")].fillna(0).astype(int)
    agg["n_total"] = agg["n_man"] + agg["n_zone"]

    # Filtre echantillon
    keep = (
        (agg["n_total"] >= MIN_TOTAL)
        & (agg["n_man"] >= MIN_PER_TYPE)
        & (agg["n_zone"] >= MIN_PER_TYPE)
    )
    agg = agg[keep].copy()

    # Score brut (garde pour transparence) : presque toujours negatif
    agg["sep_diff_raw"] = agg["sep_man"] - agg["sep_zone"]

    # On a besoin du poste pour normaliser
    agg = agg.merge(
        players.set_index("nflId")[["displayName", "officialPosition"]],
        left_index=True, right_index=True, how="left",
    )

    # Man-Beater Score normalise PAR POSTE : z(sep_man) - z(sep_zone)
    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std > 0 else s * 0.0

    agg["z_man"] = agg.groupby("officialPosition")["sep_man"].transform(_zscore)
    agg["z_zone"] = agg.groupby("officialPosition")["sep_zone"].transform(_zscore)
    agg["man_beater_score"] = agg["z_man"] - agg["z_zone"]

    agg = agg.reset_index().sort_values("man_beater_score", ascending=False)
    return agg


# --------------------------------------------------------------------------- #
# Etape 4 : visualisations
# --------------------------------------------------------------------------- #
POS_PALETTE = {"WR": "#1f77b4", "TE": "#d62728", "RB": "#2ca02c"}


def plot_scatter(agg: pd.DataFrame, path: str) -> None:
    """Nuage : X = sep vs Zone, Y = sep vs Man, diagonale, annotations."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    for pos, color in POS_PALETTE.items():
        sub = agg[agg["officialPosition"] == pos]
        ax.scatter(sub["sep_zone"], sub["sep_man"], s=60, alpha=0.75,
                   color=color, label=pos, edgecolor="white", linewidth=0.5)

    lo = min(agg["sep_zone"].min(), agg["sep_man"].min()) - 0.3
    hi = max(agg["sep_zone"].max(), agg["sep_man"].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1,
            label="Meme separation Man = Zone")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Annoter les extremes du score normalise : 5 man-beaters + 3 zone-beaters
    srt = agg.sort_values("man_beater_score", ascending=False)
    extreme = pd.concat([srt.head(5), srt.tail(3)])
    for _, r in extreme.iterrows():
        ax.annotate(r["displayName"], (r["sep_zone"], r["sep_man"]),
                    fontsize=9, weight="bold", xytext=(4, 4),
                    textcoords="offset points")

    ax.set_xlabel("Separation moyenne vs Zone (yards)")
    ax.set_ylabel("Separation moyenne vs Man (yards)")
    ax.set_title("Qui bat l'homme ? Separation au lacher de balle\n"
                 "Au-dessus de la diagonale = cree plus d'espace contre le Man",
                 fontsize=13, weight="bold")
    ax.legend(title="Poste", loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_barh(agg: pd.DataFrame, path: str) -> None:
    """Top 10 Man-Beaters en barres horizontales."""
    sns.set_theme(style="whitegrid")
    top = agg.sort_values("man_beater_score", ascending=False).head(10)
    top = top.iloc[::-1]  # pour afficher le meilleur en haut

    colors = [POS_PALETTE.get(p, "#888") for p in top["officialPosition"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["displayName"], top["man_beater_score"], color=colors)
    for y, (v, pos) in enumerate(zip(top["man_beater_score"],
                                     top["officialPosition"])):
        ax.text(v + 0.02, y, f"{v:+.2f} ({pos})", va="center", fontsize=9)

    ax.set_xlabel("Man-Beater Score (z-score par poste : z_man - z_zone)")
    ax.set_title("Top 10 Man-Beaters : meilleurs que leurs pairs de poste\n"
                 "pour se separer CONTRE L'HOMME (vs contre la zone)",
                 fontsize=13, weight="bold")
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(agg: pd.DataFrame, path: str) -> None:
    """Contexte : distribution de la separation Man vs Zone par poste."""
    sns.set_theme(style="whitegrid")
    long = agg.melt(
        id_vars=["displayName", "officialPosition"],
        value_vars=["sep_man", "sep_zone"],
        var_name="coverage", value_name="separation",
    )
    long["coverage"] = long["coverage"].map(
        {"sep_man": "Man", "sep_zone": "Zone"})
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=long, x="officialPosition", y="separation",
                hue="coverage", order=["WR", "TE", "RB"], ax=ax)
    ax.set_xlabel("Poste")
    ax.set_ylabel("Separation moyenne par joueur (yards)")
    ax.set_title("Separation Man vs Zone par poste\n"
                 "(un point = la moyenne d'un joueur)",
                 fontsize=13, weight="bold")
    ax.legend(title="Couverture")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-cache", action="store_true",
                        help="reutilise outputs/separation_raw.csv si present")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Etape 1/4 : tables de reference...")
    plays, players, pff = load_reference_tables()
    receivers = build_route_runners(plays, players, pff)
    print(f"  actions Man/Zone : {len(plays)} | "
          f"lignes receveurs (route runners) : {len(receivers)}")

    if args.reuse_cache and os.path.exists(CACHE_PATH):
        print("Etape 2/4 : lecture du cache separation_raw.csv...")
        raw = pd.read_csv(CACHE_PATH)
    else:
        print("Etape 2/4 : extraction des frames pass_forward (122 fichiers)...")
        raw = extract_all_separations(plays, receivers)
        raw.to_csv(CACHE_PATH, index=False)
    print(f"  separations calculees : {len(raw)} lignes "
          f"(joueurs distincts : {raw['nflId'].nunique()})")

    print("Etape 3/4 : agregation + Man-Beater Score...")
    agg = aggregate_players(raw, players)
    agg.to_csv(AGG_PATH, index=False)
    print(f"  joueurs retenus (apres seuils) : {len(agg)}")
    cols = ["displayName", "officialPosition", "sep_man", "sep_zone",
            "man_beater_score", "n_man", "n_zone"]
    print("\n  Top 5 Man-Beaters (score normalise par poste) :")
    print(agg[cols].head(5).to_string(index=False))
    print("\n  Bottom 5 (profils zone-beaters) :")
    print(agg[cols].tail(5).to_string(index=False))

    print("\nEtape 4/4 : graphiques...")
    plot_scatter(agg, os.path.join(OUT_DIR, "01_scatter_man_vs_zone.png"))
    plot_top_barh(agg, os.path.join(OUT_DIR, "02_top10_man_beaters.png"))
    plot_distributions(agg, os.path.join(OUT_DIR, "03_separation_by_position.png"))
    print(f"  PNG ecrits dans {OUT_DIR}/")
    print("\nTermine.")


if __name__ == "__main__":
    sys.exit(main())
