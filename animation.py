"""
Bonus : animation Plotly d'une action exemple ou un "man-beater" bat l'homme.
=============================================================================

On rejoue, image par image, une action en couverture homme dans laquelle
Mark Andrews (le meilleur Man-Beater du classement) cree une large separation.
La separation au lacher de balle est annotee, et une ligne relie le receveur
a son defenseur le plus proche pour rendre l'espace visible.

Usage :
    python animation.py

Sortie : outputs/04_animation_man_beater.html  (interactif, lecture dans le navigateur)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

DATA_DIR = os.path.join("nfl-big-data-bowl-regional-event-data-main", "data")
TRACKING_DIR = os.path.join(DATA_DIR, "tracking")
OUT_DIR = "outputs"

STAR_NFLID = 46155          # Mark Andrews (meilleur Man-Beater)
STAR_NAME = "Mark Andrews"

FIELD_LEN = 120.0           # yards (incl. end zones)
FIELD_WID = 53.3            # yards


# --------------------------------------------------------------------------- #
# 1. Trouver la meilleure action exemple pour notre star
# --------------------------------------------------------------------------- #
def find_best_play() -> tuple[int, int, str]:
    """Cherche l'action en couverture Man ou Mark Andrews se separe le plus,
    de preference une passe complete. Retourne (gameId, playId, defensiveTeam)."""
    plays = pd.read_csv(
        os.path.join(DATA_DIR, "plays.csv"),
        usecols=["gameId", "playId", "defensiveTeam",
                 "passResult", "pff_passCoverageType"],
    )
    plays = plays[plays["pff_passCoverageType"] == "Man"]

    pff = pd.read_csv(
        os.path.join(DATA_DIR, "pffScoutingData.csv"),
        usecols=["gameId", "playId", "nflId", "pff_role"],
    )
    # Actions ou notre star court une route
    star_plays = pff[(pff["nflId"] == STAR_NFLID)
                     & (pff["pff_role"] == "Pass Route")]
    cand = star_plays.merge(plays, on=["gameId", "playId"], how="inner")
    # On privilegie les passes completees pour une belle histoire
    cand["complete"] = (cand["passResult"] == "C").astype(int)

    # On veut une belle HISTOIRE : passe completee avec une large separation.
    # On privilegie donc fortement les passes completees, et on plafonne
    # l'avantage de separation pour qu'une action completee l'emporte sur
    # une separation extreme mais incomplete.
    best = None  # (score, gameId, playId, defTeam, sep, complete)
    for _, r in cand.iterrows():
        gid, pid, defteam = int(r["gameId"]), int(r["playId"]), r["defensiveTeam"]
        f = os.path.join(TRACKING_DIR, f"tracking_{gid}.csv")
        if not os.path.exists(f):
            continue
        trk = pd.read_csv(
            f, usecols=["gameId", "playId", "nflId", "team", "event", "x", "y"])
        pf = trk[(trk["playId"] == pid) & (trk["event"] == "pass_forward")]
        if pf.empty:
            continue
        star = pf[pf["nflId"] == STAR_NFLID]
        defenders = pf[pf["team"] == defteam]
        if star.empty or defenders.empty:
            continue
        sep = cdist(star[["x", "y"]].to_numpy(),
                    defenders[["x", "y"]].to_numpy()).min()
        # Gros bonus "complete", separation plafonnee a 7 yd (au-dela, un ecart
        # de plus n'ajoute presque rien a la lisibilite de l'histoire)
        score = min(sep, 7.0) + 3.0 * r["complete"]
        if best is None or score > best[0]:
            best = (score, gid, pid, defteam, sep, int(r["complete"]))

    _, gid, pid, defteam, sep, comp = best
    print(f"Action choisie : game {gid} / play {pid} | "
          f"separation au lacher = {sep:.2f} yd | "
          f"{'complete' if comp else 'incomplete'}")
    return gid, pid, defteam


# --------------------------------------------------------------------------- #
# 2. Construire l'animation
# --------------------------------------------------------------------------- #
def build_animation(gid: int, pid: int, defteam: str) -> None:
    trk = pd.read_csv(
        os.path.join(TRACKING_DIR, f"tracking_{gid}.csv"),
        usecols=["gameId", "playId", "nflId", "frameId", "team",
                 "event", "x", "y", "playDirection"],
    )
    play = trk[trk["playId"] == pid].copy()

    # Bornes temporelles : du snap a l'arrivee/issue de la passe
    events = play.groupby("frameId")["event"].first()
    snap = events[events == "ball_snap"].index.min()
    end_candidates = events[events.isin(
        ["pass_outcome_caught", "pass_outcome_incomplete",
         "pass_arrived", "tackle"])].index
    start_f = int(snap) if pd.notna(snap) else int(play["frameId"].min())
    end_f = int(end_candidates.max()) if len(end_candidates) else int(
        play["frameId"].max())
    play = play[(play["frameId"] >= start_f) & (play["frameId"] <= end_f)]

    pf_frame = events[events == "pass_forward"].index
    pass_frame = int(pf_frame.min()) if len(pf_frame) else None

    offense_team = next(t for t in play["team"].unique()
                        if t not in (defteam, "football"))

    frames_sorted = sorted(play["frameId"].unique())

    def frame_traces(fid: int):
        g = play[play["frameId"] == fid]
        off = g[g["team"] == offense_team]
        deff = g[g["team"] == defteam]
        ball = g[g["team"] == "football"]
        star = g[g["nflId"] == STAR_NFLID]

        traces = [
            go.Scatter(x=off["x"], y=off["y"], mode="markers",
                       marker=dict(color="#1f77b4", size=12),
                       name="Attaque", showlegend=False),
            go.Scatter(x=deff["x"], y=deff["y"], mode="markers",
                       marker=dict(color="#d62728", size=12),
                       name="Defense", showlegend=False),
            go.Scatter(x=ball["x"], y=ball["y"], mode="markers",
                       marker=dict(color="saddlebrown", size=9,
                                   symbol="diamond"),
                       name="Ballon", showlegend=False),
            go.Scatter(x=star["x"], y=star["y"], mode="markers+text",
                       marker=dict(color="gold", size=16,
                                   line=dict(color="black", width=2)),
                       text=[STAR_NAME], textposition="top center",
                       name=STAR_NAME, showlegend=False),
        ]

        # Ligne vers le defenseur le plus proche de la star
        if not star.empty and not deff.empty:
            sx, sy = star[["x", "y"]].to_numpy()[0]
            dxy = deff[["x", "y"]].to_numpy()
            j = int(cdist([[sx, sy]], dxy).argmin())
            dx, dy = dxy[j]
            sep = float(np.hypot(sx - dx, sy - dy))
            traces.append(go.Scatter(
                x=[sx, dx], y=[sy, dy], mode="lines+text",
                line=dict(color="black", width=2, dash="dot"),
                text=[None, f"{sep:.1f} yd"], textposition="middle right",
                showlegend=False))
        return traces

    # Figure initiale + frames
    fig = go.Figure(
        data=frame_traces(frames_sorted[0]),
        frames=[go.Frame(data=frame_traces(fid), name=str(fid))
                for fid in frames_sorted],
    )

    # Terrain
    fig.update_xaxes(range=[0, FIELD_LEN], showgrid=False, zeroline=False,
                     title="x (yards)")
    fig.update_yaxes(range=[0, FIELD_WID], showgrid=False, zeroline=False,
                     scaleanchor="x", scaleratio=1, title="y (yards)")
    # Lignes de yard tous les 10 yards
    shapes = [dict(type="line", x0=xx, x1=xx, y0=0, y1=FIELD_WID,
                   line=dict(color="lightgreen", width=1))
              for xx in range(10, 120, 10)]
    fig.update_layout(
        shapes=shapes,
        plot_bgcolor="#2e8b57",
        title=(f"{STAR_NAME} bat l'homme — separation au lacher du ballon"
               + (f" (image {pass_frame})" if pass_frame else "")
               + "<br><sub>Or = Andrews, rouge = defense, bleu = attaque, "
                 "pointille = distance au defenseur le plus proche</sub>"),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=1.15,
            buttons=[
                dict(label="Lecture", method="animate",
                     args=[None, dict(frame=dict(duration=100, redraw=True),
                                      fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            steps=[dict(method="animate", label=str(fid),
                        args=[[str(fid)], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True))])
                   for fid in frames_sorted],
            x=0.05, len=0.9, y=0, currentvalue=dict(prefix="Image : "))],
    )

    out = os.path.join(OUT_DIR, "04_animation_man_beater.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Animation ecrite : {out}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    gid, pid, defteam = find_best_play()
    build_animation(gid, pid, defteam)


if __name__ == "__main__":
    main()
