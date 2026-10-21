"""
Genere une page web autonome (index.html) pour visualiser les resultats :
insight, graphiques, tableau des top/bottom joueurs et animation integree.

Usage : python build_page.py  (a lancer apres main.py et animation.py)
Sortie : index.html a la racine (compatible GitHub Pages).
"""

import os
import pandas as pd

AGG_PATH = os.path.join("outputs", "receiver_separation.csv")
OUT = "index.html"

POS_COLORS = {"WR": "#1f77b4", "TE": "#d62728", "RB": "#2ca02c"}


def table_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        pos = r["officialPosition"]
        color = POS_COLORS.get(pos, "#888")
        rows.append(
            f"<tr><td>{r['displayName']}</td>"
            f"<td><span class='pos' style='background:{color}'>{pos}</span></td>"
            f"<td>{r['sep_man']:.2f}</td>"
            f"<td>{r['sep_zone']:.2f}</td>"
            f"<td class='score'>{r['man_beater_score']:+.2f}</td>"
            f"<td>{int(r['n_man'])}</td>"
            f"<td>{int(r['n_zone'])}</td></tr>"
        )
    return "\n".join(rows)


def main() -> None:
    agg = pd.read_csv(AGG_PATH).sort_values("man_beater_score", ascending=False)
    n_players = len(agg)
    top_rows = table_rows(agg.head(10))
    bot_rows = table_rows(agg.tail(5).iloc[::-1])

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Man-Beater Score — NFL Big Data Bowl</title>
<style>
  :root {{ --bg:#0d1117; --card:#161b22; --line:#30363d; --txt:#e6edf3;
           --muted:#8b949e; --accent:#f0b429; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
          background:var(--bg); color:var(--txt); line-height:1.6; }}
  header {{ padding:48px 24px; text-align:center;
            background:linear-gradient(135deg,#013369 0%,#0d1117 70%);
            border-bottom:1px solid var(--line); }}
  header h1 {{ margin:0 0 8px; font-size:2.2rem; }}
  header p {{ color:var(--muted); margin:4px 0; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:24px; }}
  .card {{ background:var(--card); border:1px solid var(--line);
           border-radius:12px; padding:24px; margin:24px 0; }}
  .card h2 {{ margin-top:0; border-left:4px solid var(--accent);
              padding-left:12px; }}
  .insight {{ font-size:1.1rem; }}
  .insight b {{ color:var(--accent); }}
  img {{ width:100%; border-radius:8px; display:block; }}
  figcaption {{ color:var(--muted); font-size:.9rem; margin-top:8px;
                text-align:center; }}
  table {{ width:100%; border-collapse:collapse; font-size:.95rem; }}
  th,td {{ padding:8px 10px; text-align:center; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  td:first-child, th:first-child {{ text-align:left; }}
  .score {{ font-weight:700; color:var(--accent); }}
  .pos {{ display:inline-block; min-width:34px; padding:1px 8px;
          border-radius:10px; color:#fff; font-size:.8rem; font-weight:600; }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px;
           text-align:center; }}
  .grid .num {{ font-size:1.8rem; font-weight:700; color:var(--accent); }}
  .grid .lbl {{ color:var(--muted); font-size:.85rem; }}
  iframe {{ width:100%; height:640px; border:1px solid var(--line);
            border-radius:8px; background:#fff; }}
  footer {{ text-align:center; color:var(--muted); padding:32px;
            font-size:.85rem; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>🏈 Man-Beater Score</h1>
  <p>NFL Big Data Bowl — Paris Side Quest · Saison 2021, semaines 1-8</p>
  <p>Quels receveurs (WR / TE / RB) se séparent le mieux contre l'homme ?</p>
</header>

<div class="wrap">

  <div class="card insight">
    <h2>L'insight</h2>
    <p>La séparation d'un receveur chute pour <b>presque tout le monde</b> face à
    une couverture homme-à-homme (~2.1 yd) comparée à la zone (~3.7 yd). Mais
    certains joueurs encaissent cette compression bien mieux que leurs pairs de
    poste. Le <b>Man-Beater Score</b> les identifie : c'est le z-score (par poste)
    de la séparation contre l'homme, moins celui contre la zone. En tête,
    <b>Mark Andrews</b>, confirmant sa réputation de cible fiable en man.</p>
  </div>

  <div class="card">
    <div class="grid">
      <div><div class="num">{n_players}</div><div class="lbl">joueurs analysés</div></div>
      <div><div class="num">122</div><div class="lbl">matchs</div></div>
      <div><div class="num">pass_forward</div><div class="lbl">instant de mesure</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Qui bat l'homme ?</h2>
    <figure>
      <img src="outputs/01_scatter_man_vs_zone.png" alt="Séparation Man vs Zone">
      <figcaption>Chaque point = un joueur. Au-dessus de la diagonale = crée plus
      d'espace contre le Man que contre la Zone.</figcaption>
    </figure>
  </div>

  <div class="card">
    <h2>Top 10 des Man-Beaters</h2>
    <table>
      <tr><th>Joueur</th><th>Poste</th><th>Sép. Man</th><th>Sép. Zone</th>
          <th>Score</th><th>n Man</th><th>n Zone</th></tr>
      {top_rows}
    </table>
    <figure style="margin-top:20px">
      <img src="outputs/02_top10_man_beaters.png" alt="Top 10 Man-Beaters">
    </figure>
  </div>

  <div class="card">
    <h2>À l'opposé : les profils "zone-beaters"</h2>
    <table>
      <tr><th>Joueur</th><th>Poste</th><th>Sép. Man</th><th>Sép. Zone</th>
          <th>Score</th><th>n Man</th><th>n Zone</th></tr>
      {bot_rows}
    </table>
  </div>

  <div class="card">
    <h2>Séparation par poste</h2>
    <figure>
      <img src="outputs/03_separation_by_position.png" alt="Séparation par poste">
      <figcaption>Les RB ont une séparation absolue plus élevée (souvent couverts
      par des linebackers), d'où la normalisation par poste.</figcaption>
    </figure>
  </div>

  <div class="card">
    <h2>🎬 Animation : Mark Andrews bat l'homme</h2>
    <p>Une passe complétée en couverture homme. Or = Andrews, rouge = défense,
    bleu = attaque ; le trait pointillé mesure la distance au défenseur le plus
    proche, image par image. Cliquez sur <b>Lecture</b>.</p>
    <iframe src="outputs/04_animation_man_beater.html"
            title="Animation Man-Beater"></iframe>
  </div>

  <div class="card">
    <h2>Méthode & limites</h2>
    <p>Séparation = distance euclidienne au défenseur le plus proche à l'instant
    <code>pass_forward</code>, calculée avec <code>scipy.cdist</code>. Seuils : ≥ 30
    actions au total, ≥ 10 par type de couverture.</p>
    <p style="color:var(--muted)"><b>Limites :</b> mesure à un seul instant (pas de
    YAC) · « plus proche » ≠ défenseur assigné · tous les coureurs de route, pas
    seulement la cible · échantillons modestes sur 8 semaines. Détails dans le
    <a href="README.md">README</a>.</p>
  </div>

</div>

<footer>
  NFL Big Data Bowl 2023 — Regional Event Data · Code &amp; données :
  <a href="https://github.com/titease10/NFL_Big_Data">github.com/titease10/NFL_Big_Data</a>
</footer>
</body>
</html>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Page ecrite : {OUT} ({n_players} joueurs)")


if __name__ == "__main__":
    main()
