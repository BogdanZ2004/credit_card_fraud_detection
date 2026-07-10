"""Poređenje rezultata dve grane (main = 30 atributa vs 18_atributa).

Čita 'ZBIRNA TABELA' sekciju iz model_comparison.txt svake grane i pravi
pregled: po modelu i po metrici — koja grana je bolja i za koliko. Sve metrike
su takve da je VEĆE bolje (Prec, Recall, F1, F2, ROC-AUC, AUPRC).

Pokretanje:  python poredjenje_rezultata/uporedi.py
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
A_NAME, B_NAME = "main", "18_atributa"   # A = referenca (30 atr.), B = 18 atr.
PATHS = {
    A_NAME: os.path.join(BASE, "main", "metrics", "model_comparison.txt"),
    B_NAME: os.path.join(BASE, "18_atributa", "metrics", "model_comparison.txt"),
}
METRICS = ["Prec", "Recall", "F1", "F2", "ROC-AUC", "AUPRC"]


def parse_zbirna(path):
    """Vraća {model: {metrika: vrednost}} iz 'ZBIRNA TABELA' sekcije."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    start = next((i for i, l in enumerate(lines) if "ZBIRNA TABELA" in l), None)
    if start is None:
        raise ValueError(f"Nema 'ZBIRNA TABELA' u {path}")
    rows = {}
    for l in lines[start + 2:]:            # +1 red = zaglavlje kolona, dalje = podaci
        parts = l.split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(x) for x in parts[1:7]]
        except ValueError:
            continue
        rows[parts[0]] = dict(zip(METRICS, vals))
    return rows


def main():
    data = {name: parse_zbirna(path) for name, path in PATHS.items()}
    a, b = data[A_NAME], data[B_NAME]
    models = [m for m in a if m in b]

    lines = []
    def emit(s=""):
        lines.append(s)

    emit(f"=== POREĐENJE GRANA: {A_NAME} (30 atr.) vs {B_NAME} (18 atr.) — test skup, prag 0.5 ===")
    emit("Δ = 18_atributa − main  (pozitivno => 18 atributa bolje). Veće je bolje za sve metrike.\n")

    total = {A_NAME: 0, B_NAME: 0, "tie": 0}
    per_metric = {m: {A_NAME: 0, B_NAME: 0, "tie": 0} for m in METRICS}

    for model in models:
        emit(f"--- {model} ---")
        for m in METRICS:
            va, vb = a[model][m], b[model][m]
            d = vb - va
            if abs(d) < 5e-5:
                winner = "tie"
            elif vb > va:
                winner = B_NAME
            else:
                winner = A_NAME
            total[winner] += 1
            per_metric[m][winner] += 1
            emit(f"  {m:8s} main={va:.4f}  18={vb:.4f}   Δ={d:+.4f}   -> {winner}")
        emit()

    emit("=== SAŽETAK PO METRICI (na koliko modela je koja grana bolja) ===")
    for m in METRICS:
        w = per_metric[m]
        emit(f"  {m:8s} main: {w[A_NAME]}   18_atributa: {w[B_NAME]}   nereseno: {w['tie']}")
    emit()

    emit("=== FOKUS: AUPRC (primarna metrika) po modelu ===")
    for model in models:
        va, vb = a[model]["AUPRC"], b[model]["AUPRC"]
        d = vb - va
        winner = "tie" if abs(d) < 5e-5 else (B_NAME if d > 0 else A_NAME)
        emit(f"  {model:20s} main={va:.4f}  18={vb:.4f}   Δ={d:+.4f}   -> {winner}")
    emit()

    n = len(models) * len(METRICS)
    emit(f"=== UKUPNO ({len(models)} modela × {len(METRICS)} metrika = {n} poređenja) ===")
    emit(f"  main bolji:        {total[A_NAME]}")
    emit(f"  18_atributa bolji: {total[B_NAME]}")
    emit(f"  nereseno:          {total['tie']}")

    report = "\n".join(lines)
    print(report)
    out_path = os.path.join(BASE, "poredjenje.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSačuvano u: {out_path}")


if __name__ == "__main__":
    main()
