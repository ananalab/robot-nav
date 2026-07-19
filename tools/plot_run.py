#!/usr/bin/env python3
"""Trace un essai enregistré dans runs/<date>/... (log.csv, path.csv, events.csv).

    .venv/bin/python tools/plot_run.py runs/20261015_120000/b01_seed0
    .venv/bin/python tools/plot_run.py <dossier> --save figure.png

Quatre graphes : trajectoire vs chemin, écart latéral, commandes, temps de calcul.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt


def read_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {key: [float(r[key]) for r in rows] for key in rows[0] if key not in ("type", "message")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--save", help="enregistre l'image au lieu d'ouvrir une fenêtre")
    args = parser.parse_args()

    log = read_csv(os.path.join(args.run_dir, "log.csv"))
    path = read_csv(os.path.join(args.run_dir, "path.csv"))
    if not log:
        raise SystemExit("log.csv vide ou absent dans " + args.run_dir)
    t = log["t"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    traj = ax[0][0]
    if path:
        traj.plot(path["x"], path["y"], "g-", label="chemin planifié")
    if "gt_x" in log:
        traj.plot(log["gt_x"], log["gt_y"], "b-", label="vraie trajectoire (sim)")
    valid = [i for i, v in enumerate(log.get("est_valid", [])) if v > 0.5]
    traj.plot([log["est_x"][i] for i in valid], [log["est_y"][i] for i in valid], "r.", ms=2, label="estimation")
    traj.set_aspect("equal")
    traj.set_title("Trajectoire")
    traj.set_xlabel("x (m)")
    traj.set_ylabel("y (m)")
    traj.legend(fontsize="small", loc="best")

    ax[0][1].plot(t, [1000 * e for e in log["lateral_error"]])
    ax[0][1].set_title("Écart latéral (mm)")
    ax[0][1].set_xlabel("t (s)")

    cmd = ax[1][0]
    cmd.plot(t, log["v_cmd"], label="v (m/s)")
    cmd.plot(t, log["omega_cmd"], label="omega (rad/s)")
    cmd.plot(t, log["u_left"], "--", label="u gauche")
    cmd.plot(t, log["u_right"], "--", label="u droite")
    cmd.set_title("Commandes")
    cmd.set_xlabel("t (s)")
    cmd.legend()

    ax[1][1].plot(t, log["compute_ms"])
    ax[1][1].set_title("Temps de calcul par cycle (ms)")
    ax[1][1].set_xlabel("t (s)")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print("image :", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
