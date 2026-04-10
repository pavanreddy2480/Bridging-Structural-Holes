"""
Plot radar charts and write scores.md from Claude-assigned scores.
Run after generate_vanilla.py has created all hypothesis files.

Usage:
    python3 -m src.plot_scores
"""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT  = DATA / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

DIMS = ["Novelty", "Significance", "Effectiveness", "Clarity", "Feasibility"]

# =============================================================================
# SCORES — assigned by Claude after reading all hypothesis texts
# Format: {pipeline_id: {"discova": [{novelty,significance,...}×N], "vanilla": [...]}}
# =============================================================================
SCORES = {
    "A": {
        "discova": [
            # H1: cs.GR ↔ cs.MA  (low-rank geometry + over-parameterized learning)
            {"novelty":4,"significance":4,"effectiveness":3,"clarity":4,"feasibility":3},
            # H2: cs.GL ↔ cs.MA  (IoU loss for fMRI neural dynamics)
            {"novelty":3,"significance":3,"effectiveness":3,"clarity":4,"feasibility":2},
            # H3: cs.GR ↔ cs.MA  (PLRNN state-space for geometry filtering)
            {"novelty":4,"significance":4,"effectiveness":4,"clarity":5,"feasibility":3},
            # H4: cs.NA ↔ cs.GL  (differentiable physics + KSVD tracking)
            {"novelty":5,"significance":5,"effectiveness":4,"clarity":4,"feasibility":3},
            # H5: cs.GL ↔ cs.GR  (blind deblurring module for CAD mesh denoising)
            {"novelty":3,"significance":3,"effectiveness":4,"clarity":4,"feasibility":4},
        ],
        "vanilla": [
            # H1: cs.GR ↔ cs.MA  — vanilla
            {"novelty":2,"significance":3,"effectiveness":2,"clarity":3,"feasibility":3},
            # H2: cs.GL ↔ cs.MA  — vanilla
            {"novelty":2,"significance":2,"effectiveness":2,"clarity":3,"feasibility":2},
            # H3: cs.GR ↔ cs.MA  — vanilla (fMRI GRNN for geometry — high implementation barrier)
            {"novelty":3,"significance":3,"effectiveness":2,"clarity":3,"feasibility":2},
            # H4: cs.NA ↔ cs.GL  — vanilla (physics sensitivity + KSVD — strongest vanilla)
            {"novelty":3,"significance":3,"effectiveness":3,"clarity":3,"feasibility":3},
            # H5: cs.GL ↔ cs.GR  — vanilla (normal estimation matrix in 2D image deblurring — contrived)
            {"novelty":2,"significance":2,"effectiveness":2,"clarity":3,"feasibility":2},
        ],
    },
    "B": {
        "discova": [
            # H1: cs.NA ↔ cs.GL  (AD sensitivity + STKSVD tracking — structural hole framework)
            {"novelty":4,"significance":3,"effectiveness":3,"clarity":3,"feasibility":3},
            # H2: cs.OH ↔ cs.GL  (AVSD + BID — parameter-constrained decoupled optimization)
            {"novelty":3,"significance":3,"effectiveness":2,"clarity":3,"feasibility":2},
            # H3: cs.OS ↔ cs.GL  (GNN recommender + gaze estimation — domain-adaptive embeddings)
            {"novelty":4,"significance":4,"effectiveness":3,"clarity":4,"feasibility":3},
        ],
        "vanilla": [
            # H1: cs.NA ↔ cs.GL  (Differentiable Sensitivity-Adaptive STKSVD — clear but standard)
            {"novelty":3,"significance":3,"effectiveness":3,"clarity":4,"feasibility":3},
            # H2: cs.OH ↔ cs.GL  (Conversational video deblurring — creative but odd coupling)
            {"novelty":3,"significance":2,"effectiveness":2,"clarity":3,"feasibility":2},
            # H3: cs.OS ↔ cs.GL  (Domain-Informed Embedding Learning — complex, generic)
            {"novelty":2,"significance":3,"effectiveness":2,"clarity":3,"feasibility":2},
        ],
    },
    "C": {
        "discova": [
            # H1: cs.GR ↔ cs.MA  (low-rank geometry + memorization/learning-theory trade-offs)
            {"novelty":4,"significance":3,"effectiveness":3,"clarity":4,"feasibility":3},
            # H2: cs.NA ↔ cs.MS  (robot AD + hashtag propagation — high cross-domain novelty)
            {"novelty":5,"significance":3,"effectiveness":3,"clarity":4,"feasibility":2},
            # H3: cs.MA ↔ cs.NA  (Gegenbauer NN + kinodynamic quadrotor planning)
            {"novelty":4,"significance":4,"effectiveness":3,"clarity":4,"feasibility":3},
            # H4: cs.NA ↔ cs.GL  (AD sensitivity + STKSVD unified framework)
            {"novelty":4,"significance":3,"effectiveness":3,"clarity":4,"feasibility":3},
            # H5: cs.OH ↔ cs.GL  (AVSD + SMPL 3D pose — genuinely novel modality bridge)
            {"novelty":4,"significance":4,"effectiveness":3,"clarity":4,"feasibility":2},
        ],
        "vanilla": [
            # H1: cs.GR ↔ cs.MA  — vanilla
            {"novelty":2,"significance":2,"effectiveness":2,"clarity":3,"feasibility":3},
            # H2: cs.NA ↔ cs.MS  — vanilla
            {"novelty":3,"significance":2,"effectiveness":2,"clarity":3,"feasibility":2},
            # H3: cs.MA ↔ cs.NA  — vanilla (SLFN + B-spline replanning — reasonable but thin)
            {"novelty":3,"significance":3,"effectiveness":2,"clarity":3,"feasibility":3},
            # H4: cs.NA ↔ cs.GL  — vanilla (AD sensitivity + STKSVD — solid but derivative)
            {"novelty":3,"significance":3,"effectiveness":3,"clarity":3,"feasibility":3},
            # H5: cs.OH ↔ cs.GL  — vanilla (AVSD + SMPL pose — vague contribution)
            {"novelty":3,"significance":3,"effectiveness":2,"clarity":3,"feasibility":2},
        ],
    },
    "D": {
        "discova": [
            # H1: cs.NA ↔ cs.MS  (robot AD + social hashtag propagation — constraint bridge)
            {"novelty":5,"significance":3,"effectiveness":3,"clarity":3,"feasibility":2},
            # H2: cs.OS ↔ cs.NA  (GNN recommender + microcontroller robot embedding)
            {"novelty":4,"significance":4,"effectiveness":3,"clarity":4,"feasibility":3},
            # H3: cs.NA ↔ cs.CL  (robot dynamics + SOR hardware solver — high novelty)
            {"novelty":5,"significance":4,"effectiveness":3,"clarity":3,"feasibility":2},
            # H4: cs.MA ↔ cs.NA  (Gegenbauer NN + quadrotor kinodynamic planning)
            {"novelty":4,"significance":4,"effectiveness":3,"clarity":4,"feasibility":3},
            # H5: cs.OS ↔ cs.GL  (GNN continuous preference + gaze estimation)
            {"novelty":3,"significance":3,"effectiveness":3,"clarity":4,"feasibility":3},
        ],
        "vanilla": [
            # H1: cs.NA ↔ cs.MS  — vanilla (robot state-space + social propagation)
            {"novelty":3,"significance":2,"effectiveness":2,"clarity":3,"feasibility":2},
            # H2: cs.OS ↔ cs.NA  — vanilla (GNN recommender + microcontroller — surface-level)
            {"novelty":3,"significance":3,"effectiveness":2,"clarity":3,"feasibility":3},
            # H3: cs.NA ↔ cs.CL  — vanilla (sensitivity + reconfigurable hardware)
            {"novelty":3,"significance":3,"effectiveness":2,"clarity":3,"feasibility":2},
            # H4: cs.MA ↔ cs.NA  — vanilla (SLFN regularization + B-spline quadrotor)
            {"novelty":3,"significance":3,"effectiveness":3,"clarity":3,"feasibility":3},
            # H5: cs.OS ↔ cs.GL  — vanilla (GNN + gaze — no structural context)
            {"novelty":2,"significance":2,"effectiveness":2,"clarity":3,"feasibility":2},
        ],
    },
}

def avg(s): return round(sum(s[d.lower()] for d in DIMS)/len(DIMS), 2)
def mean_dim(scores_list, dim): return round(sum(s[dim]/len(scores_list) for s in scores_list), 2)

def _angles(n):
    a = [i*2*math.pi/n for i in range(n)]
    return a + [a[0]]

COLOR_D = "#2196F3"   # DISCOVA blue
COLOR_V = "#F44336"   # Vanilla red

def plot_pipeline_radar(pipe_id, d_scores, v_scores, label=""):
    dims = DIMS + ["Average"]
    n = len(dims)
    angles = _angles(n)

    def mean_vals(sl):
        return [round(sum(s[d.lower()] for s in sl)/len(sl), 2) for d in DIMS] + \
               [round(sum(avg(s) for s in sl)/len(sl), 2)]

    dv = mean_vals(d_scores)
    vv = mean_vals(v_scores)

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    for lv in range(1,6):
        ax.plot(angles,[lv]*(n+1),"grey",lw=0.5,ls="--",alpha=0.35)
    for lv in (2,3,4):
        ax.text(angles[1],lv+0.15,str(lv),ha="left",va="bottom",fontsize=8,color="grey")

    # Vanilla trace
    ax.fill(angles,vv+[vv[0]],color=COLOR_V,alpha=0.15)
    ax.plot(angles,vv+[vv[0]],color=COLOR_V,lw=2.2,marker="o",ms=6)

    # DISCOVA trace
    ax.fill(angles,dv+[dv[0]],color=COLOR_D,alpha=0.22)
    ax.plot(angles,dv+[dv[0]],color=COLOR_D,lw=2.6,marker="o",ms=7)

    # Annotate DISCOVA values
    for ang,val in zip(angles[:-1],dv):
        ax.annotate(f"{val:.1f}",xy=(ang,val+0.45),fontsize=8,ha="center",va="center",
                    color=COLOR_D,fontweight="bold",xycoords="polar")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims,fontsize=11,fontweight="bold")
    ax.set_ylim(0,5.6)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    handles = [
        mpatches.Patch(color=COLOR_D,label="DISCOVA (ours)",alpha=0.85),
        mpatches.Patch(color=COLOR_V,label="Vanilla LLM",alpha=0.70),
    ]
    ax.legend(handles=handles,loc="upper right",bbox_to_anchor=(1.40,1.15),
              fontsize=10,framealpha=0.85,title="Method",title_fontsize=9)
    title_str = label or f"Pipeline {pipe_id}"
    ax.set_title(f"Hypothesis Evaluation\n{title_str}",fontsize=11,fontweight="bold",pad=22)

    fig.tight_layout()
    path = OUT/f"radar_pipeline_{pipe_id}.png"
    fig.savefig(str(path),dpi=160,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")
    return dv, vv


def plot_all_pipelines(all_data):
    """Combined chart: one DISCOVA trace per pipeline + one Vanilla average."""
    dims = DIMS + ["Average"]
    n = len(dims)
    angles = _angles(n)
    pipe_colors = {"A":"#2196F3","B":"#FF9800","C":"#4CAF50","D":"#9C27B0"}

    fig, ax = plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
    for lv in range(1,6):
        ax.plot(angles,[lv]*(n+1),"grey",lw=0.5,ls="--",alpha=0.35)
    for lv in (2,3,4):
        ax.text(angles[1],lv+0.15,str(lv),ha="left",va="bottom",fontsize=8,color="grey")

    handles = []
    all_vanilla = []
    for pid,(dv,vv) in all_data.items():
        c = pipe_colors[pid]
        ax.fill(angles,dv+[dv[0]],color=c,alpha=0.12)
        ax.plot(angles,dv+[dv[0]],color=c,lw=2.2,marker="o",ms=5)
        handles.append(mpatches.Patch(color=c,label=f"DISCOVA-{pid}",alpha=0.8))
        all_vanilla.append(vv)

    # Average vanilla
    avg_v = [round(sum(vv[i] for vv in all_vanilla)/len(all_vanilla),2) for i in range(n)]
    ax.fill(angles,avg_v+[avg_v[0]],color=COLOR_V,alpha=0.10)
    ax.plot(angles,avg_v+[avg_v[0]],color=COLOR_V,lw=2.2,marker="s",ms=6,ls="--")
    handles.append(mpatches.Patch(color=COLOR_V,label="Vanilla LLM (avg)",alpha=0.6))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims,fontsize=11,fontweight="bold")
    ax.set_ylim(0,5.6)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.legend(handles=handles,loc="upper right",bbox_to_anchor=(1.42,1.15),
              fontsize=9,framealpha=0.85,title="Pipeline",title_fontsize=9)
    ax.set_title("DISCOVA vs Vanilla — All Pipelines",fontsize=12,fontweight="bold",pad=24)
    fig.tight_layout()
    path = OUT/"radar_all_pipelines.png"
    fig.savefig(str(path),dpi=160,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")

# Pipe labels for report
PIPE_LABELS = {
    "A": "Pipeline A — Global TF-IDF + spaCy (main)",
    "B": "Pipeline B — Stratified + spaCy",
    "C": "Pipeline C — Global TF-IDF + Stanza",
    "D": "Pipeline D — Stratified + Stanza",
}

def run():
    # Load ablation scores from JSON files (if available)
    ablation_paths = {
        "B": DATA/"ablation/pipeline_B/comparison",
        "C": DATA/"ablation/pipeline_C/comparison",
        "D": DATA/"ablation/pipeline_D/comparison",
    }
    for pid in ["B","C","D"]:
        for kind in ["discova","vanilla"]:
            f = ablation_paths[pid]/f"{pid}_{kind}.json"
            if f.exists():
                with open(f) as fp:
                    data = json.load(fp)
                scores_in_file = data.get("scores",[])
                if scores_in_file:
                    SCORES[pid][kind] = scores_in_file

    chart_data = {}
    for pid, sc in SCORES.items():
        if not sc["discova"] or not sc["vanilla"]:
            print(f"Pipeline {pid}: no scores available, skipping chart.")
            continue
        dv, vv = plot_pipeline_radar(pid, sc["discova"], sc["vanilla"], PIPE_LABELS.get(pid,""))
        chart_data[pid] = (dv, vv)

    if len(chart_data) > 1:
        plot_all_pipelines(chart_data)

    # Copy to outputs/figures
    import shutil
    fig_dir = ROOT/"outputs"/"figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("radar_*.png"):
        shutil.copy2(f, fig_dir/f.name)
        print(f"Copied {f.name} → outputs/figures/")

if __name__ == "__main__":
    run()
