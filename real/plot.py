import pandas as pd
import matplotlib.pyplot as plt
import os, re

# ── 데이터 로드 ────────────────────────────────────────
df = pd.read_csv("results/summary_Leiden-LPA.csv")

def extract_mean(v):
    if isinstance(v, str):
        m = re.match(r"([0-9.]+)", v)
        return float(m.group(1)) if m else None
    return v

for col in ["Time (s)", "Modularity", "NMI"]:
    df[col] = df[col].apply(extract_mean)

out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)

# ── 공통 색상 매핑 ─────────────────────────────────────
colors = {"Leiden": "black", "Leiden-LPA": "gray"}

# ── 1) Time (s) broken-axis ────────────────────────────
pivot = df.pivot(index="Graph", columns="Algorithm", values="Time (s)")

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(10, 6), height_ratios=[3, 1]
)

bars_top = pivot.plot(
    kind='bar', ax=ax_top, legend=True,
    edgecolor='black', linewidth=0.5,
    color=[colors[c] for c in pivot.columns]
)
pivot.plot(
    kind='bar', ax=ax_bot, legend=False,
    edgecolor='black', linewidth=0.5,
    color=[colors[c] for c in pivot.columns]
)

# y-축 범위
ax_top.set_ylim(0.1, pivot.to_numpy().max() + 100)
ax_bot.set_ylim(0, 0.1)
ax_top.set_ylabel("Time (s)", fontsize=10)

# 축 잘라내기 표시
for ax in (ax_top, ax_bot):
    ax.spines['bottom' if ax is ax_top else 'top'].set_visible(False)
ax_top.tick_params(labeltop=False, bottom=False)
ax_bot.xaxis.tick_bottom()
d = .015
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1-d, 1+d), **kwargs)
ax_bot.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{out_dir}/time_broken_axis.png", dpi=300)
plt.show()

# ── 2) Modularity & NMI ────────────────────────────────
def plot_metric(metric):
    piv = df.pivot(index="Graph", columns="Algorithm", values=metric)
    ax = piv.plot(
        kind='bar', figsize=(10, 5), rot=30,
        edgecolor='black', linewidth=0.5,
        color=[colors[c] for c in piv.columns]
    )
    ax.set_ylabel(metric, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{metric.lower()}_bar.png", dpi=300)
    plt.show()

for m in ["Modularity", "NMI"]:
    plot_metric(m)
