import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV 로드
df = pd.read_csv("results/summary_core_ratio_comparison.csv")
df["n"] = df["Graph"].str.extract(r"n(\d+)").astype(int)
df["mu"] = df["Graph"].str.extract(r"mu(\d+)").astype(int)

df["NMI_mean"] = df["NMI"].str.extract(r"([\d\.]+)").astype(float)
df["Modularity_mean"] = df["Modularity"].str.extract(r"([\d\.]+)").astype(float)
df["Time_mean"] = df["Time (s)"].str.extract(r"([\d\.]+)").astype(float)

os.makedirs("results/plots", exist_ok=True)

# 색상 분리용 라벨
df["Label"] = df.apply(lambda row: f"n={row['n']}, mu={row['mu']}", axis=1)

# 공통 함수 정의
def plot_metric(df, metric, ylabel, filename):
    plt.figure(figsize=(8, 5))
    for label, group in df.groupby("Label"):
        plt.plot(group["Core Ratio"], group[metric], marker='o', label=label)

    plt.title(f"Core Ratio vs {ylabel}")
    plt.xlabel("Core Ratio")
    plt.ylabel(ylabel)
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/{filename}.png")
    plt.close()

# 각각 그리기
plot_metric(df, "NMI_mean", "NMI", "core_ratio_all_nmi")
plot_metric(df, "Modularity_mean", "Modularity", "core_ratio_all_modularity")
plot_metric(df, "Time_mean", "Time (s)", "core_ratio_all_time")

print("✅ 통합 그래프 저장 완료: results/plots/core_ratio_all_*.png")