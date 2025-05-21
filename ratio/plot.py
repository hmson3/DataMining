import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 📥 CSV 불러오기
summary_df = pd.read_csv("results/summary_ratio.csv")

# ✅ 숫자형 값 추출
def extract_mean(val):
    return float(val.split(" ")[0])

summary_df["Time"] = summary_df["Time (s)"].apply(extract_mean)
summary_df["Mod"] = summary_df["Modularity"].apply(extract_mean)
summary_df["NMI"] = summary_df["NMI"].apply(extract_mean)

# 📁 디렉토리 준비
os.makedirs("graphs", exist_ok=True)

# 📈 개별 그래프 저장
metrics = [("Time", "Time (s)", "orange"),
           ("Mod", "Modularity", "blue"),
           ("NMI", "NMI", "green")]

for graph in summary_df["Graph"].unique():
    graph_df = summary_df[summary_df["Graph"] == graph]
    graph_dir = os.path.join("graphs", graph)
    os.makedirs(graph_dir, exist_ok=True)

    for col, label, color in metrics:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=graph_df, x="Core Ratio", y=col, marker="o", color=color)
        plt.title(f"{label} vs Core Ratio\n({graph})")
        plt.xlabel("Core Ratio")
        plt.ylabel(label)
        plt.ylim(0, 1 if col != "Time" else None)
        plt.tight_layout()
        
        file_path = os.path.join(graph_dir, f"{label.replace(' ', '_').lower()}.png")
        plt.savefig(file_path)
        plt.close()
        print(f"✅ Saved: {file_path}")

# 📊 전체 종합 그래프 3종
for col, label, color in metrics:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x="Core Ratio", y=col, hue="Graph", marker="o")
    plt.title(f"{label} vs Core Ratio (All Graphs)")
    plt.xlabel("Core Ratio")
    plt.ylabel(label)
    if col != "Time":
        plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    file_path = f"graphs/overall_{label.replace(' ', '_').lower()}.png"
    plt.savefig(file_path)
    plt.close()
    print(f"📊 Saved overall: {file_path}")
