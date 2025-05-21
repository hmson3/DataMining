import pandas as pd

# 불러오기 및 전처리
df = pd.read_csv("results/repeat_raw_results_ratio.csv")
df = df[df["Algorithm"].str.startswith("Leiden-LPA-coreratio_")]

# core_ratio 값 추출
df["core_ratio"] = df["Algorithm"].str.extract(r"coreratio_(\d\.\d)").astype(float)

# 정렬용 정보
df["n"] = df["Graph"].str.extract(r"n(\d+)").astype(int)
df["mu"] = df["Graph"].str.extract(r"mu(\d+)").astype(int)

# 집계
summary = df.groupby(["Graph", "core_ratio"]).agg({
    "Time (s)": ["mean", "std"],
    "Modularity": ["mean", "std"],
    "NMI": ["mean", "std"]
}).reset_index()

# 컬럼 정리
summary.columns = ["Graph", "Core Ratio",
                   "Time_mean", "Time_std",
                   "Mod_mean", "Mod_std",
                   "NMI_mean", "NMI_std"]

def format_mean_std(mean, std, digits=4):
    return f"{mean:.{digits}f} (±{std:.{digits}f})"

# 포맷팅
summary["Time (s)"] = summary.apply(lambda row: format_mean_std(row["Time_mean"], row["Time_std"]), axis=1)
summary["Modularity"] = summary.apply(lambda row: format_mean_std(row["Mod_mean"], row["Mod_std"]), axis=1)
summary["NMI"] = summary.apply(lambda row: format_mean_std(row["NMI_mean"], row["NMI_std"]), axis=1)

# 정렬 및 저장
summary = summary.sort_values(by=["Graph", "Core Ratio"])
summary = summary[["Graph", "Core Ratio", "Time (s)", "Modularity", "NMI"]]
summary.to_csv("results/summary_core_ratio_comparison.csv", index=False)
print("✅ summary_core_ratio_comparison_ratio.csv SAVED!")
