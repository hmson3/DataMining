import pandas as pd
from scipy.stats import zscore

# 버전 설정
ALGORITHM_VERSION = "Leiden-LPA-v11"

# 불러오기
df = pd.read_csv(f"results/results_{ALGORITHM_VERSION}.csv")
df = df[df["Algorithm"].isin(["Leiden", ALGORITHM_VERSION])]
df = df.drop_duplicates(subset=["Graph", "Repeat", "Algorithm"])

# 정렬용 정보
df["n"] = df["Graph"].str.extract(r"n(\d+)").astype(int)
df["mu"] = df["Graph"].str.extract(r"mu(\d+)").astype(int)

# ✅ z-score 기반 이상치 제거
def remove_outliers(group):
    z_cols = ["Time (s)", "Modularity", "NMI"]
    mask = pd.Series([True] * len(group), index=group.index)
    for col in z_cols:
        if group[col].std() > 1e-8:
            mask &= abs(zscore(group[col])) < 2
    return group[mask]

df_clean = df.groupby(["Graph", "Algorithm"], group_keys=False).apply(remove_outliers)

# 요약 통계
summary = df_clean.groupby(["Graph", "Algorithm"]).agg({
    "Time (s)": ["mean", "std"],
    "Modularity": ["mean", "std"],
    "NMI": ["mean", "std"]
}).reset_index()

# 컬럼 이름 정리
summary.columns = ["Graph", "Algorithm",
                   "Time_mean", "Time_std",
                   "Mod_mean", "Mod_std",
                   "NMI_mean", "NMI_std"]

# 포맷 함수
def format_mean_std(mean, std, digits=4):
    return f"{mean:.{digits}f} (±{std:.{digits}f})"

# 포맷 적용
summary["Time (s)"] = summary.apply(lambda row: format_mean_std(row["Time_mean"], row["Time_std"]), axis=1)
summary["Modularity"] = summary.apply(lambda row: format_mean_std(row["Mod_mean"], row["Mod_std"]), axis=1)
summary["NMI"] = summary.apply(lambda row: format_mean_std(row["NMI_mean"], row["NMI_std"]), axis=1)

# 정렬
summary["n"] = summary["Graph"].str.extract(r"n(\d+)").astype(int)
summary["mu"] = summary["Graph"].str.extract(r"mu(\d+)").astype(int)
summary = summary.sort_values(by=["n", "mu", "Algorithm"])

# 저장
summary = summary[["Graph", "Algorithm", "Time (s)", "Modularity", "NMI"]]
versioned_path = f"results/summary_{ALGORITHM_VERSION}.csv"
summary.to_csv(versioned_path, index=False)
print(f"✅ {versioned_path} SAVED!")
