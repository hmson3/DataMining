import pandas as pd

# 버전 설정 (예: "Leiden-LPA-v1")
ALGORITHM_VERSION = "Leiden-LPA-v2"

# 불러오기 및 전처리
df = pd.read_csv(f"results/results_{ALGORITHM_VERSION}.csv")
df = df[df["Algorithm"].isin(["Leiden", ALGORITHM_VERSION])]

# 중복 제거 (선택)
df = df.drop_duplicates(subset=["Graph", "Repeat", "Algorithm"])

# 정렬용 정보 추출
df["n"] = df["Graph"].str.extract(r"n(\d+)").astype(int)
df["mu"] = df["Graph"].str.extract(r"mu(\d+)").astype(int)

# 집계
summary = df.groupby(["Graph", "Algorithm"]).agg({
    "Time (s)": ["mean", "std"],
    "Modularity": ["mean", "std"],
    "NMI": ["mean", "std"]
}).reset_index()

# 컬럼 정리
summary.columns = ["Graph", "Algorithm",
                   "Time_mean", "Time_std",
                   "Mod_mean", "Mod_std",
                   "NMI_mean", "NMI_std"]

def format_mean_std(mean, std, digits=4):
    return f"{mean:.{digits}f} (±{std:.{digits}f})"

# 포맷팅 적용
summary["Time (s)"] = summary.apply(lambda row: format_mean_std(row["Time_mean"], row["Time_std"]), axis=1)
summary["Modularity"] = summary.apply(lambda row: format_mean_std(row["Mod_mean"], row["Mod_std"]), axis=1)
summary["NMI"] = summary.apply(lambda row: format_mean_std(row["NMI_mean"], row["NMI_std"]), axis=1)

# 최종 정렬
summary["n"] = summary["Graph"].str.extract(r"n(\d+)").astype(int)
summary["mu"] = summary["Graph"].str.extract(r"mu(\d+)").astype(int)
summary = summary.sort_values(by=["n", "mu", "Algorithm"])

# 결과 저장
summary = summary[["Graph", "Algorithm", "Time (s)", "Modularity", "NMI"]]
versioned_path = f"results/summary_{ALGORITHM_VERSION}.csv"
summary.to_csv(versioned_path, index=False)
print(f"✅ {versioned_path} SAVED!")