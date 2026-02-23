# ========== 베이스라인 모델 비교 실험 시각화 ==========
import matplotlib.pyplot as plt
import numpy as np

# 베이스라인 비교 결과 (NDCG, Recall, 총 시간)
models = ["1. ALS only", "2. BPR only", "3. ALS+BPR", "4. GNN", "5. ALS+GNN"]
ndcg_50 = [0.1810, 0.1478, 0.1994, 0.0949, 0.3710]
recall_50 = [0.2421, 0.2115, 0.2472, 0.2950, 0.7113]
total_sec = [97.04, 24.67, 1642.79, 912.74, 2705.54]

x = np.arange(len(models))
width = 0.25

fig, ax1 = plt.subplots(figsize=(11, 6))

# 왼쪽 Y축: NDCG, Recall 막대
bars1 = ax1.bar(x - width, ndcg_50, width, label="NDCG@50", color="#2ecc71", edgecolor="black", linewidth=0.8)
bars2 = ax1.bar(x, recall_50, width, label="Recall@50", color="#3498db", edgecolor="black", linewidth=0.8)

ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
ax1.set_ylabel("NDCG@50 / Recall@50", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 0.85)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

def add_value_labels(bars, ax):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
add_value_labels(bars1, ax1)
add_value_labels(bars2, ax1)

# 오른쪽 Y축: 총 시간(초) 히스토그램
ax2 = ax1.twinx()
bars_time = ax2.bar(x + width, total_sec, width, label="total(sec)", color="#9b59b6", alpha=0.7, edgecolor="black", linewidth=0.8)
ax2.set_ylabel("total(sec)", fontsize=12, fontweight="bold")
ax2.tick_params(axis="y")
ax2.set_ylim(0, max(total_sec) * 1.12)
for i, (xi, t) in enumerate(zip(x + width, total_sec)):
    ax2.annotate(f"{t:.1f}", xy=(xi, t), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

# 범례 (ax1 기준, 시간 막대 포함)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

ax1.set_title("Baseline Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


# # ========== 하이퍼파라미터 실험 시각화 ==========
# import matplotlib.pyplot as plt
# import numpy as np

# # 데이터: (Parameter, Value, NDCG, Recall) — baseline (default) 제외
# HP_DATA = [
#     # ALS weight
#     ("ALS weight", "0", 0.183638724, 0.278755828),
#     ("ALS weight", "0.2", 0.365565189, 0.71090026),
#     ("ALS weight", "0.4", 0.371008886, 0.71129513),
#     ("ALS weight", "0.6", 0.392414027, 0.71217089),
#     ("ALS weight", "0.8", 0.404795818, 0.712819349),
#     ("ALS weight", "1", 0.398563437, 0.696654782),
#     # ALS factors
#     ("ALS factors", "16", 0.108699835, 0.263595295),
#     ("ALS factors", "32", 0.191905906, 0.416876874),
#     ("ALS factors", "64", 0.291574272, 0.578443588),
#     ("ALS factors", "128", 0.371008886, 0.71129513),
#     # ALS alpha
#     ("ALS alpha", "0.5", 0.292690628, 0.609283993),
#     ("ALS alpha", "1", 0.346290804, 0.674038057),
#     ("ALS alpha", "1.5", 0.371008886, 0.71129513),
#     ("ALS alpha", "2", 0.379746609, 0.73096977),
#     # ALS regularization
#     ("ALS regularization", "0.01", 0.37067412, 0.710900702),
#     ("ALS regularization", "0.03", 0.371008886, 0.71129513),
#     ("ALS regularization", "0.1", 0.37000126, 0.710994577),
#     ("ALS regularization", "0.3", 0.369939694, 0.710710271),
#     # GNN embedding_dim
#     ("GNN embedding_dim", "8", 0.371008886, 0.71129513),
#     ("GNN embedding_dim", "16", 0.370754488, 0.711232402),
#     ("GNN embedding_dim", "32", 0.370177711, 0.711399827),
#     ("GNN embedding_dim", "64", 0.369424138, 0.711406145),
#     # GNN layers
#     ("GNN layers", "1", 0.373563611, 0.709807989),
#     ("GNN layers", "2", 0.371008886, 0.71129513),
#     ("GNN layers", "3", 0.367962458, 0.710793584),
#     # GNN learning_rate
#     ("GNN learning_rate", "0.0001", 0.37140211, 0.711008845),
#     ("GNN learning_rate", "0.0005", 0.369963777, 0.710571728),
#     ("GNN learning_rate", "0.001", 0.371008886, 0.71129513),
#     ("GNN learning_rate", "0.005", 0.369209145, 0.709052269),
# ]

# param_groups = {}
# for param, val, ndcg, rec in HP_DATA:
#     if param not in param_groups:
#         param_groups[param] = {"values": [], "ndcg": [], "recall": []}
#     param_groups[param]["values"].append(val)
#     param_groups[param]["ndcg"].append(ndcg)
#     param_groups[param]["recall"].append(rec)

# n_params = len(param_groups)
# fig, axes = plt.subplots(2, 4, figsize=(16, 10))
# axes = axes.flatten()

# for idx, (param_name, data) in enumerate(param_groups.items()):
#     ax = axes[idx]
#     x = np.arange(len(data["values"]))
#     width = 0.35
#     bars1 = ax.bar(x - width / 2, data["ndcg"], width, label="NDCG@50", color="#2ecc71", edgecolor="black", linewidth=0.8)
#     bars2 = ax.bar(x + width / 2, data["recall"], width, label="Recall@50", color="#3498db", edgecolor="black", linewidth=0.8)
#     ax.set_title(param_name, fontsize=11, fontweight="bold")
#     ax.set_xticks(x)
#     ax.set_xticklabels(data["values"], fontsize=9)
#     ax.set_ylabel("Score", fontsize=9)
#     ax.legend(loc="upper right", fontsize=8)
#     ax.grid(axis="y", alpha=0.3, linestyle="--")
#     ax.set_ylim(0, 0.85)

# if n_params < len(axes):
#     for j in range(n_params, len(axes)):
#         axes[j].set_visible(False)

# fig.suptitle("Hyperparameter Sensitivity: NDCG@50 and Recall@50", fontsize=14, fontweight="bold", y=1.02)
# plt.tight_layout()
# plt.show()


# # ========== Top-K 실험 시각화 ==========
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

# K_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
# NDCG_AT_K = [0.3285, 0.3571, 0.3693, 0.3766, 0.3816, 0.3852, 0.3885, 0.3912, 0.3936, 0.3954, 0.3968, 0.3980, 0.3993, 0.4005, 0.4012, 0.4018, 0.4026, 0.4033, 0.4039, 0.4042]
# RECALL_AT_K = [0.5098, 0.6105, 0.6592, 0.6904, 0.7120, 0.7280, 0.7439, 0.7573, 0.7696, 0.7792, 0.7860, 0.7927, 0.8001, 0.8073, 0.8109, 0.8146, 0.8188, 0.8232, 0.8271, 0.8287]
# TIME_SEC = [14.0288, 14.9087, 15.7615, 16.5632, 17.3668, 18.1122, 19.1630, 19.7286, 20.5668, 21.1827, 22.1466, 22.7010, 23.6719, 24.1465, 25.6536, 26.2626, 26.8549, 27.7453, 28.2033, 29.1560]

# # 논문용 스타일 (선택)
# plt.rcParams["font.size"] = 11
# plt.rcParams["axes.labelsize"] = 12
# plt.rcParams["axes.titlesize"] = 13
# plt.rcParams["legend.fontsize"] = 10

# fig, ax1 = plt.subplots(figsize=(7, 4.2))
# ax1.set_xlabel("K", fontsize=12)
# ax1.set_ylabel("NDCG@K / Recall@K", fontsize=12, color="#333")
# ax1.set_ylim(0.28, 0.88)
# ax1.tick_params(axis="y", labelcolor="#333")
# ax1.grid(True, alpha=0.25, linestyle="-")

# # 라인: NDCG, Recall
# l1, = ax1.plot(K_VALUES, NDCG_AT_K, "o-", color="#2171b5", linewidth=2, markersize=5, label="NDCG@K")
# l2, = ax1.plot(K_VALUES, RECALL_AT_K, "s-", color="#238b45", linewidth=2, markersize=5, label="Recall@K")

# # 오른쪽 Y축: 시간(막대)
# ax2 = ax1.twinx()
# bars = ax2.bar(K_VALUES, TIME_SEC, width=7, color="gray", alpha=0.35, edgecolor="gray", linewidth=0.8, label="Time (s)")
# ax2.set_ylabel("Time (s)", fontsize=12)
# ax2.tick_params(axis="y", labelcolor="#555")
# ax2.set_ylim(0, max(TIME_SEC) * 1.12)

# # 막대 위 값
# for k, t in zip(K_VALUES, TIME_SEC):
#     ax2.annotate(f"{t:.1f}", xy=(k, t), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, color="#333")

# # 하나의 범례에 모두 (막대는 Patch로)
# bar_patch = mpatches.Patch(facecolor="gray", alpha=0.35, edgecolor="gray", label="Time (s)")
# ax1.legend(handles=[l1, l2, bar_patch], labels=["NDCG@K", "Recall@K", "Time (s)"], loc="upper left", framealpha=0.95)
# fig.suptitle("Top-K experiment: NDCG, Recall and runtime", fontsize=13, fontweight="bold", y=1.02)
# plt.tight_layout()
# plt.title("Top-K Comparison")
# plt.show()