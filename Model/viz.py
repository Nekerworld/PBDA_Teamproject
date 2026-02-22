# # ========== 베이스라인 모델 비교 실험 시각화 ==========
# import matplotlib.pyplot as plt
# import numpy as np

# # Ablation study 결과
# models = ["ALS", "MF-BPR", "ALS + MF-BPR", "GNN", "Hybrid\n(ALS + MF-BPR)+GNN"]
# ndcg_50 = [0.1810, 0.1478, 0.1994, 0.0949, 0.3710]
# recall_50 = [0.2421, 0.2115, 0.2472, 0.2950, 0.7113]

# x = np.arange(len(models))
# width = 0.35

# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width / 2, ndcg_50, width, label="NDCG@50", color="#2ecc71", edgecolor="black", linewidth=0.8)
# bars2 = ax.bar(x + width / 2, recall_50, width, label="Recall@50", color="#3498db", edgecolor="black", linewidth=0.8)

# ax.set_xlabel("Model", fontsize=12, fontweight="bold")
# ax.set_ylabel("Score", fontsize=12, fontweight="bold")
# ax.set_title("Baseline Comparison: NDCG@50 and Recall@50 by Model", fontsize=14, fontweight="bold")
# ax.set_xticks(x)
# ax.set_xticklabels(models, fontsize=10)
# ax.legend(loc="upper right", fontsize=10)
# ax.set_ylim(0, 0.85)
# ax.grid(axis="y", alpha=0.3, linestyle="--")

# # 막대 위에 값 표시
# def add_value_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(
#             f"{height:.3f}",
#             xy=(bar.get_x() + bar.get_width() / 2, height),
#             xytext=(0, 3),
#             textcoords="offset points",
#             ha="center",
#             va="bottom",
#             fontsize=8,
#             fontweight="bold",
#         )

# add_value_labels(bars1)
# add_value_labels(bars2)

# plt.tight_layout()
# plt.show()


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


# ========== Top-K 실험 시각화 ==========
import matplotlib.pyplot as plt
import numpy as np

K_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
NDCG_AT_K = [0.3285, 0.3571, 0.3693, 0.3766, 0.3816, 0.3852, 0.3885, 0.3912, 0.3936, 0.3954, 0.3968, 0.3980, 0.3993, 0.4005, 0.4012, 0.4018, 0.4026, 0.4033, 0.4039, 0.4042]
RECALL_AT_K = [0.5098, 0.6105, 0.6592, 0.6904, 0.7120, 0.7280, 0.7439, 0.7573, 0.7696, 0.7792, 0.7860, 0.7927, 0.8001, 0.8073, 0.8109, 0.8146, 0.8188, 0.8232, 0.8271, 0.8287]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(K_VALUES, NDCG_AT_K, "o-", color="#2ecc71", linewidth=2, markersize=6, label="NDCG@K")
ax1.set_ylabel("NDCG@K", fontsize=12, fontweight="bold")
ax1.set_title("Top-K Experiment: NDCG@K and Recall@K", fontsize=14, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3, linestyle="--")
ax1.set_ylim(0.3, 0.45)

ax2.plot(K_VALUES, RECALL_AT_K, "s-", color="#3498db", linewidth=2, markersize=6, label="Recall@K")
ax2.set_xlabel("K", fontsize=12, fontweight="bold")
ax2.set_ylabel("Recall@K", fontsize=12, fontweight="bold")
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3, linestyle="--")
ax2.set_ylim(0.5, 0.9)

plt.tight_layout()
plt.show()
