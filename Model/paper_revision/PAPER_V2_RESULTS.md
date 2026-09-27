# paper_v2 results (for manuscript rewrite)

Protocol unchanged: chronological-fixed-reference, `no_if_events`, original scope, K=50, seeds 42–44.
Selection (fusion/gamma/epochs) uses validation only.

## Locked stack (`mode=paper_v2`)
- ALS/BPR: same as improvement_v1 (time decay, α=2, reg=0.02, popularity negatives)
- LightGCN: **bipartite** user–item only, **unweighted** propagation, **InfoNCE** (32 negatives, T=0.2), dim 64 / 3 layers, min_epochs=20
- Hybrid: validation sweep over fusion / als_weight / candidate_k / gamma (still ALS+BPR retrieval → ALS+GNN fuse)

Outputs: `data/research_revision_paper/`  
Figures: `data/research_revision_paper/figures/`

## Outer-test original K=50 (mean ± std over 3 seeds)

| Model | paper_v2 NDCG | improve NDCG | paper_v2 Recall | improve Recall |
|-------|---------------|--------------|-----------------|----------------|
| ALS | 0.0737 ± 0.0009 | 0.0737 ± 0.0009 | 0.1161 | 0.1161 |
| BPR | 0.0978 ± 0.0008 | 0.0978 ± 0.0008 | 0.1321 | 0.1321 |
| ALS+BPR | 0.1046 ± 0.0002 | 0.1046 ± 0.0002 | 0.1447 | 0.1447 |
| **GNN** | **0.1062 ± 0.0023** | 0.0459 ± 0.0016 | **0.1644** | 0.0782 |
| Hybrid | 0.0911 ± 0.0009 | 0.0823 ± 0.0025 | 0.1455 | 0.1265 |

## Interpretation for the paper
1. **GNN is now the strongest single model** under the fair protocol (beats ALS+BPR on NDCG and clearly on Recall).
2. ALS / BPR / ALS+BPR match improvement_v1 (same training stack).
3. Hybrid improved vs improvement_v1 but **still trails GNN and ALS+BPR** on outer test: candidates are retrieved from ALS+BPR only, so a stronger GNN is bottlenecked before fusion. Do **not** claim hybrid superiority from these numbers.
4. Validation chose Hybrid around `als_weight=0.6`, `fusion=minmax`, `gamma=1.5` (val NDCG ≈0.108), below standalone GNN validation (~0.128).

## Reproduce
```bash
cd Model/paper_revision
python experiments.py --output-dir ../../data/research_revision_paper --device cuda
python viz.py --experiment-dir ../../data/research_revision_paper --output-dir ../../data/research_revision_paper/figures --dpi 300
```
