# paper_v2 experiment notes

Default locked settings for `data/research_revision_paper` (outer test never used for selection).

## GNN changes vs improvement_v1
- **bipartite** user–item graph only (property/category/item–item edges off by default)
- **unweighted** LightGCN propagation; behavior weights stay in the pairwise/contrastive loss
- **InfoNCE** with 32 negatives, temperature 0.2
- **min_epochs=20** before validation can select a checkpoint (avoids premature epoch-1 picks)
- edge dropout 0.1, dim 64, 3 layers, batch 8192

## ALS / Hybrid (kept from improvement_v1, slightly wider fusion grid)
- time decay, α=2, reg=0.02, popularity negatives for ALS+BPR
- hybrid sweep: als_weight ∈ {0.6…1.0}, fusion ∈ {minmax, rrf, gate}, rerank_m, candidate_k, gamma

## Run
```bash
python experiments.py --output-dir ../data/research_revision_paper --device cuda
python viz.py --processed-dir ../data/research_revision_paper
```
