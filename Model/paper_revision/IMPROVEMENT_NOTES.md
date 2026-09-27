# 실험 실행 (기본 = improvement_v1)

옵션 없이 실행하면 개선 설정이 기본이다.

```powershell
cd C:\YS\TUK\S4E2\PBDA\teamproject\PBDA_Teamproject\Model\paper_revision
python experiments.py
```

기본값:
- 설정: improvement_v1 (dim 64, 가중 그래프, hybrid validation sweep 등)
- 출력: `data/research_revision_improve`
- seed 42·43·44, CUDA, ablation 없음

끝난 뒤:

```powershell
python viz.py --experiment-dir ../../data/research_revision_improve --output-dir ../../data/research_revision_improve/figures
```

예전 revision_v3만 다시 돌릴 때:

```powershell
python experiments.py --legacy-v3 --output-dir ../../data/research_revision
```

ablation까지 포함:

```powershell
python experiments.py --with-ablations
```
