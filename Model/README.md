# Model

추천 시스템 코드를 논문 원본과 revision으로 나눠 둡니다.

| 폴더 | 내용 | 출처 |
|------|------|------|
| [`paper/`](paper/) | 제출·GitHub에 올라간 원본 모델 코드 | [Nekerworld/PBDA_Teamproject](https://github.com/Nekerworld/PBDA_Teamproject.git) `Model/` |
| [`paper_revision/`](paper_revision/) | IEEE Access 리비전용 재현·공정 프로토콜·paper_v2 실험 코드 | 현재 워크스페이스 개량본 |

## 실행

원본 파이프라인:

```powershell
cd Model/paper
python model.py
```

리비전 실험·시각화:

```powershell
cd Model/paper_revision
python experiments.py --output-dir ../../data/research_revision_paper --device cuda
python viz.py --experiment-dir ../../data/research_revision_paper --output-dir ../../data/research_revision_paper/figures
python -m pytest test_revision.py -q
```

데이터는 프로젝트 루트의 `data/`를 사용합니다 (`paper_revision`의 `data_protocol.ROOT` 기준).
