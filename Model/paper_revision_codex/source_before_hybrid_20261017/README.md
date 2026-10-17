# paper_revision_codex

먼저 `REVISION_ANALYSIS_KO.md`를 읽으세요. 기존 `paper_revision`을 바탕으로 만든 별도 수정본입니다.

원시 데이터: 프로젝트 루트의 `data/`. 새 실행 결과: 이 폴더의 `results/`.

```powershell
cd C:\YS\TUK\S4E2\PBDA\teamproject\PBDA_Teamproject2\Model\paper_revision_codex
python -m unittest test_codex_audit -v
python audit_revision.py --experiment-dir ../../data/research_revision_paper --output-dir audit_new --strict
```

감사는 기존 결과에 문제가 있으면 `--strict`에서 종료코드 2를 반환합니다. `audit_verified`에 이번 감사 결과가 있습니다. 원본 결과는 덮어쓰지 않습니다.

모델 실행 환경에는 requirements.txt의 패키지가 필요합니다. 추가 검사에는 pandas가 필요하고 감사 자체는 표준 라이브러리만 사용합니다.

```powershell
python -m pytest test_revision.py -q
# 아래는 기본 IF 켬, 시간순 90:10으로 전체 재학습하는 명령입니다.
# 실행 전 분석 보고서의 실험 조건 결정 사항을 확인하세요.
python experiments.py --device cuda --output-dir results/research_new
```

희소도 수정과 출력 경로 외 모델 학습 로직은 원본을 유지했습니다. `audit_revision.py`와 `test_codex_audit.py`는 새 도구입니다. `validation_log.txt`는 이번 실행 기록입니다.
