# 교수님 설명 자료

> **감사 후 안내 (2026-10-21)**: 이 자료는 v5 결과 기준이다. v5는 v3 시험 결과를 본 뒤 설계를 바꿨고, 같은 시험 구간을 다시 열었다. 사전등록 재평가(v6 3 fold, v7 확장 grid)에서 Hybrid는 가장 강한 학습 baseline(v6: GNN·ALS+BPR, v7: 평균 최고가 repeat BPR·novel ALS+BPR)보다 일관되게 유의하게 높지는 않았고 (일부 fold·지표에서만 유의. novel 정책의 GNN·ALS 대비 유의 결과는 repeat validation으로 고른 비교 대상 대비, `../audit/validation_report.md` §8 항목 5), "NDCG +10.83%"는 재현되지 않았다. 검증된 결과와 그림은 `../audit/validation_report.md`, `../audit/figures_v6`, `../audit/figures_v7`에 있다. 아래 원문은 이력으로 보존한다. `professor_briefing_ko.md`와 `MEETING_NOTES_KO.md`에는 이 안내를 줄인 안내(가장 강한 baseline 이름, 일부 fold·지표에서만 유의, novel 정책의 비교 대상 조건은 빠짐)를 붙였지만, `professor_briefing_ko.pdf`는 원문 그대로라 이 안내가 없다(따로 전달하지 않는다). `delivery_manifest.json`·`final_delivery_manifest.json`의 해시 중 PDF는 지금 파일과 같다. md 해시는 지금 파일과 맞지 않는다: md 세 개 모두 안내를 붙이기 전 원문(git 커밋 8e70037)을 CRLF 줄바꿈으로 바꾼 해시와 같다(`professor_briefing_ko.md`는 원문 끝에 줄바꿈이 없음).

Hybrid가 두 지표의 세 seed 평균 모두 가장 높았습니다. RetailRocket의 공통 시간순 80:20 분할에서 seed 42·43·44의 평균을 비교했습니다. Hybrid의 NDCG@50은 0.117177, Recall@50은 0.166260입니다.

각 지표의 가장 강한 비교 모델 대비 상대 차이는 NDCG +10.83%, Recall +1.27%입니다. 이 문서의 결과는 기존 논문의 점수를 재사용하지 않고 이번 실행에서 산출한 값입니다.

- professor_briefing_ko.pdf: 회의용 보고서
- professor_briefing_ko.md: 수정 가능한 본문
- figures/: 실제 지표 기반 그림
- evidence.json: 원본 확인 근거
