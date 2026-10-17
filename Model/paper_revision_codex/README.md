# Hybrid 개정 실험 최종 자료

실제 코드 수정, RetailRocket 재실험, 시각화, 교수님 설명 자료를 생성하고 검산·시각 검수를 마쳤다. 두 지표의 세 seed 평균이 모든 비교 모델보다 높았다.

공통 시간순 80:20, seed 42/43/44, original 평가 사용자 854명. Hybrid NDCG@50 0.117177, Recall@50 0.166260.

- professor_materials/professor_briefing_ko.pdf: 교수님 검토용 보고서
- professor_materials/MEETING_NOTES_KO.md: 설명 요약 초안
- professor_materials/figures: 실제 수치의 PNG/PDF/SVG
- results/hybrid_joint_8020_v5: 최종 설정, 지표, 사용자별 추천·검산 입력
- README_HYBRID_KO.md: 재실행 방법, 조건과 한계
- CODE_CHANGES_KO.md / CODE_CHANGES.patch: 실제 변경 검토

회귀검사 46개 통과, 추천 CSV의 90개 지표 쌍 독립 재계산, 원래 ALS/BPR/GNN baseline 추천 파일 일치 확인.

첫 v3의 Recall 미달 후 추가 연구를 수행한 사실을 공개했다. 단일 시기·단일 데이터셋의 정정 평가이며 완전히 미관찰 시험이나 일반화 우위를 주장하지 않는다. 이전 소스와 실패 결과를 보존했다.

큰 모델 배열·원시 학습 데이터는 프로젝트에 남겨 두었고 ZIP에서는 제외했다. ZIP의 경량 추천·정답 CSV로 독립 지표 검산은 가능하다. 교수님께 전송하거나 논문을 재투고하지 않았다.

새 PC 설치와 원시 데이터부터 재실행: [REPRODUCE_KO.md](REPRODUCE_KO.md).
