# EDA 가이드

반갑습니다! 
이곳은 RetailRocket 데이터셋을 활용한 추천 시스템 구축을 위한 EDA 모듈입니다.

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [설치 요구사항](#설치-요구사항)
- [빠른 시작](#빠른-시작)
- [분석 항목](#분석-항목)
- [결과 파일](#결과-파일)
- [사용 예시](#사용-예시)
- [문제 해결](#문제-해결)

## 개요

이 모듈은 RetailRocket E-commerce 데이터셋을 분석하여 추천 시스템 구축에 필요한 인사이트를 제공합니다. 사용자 행동 패턴, 아이템 특성, 카테고리 구조 등을 종합적으로 분석하고 인터랙티브한 시각화 리포트를 생성합니다.

## 주요 기능

- **데이터 기본 구조 파악**: 데이터셋 크기, 결측치, 컬럼 정보 등 기본 통계
- **이벤트 분석**: 전환 퍼널, 시간대별/요일별 트렌드 분석
- **아이템 속성 분석**: 상품 메타데이터 및 속성 분포 분석
- **카테고리 트리 분석**: 계층 구조 시각화 및 분석
- **사용자 행동 패턴 분석**: 사용자별 행동 유형 분류 및 패턴 분석
- **아이템 유사도 특성 분석**: 추천 시스템을 위한 아이템 특성 추출
- **세션 기반 추천 분석**: 세션 분할 및 사용자 여정 분석
- **이상치 탐지**: 비정상적인 사용자/아이템 행동 식별
- **인터랙티브 시각화**: Plotly 기반 동적 차트 및 HTML 리포트 생성

## 설치 요구사항

```bash
# EDA 폴더의 requirements.txt를 사용하여 패키지 설치
cd EDA
pip install -r requirements.txt
```

### 필수 패키지

- `pandas` (>=1.5.0): 데이터프레임 처리 및 분석
- `numpy` (>=1.21.0): 수치 연산
- `matplotlib` (>=3.5.0): 정적 시각화
- `seaborn` (>=0.11.0): 통계적 시각화
- `plotly` (>=5.0.0): 인터랙티브 시각화
- `networkx` (>=2.8.0): 네트워크 분석 (카테고리 트리)

## 빠른 시작

### 1. 데이터 준비

`data/` 폴더에 다음 파일들이 있어야 합니다:

- `events.csv`: 고객 행동 이벤트 로그
- `item_properties_part1.csv`: 아이템 속성 (1부)
- `item_properties_part2.csv`: 아이템 속성 (2부)
- `category_tree.csv`: 카테고리 계층 구조

### 2. EDA 실행

```bash
cd EDA
python EDA.py
```

### 3. 결과 확인

실행 완료 후 `retailrocket_recommendation_report.html` 파일이 생성됩니다. 브라우저에서 열어 인터랙티브 리포트를 확인할 수 있습니다.

## 분석 항목

### 1. 데이터 기본 구조 파악 (`basic_data_overview`)

- 각 데이터셋의 행/열 수, 메모리 사용량
- 컬럼 정보 및 데이터 타입
- 결측치 확인

### 2. 이벤트 분석 (`analyze_events`)

- **이벤트 타입 분포**: view, addtocart, transaction 비율
- **전환 퍼널**: View → AddToCart → Transaction 전환율
- **시간 기반 분석**: 
  - 데이터 기간 및 일별 이벤트 수
  - 시간대별/요일별 이벤트 분포
- **사용자 행동 수준**: 고유 방문자 수, 방문자당 평균 이벤트 수

### 3. 아이템 속성 분석 (`analyze_item_properties`)

- 아이템 속성 분포 및 통계
- 카테고리별 아이템 수
- 속성값 분포 분석

### 4. 카테고리 트리 분석 (`analyze_category_tree`)

- 카테고리 계층 구조 분석
- 카테고리 깊이 분포
- 네트워크 그래프 시각화

### 5. 사용자 행동 패턴 분석 (`analyze_user_behavior_patterns`)

- 사용자 행동 유형 분류:
  - **탐색형**: 많은 view, 적은 구매
  - **구매형**: 높은 전환율
  - **관심형**: 많은 addtocart
- 사용자별 행동 통계

### 6. 아이템 유사도 특성 분석 (`analyze_item_similarity_features`)

- 아이템별 상호작용 통계
- 인기도 점수 계산
- 전환율 계산
- 카테고리 정보 통합

### 7. 세션 기반 추천 분석 (`analyze_session_based_recommendations`)

- 30분 기준 세션 분할
- 세션별 행동 패턴 분석
- 아이템 전환 패턴 분석

### 8. 이상치 탐지 (`detect_anomalies`)

- IQR 방법을 사용한 이상치 탐지
- 비정상적인 사용자/아이템 행동 식별
- 이상치 비율 및 통계

### 9. 시각화 (`create_visualizations`)

- 이벤트 타입 분포 차트
- 전환 퍼널 시각화
- 시간대별/요일별 트렌드 차트
- 사용자 행동 패턴 시각화
- 아이템 인기도 분포

## 결과 파일

### 생성되는 파일

- **`retailrocket_recommendation_report.html`**: 통합 분석 리포트 (인터랙티브 시각화 포함)

이 HTML 파일에는 다음 내용이 포함됩니다:

- 데이터 기본 통계
- 이벤트 분석 결과
- 아이템 및 카테고리 분석
- 사용자 행동 패턴 분석
- 이상치 탐지 결과
- 인터랙티브 차트 및 시각화

## 사용 예시

### 전체 EDA 실행

```python
from EDA.EDA import RetailRocketEDA

# EDA 객체 생성
eda = RetailRocketEDA()

# 전체 분석 실행
results = eda.run_complete_eda()
```

### 개별 분석 실행

```python
from EDA.EDA import RetailRocketEDA

eda = RetailRocketEDA()

# 데이터 로딩
eda.load_data()

# 특정 분석만 실행
eda.basic_data_overview()
eda.analyze_events()
eda.analyze_user_behavior_patterns()

# 결과 확인
print(eda.results)
```

### 커스텀 분석

```python
from EDA.EDA import RetailRocketEDA

eda = RetailRocketEDA()
eda.load_data()

# 특정 분석만 실행하고 결과 활용
event_analysis = eda.analyze_events()
user_patterns = eda.analyze_user_behavior_patterns()

# 결과를 활용한 추가 분석
# ...
```

## 분석 결과 활용

EDA 결과는 추천 시스템 모델 구축에 다음과 같이 활용할 수 있습니다:

1. **데이터 전처리 전략 수립**: 이상치 탐지 결과를 바탕으로 데이터 정제 계획
2. **특성 엔지니어링**: 사용자/아이템 특성 분석 결과를 활용한 피처 설계
3. **모델 선택**: 사용자 행동 패턴에 맞는 추천 알고리즘 선택
4. **하이퍼파라미터 튜닝**: 데이터 특성을 고려한 초기 파라미터 설정

## 문제 해결

### 데이터 파일을 찾을 수 없습니다

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/events.csv'
```

**해결 방법**:
- `data/` 폴더에 필요한 CSV 파일들이 있는지 확인하세요
- 파일 경로가 올바른지 확인하세요 (상대 경로 기준: 프로젝트 루트에서 실행)

### 메모리 부족 오류

**해결 방법**:
- 데이터 샘플링을 사용하여 분석 범위를 줄이세요
- 큰 데이터셋의 경우 청크 단위로 처리하는 것을 고려하세요

### 시각화가 표시되지 않습니다

**해결 방법**:
- Plotly가 제대로 설치되었는지 확인: `pip install plotly`
- HTML 파일을 브라우저에서 직접 열어보세요
- Jupyter Notebook 환경에서는 `plotly.offline.init_notebook_mode()` 사용

### 한글 폰트 문제

**해결 방법**:
- matplotlib 한글 폰트 설정이 필요할 수 있습니다
- 현재는 DejaVu Sans 폰트를 사용하고 있습니다
- 한글 표시가 필요한 경우 폰트 설정을 수정하세요

## 참고사항

- 전체 EDA 실행은 데이터 크기에 따라 시간이 오래 걸릴 수 있습니다
- 대용량 데이터의 경우 메모리 사용량에 주의하세요
- HTML 리포트는 인터랙티브하므로 브라우저에서 열어 확인하는 것을 권장합니다

---

**문의사항이나 버그 리포트는 이슈를 통해 제출해주세요.**

