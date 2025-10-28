"""
RetailRocket 데이터셋 EDA (탐색적 데이터 분석)
목표: 사용자 행동 기반 상품 추천 시스템을 위한 데이터 분석
- 사용자가 클릭/장바구니/구매한 상품 목록을 기반으로 비슷한 제품 추천
- 세션 기반 추천 시스템 구축을 위한 데이터 특성 파악
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
import networkx as nx
from collections import Counter
import json
import os

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RetailRocketEDA:
    def __init__(self):
        self.events_df = None
        self.item_properties_df = None
        self.category_tree_df = None
        self.results = {}
        
    def load_data(self):
        """데이터 로딩"""
        print("데이터 로딩 중...")
        
        # Events 데이터 로딩
        self.events_df = pd.read_csv('data/events.csv')
        
        # Item properties 데이터 로딩 (part1, part2 합치기)
        item_props1 = pd.read_csv('data/item_properties_part1.csv')
        item_props2 = pd.read_csv('data/item_properties_part2.csv')
        self.item_properties_df = pd.concat([item_props1, item_props2], ignore_index=True)
        
        # Category tree 데이터 로딩
        self.category_tree_df = pd.read_csv('data/category_tree.csv')
        
        print("데이터 로딩 완료!")
        
    def basic_data_overview(self):
        """1️⃣ 데이터 로딩 및 기본 구조 파악"""
        print("\n" + "="*50)
        print("1. 데이터 기본 구조 파악")
        print("="*50)
        
        datasets = {
            'events': self.events_df,
            'item_properties': self.item_properties_df,
            'category_tree': self.category_tree_df
        }
        
        overview_data = []
        
        for name, df in datasets.items():
            memory_mb = df.memory_usage(deep=True).sum() / 1e6
            overview_data.append({
                'Dataset': name,
                'Rows': df.shape[0],
                'Columns': df.shape[1],
                'Memory (MB)': round(memory_mb, 2),
                'Columns': list(df.columns)
            })
            
            print(f"\n{name.upper()}")
            print(f"   Shape: {df.shape}")
            print(f"   Memory: {memory_mb:.2f} MB")
            print(f"   Columns: {list(df.columns)}")
            
            # 결측치 확인
            if name == 'events':
                print(f"   Missing values:")
                for col in df.columns:
                    missing_pct = (df[col].isnull().sum() / len(df)) * 100
                    print(f"     {col}: {missing_pct:.2f}%")
        
        self.results['basic_overview'] = overview_data
        
        # Events 데이터 타입 및 샘플 확인
        print(f"\nEvents 데이터 샘플:")
        print(self.events_df.head())
        print(f"\nEvents 데이터 타입:")
        print(self.events_df.dtypes)
        
        return overview_data
    
    def analyze_events(self):
        """2️⃣ 이벤트 로그(events.csv) 분석"""
        print("\n" + "="*50)
        print("2. 이벤트 로그 분석")
        print("="*50)
        
        # Timestamp를 datetime으로 변환
        self.events_df['datetime'] = pd.to_datetime(self.events_df['timestamp'], unit='ms')
        self.events_df['date'] = self.events_df['datetime'].dt.date
        self.events_df['hour'] = self.events_df['datetime'].dt.hour
        self.events_df['dayofweek'] = self.events_df['datetime'].dt.dayofweek
        self.events_df['day_name'] = self.events_df['datetime'].dt.day_name()
        
        # (1) 이벤트 타입 분포
        print("\n이벤트 타입 분포:")
        event_counts = self.events_df['event'].value_counts()
        print(event_counts)
        
        # 전환 퍼널 계산
        total_views = event_counts.get('view', 0)
        total_addtocart = event_counts.get('addtocart', 0)
        total_transactions = event_counts.get('transaction', 0)
        
        view_to_cart_rate = (total_addtocart / total_views * 100) if total_views > 0 else 0
        cart_to_transaction_rate = (total_transactions / total_addtocart * 100) if total_addtocart > 0 else 0
        view_to_transaction_rate = (total_transactions / total_views * 100) if total_views > 0 else 0
        
        print(f"\n전환 퍼널:")
        print(f"   View → AddToCart: {view_to_cart_rate:.2f}%")
        print(f"   AddToCart → Transaction: {cart_to_transaction_rate:.2f}%")
        print(f"   View → Transaction: {view_to_transaction_rate:.2f}%")
        
        # (2) 시간 기반 분석
        print(f"\n시간 기반 분석:")
        print(f"   데이터 기간: {self.events_df['datetime'].min()} ~ {self.events_df['datetime'].max()}")
        print(f"   총 일수: {(self.events_df['datetime'].max() - self.events_df['datetime'].min()).days}일")
        
        # 일별 이벤트 수
        daily_events = self.events_df.groupby('date')['event'].count()
        print(f"   일평균 이벤트 수: {daily_events.mean():.0f}")
        
        # (3) 사용자 행동 수준
        unique_visitors = self.events_df['visitorid'].nunique()
        total_events = len(self.events_df)
        avg_events_per_visitor = total_events / unique_visitors
        
        print(f"\n사용자 행동 수준:")
        print(f"   고유 방문자 수: {unique_visitors:,}")
        print(f"   총 이벤트 수: {total_events:,}")
        print(f"   방문자당 평균 이벤트 수: {avg_events_per_visitor:.2f}")
        
        # (4) 아이템 상호작용 수준
        unique_items = self.events_df['itemid'].nunique()
        print(f"\n아이템 상호작용 수준:")
        print(f"   고유 아이템 수: {unique_items:,}")
        
        # 아이템별 이벤트 수 (상위 20개)
        item_popularity = self.events_df['itemid'].value_counts().head(20)
        print(f"   상위 20개 아이템의 이벤트 수:")
        print(item_popularity)
        
        # Long-tail 분포 분석
        item_view_counts = self.events_df[self.events_df['event'] == 'view']['itemid'].value_counts()
        total_item_views = item_view_counts.sum()
        top_1_percent_items = int(len(item_view_counts) * 0.01)
        top_1_percent_views = item_view_counts.head(top_1_percent_items).sum()
        concentration_ratio = (top_1_percent_views / total_item_views) * 100
        
        print(f"\nLong-tail 분포:")
        print(f"   상위 1% 아이템이 전체 조회의 {concentration_ratio:.2f}% 차지")
        
        # 결과 저장
        self.results['events_analysis'] = {
            'event_counts': event_counts.to_dict(),
            'conversion_funnel': {
                'view_to_cart_rate': view_to_cart_rate,
                'cart_to_transaction_rate': cart_to_transaction_rate,
                'view_to_transaction_rate': view_to_transaction_rate
            },
            'user_stats': {
                'unique_visitors': unique_visitors,
                'total_events': total_events,
                'avg_events_per_visitor': avg_events_per_visitor
            },
            'item_stats': {
                'unique_items': unique_items,
                'concentration_ratio': concentration_ratio
            }
        }
        
        return self.results['events_analysis']
    
    def analyze_item_properties(self):
        """3️⃣ 아이템 속성(item_properties.csv) 분석"""
        print("\n" + "="*50)
        print("3. 아이템 속성 분석")
        print("="*50)
        
        # (1) Property 종류 파악
        print("\nProperty 종류:")
        property_counts = self.item_properties_df['property'].value_counts()
        print(property_counts.head(20))
        
        # Category ID 매핑
        category_mapping = self.item_properties_df[
            self.item_properties_df['property'] == 'categoryid'
        ][['itemid', 'value']].rename(columns={'value': 'categoryid'})
        category_mapping['categoryid'] = category_mapping['categoryid'].astype(int)
        
        print(f"\n카테고리 매핑:")
        print(f"   카테고리가 있는 아이템 수: {len(category_mapping)}")
        print(f"   고유 카테고리 수: {category_mapping['categoryid'].nunique()}")
        
        # Available 상태 분석
        available_data = self.item_properties_df[
            self.item_properties_df['property'] == 'available'
        ].copy()
        available_data['value'] = available_data['value'].astype(int)
        
        if len(available_data) > 0:
            available_ratio = (available_data['value'] == 1).mean() * 100
            print(f"\n재고 상태:")
            print(f"   재고 있는 아이템 비율: {available_ratio:.2f}%")
        
        # 카테고리별 아이템 수
        category_item_counts = category_mapping['categoryid'].value_counts()
        print(f"\n카테고리별 아이템 수 (상위 10개):")
        print(category_item_counts.head(10))
        
        # Events와 카테고리 매핑하여 분석
        events_with_category = self.events_df.merge(category_mapping, on='itemid', how='left')
        category_event_counts = events_with_category['categoryid'].value_counts().head(20)
        
        print(f"\n카테고리별 이벤트 수 (상위 20개):")
        print(category_event_counts)
        
        # 결과 저장
        self.results['item_properties_analysis'] = {
            'property_counts': property_counts.head(20).to_dict(),
            'category_stats': {
                'total_items_with_category': len(category_mapping),
                'unique_categories': category_mapping['categoryid'].nunique(),
                'top_categories_by_items': category_item_counts.head(10).to_dict(),
                'top_categories_by_events': category_event_counts.head(20).to_dict()
            }
        }
        
        return self.results['item_properties_analysis']
    
    def analyze_category_tree(self):
        """4️⃣ 카테고리 트리(category_tree.csv) 분석"""
        print("\n" + "="*50)
        print("4. 카테고리 트리 분석")
        print("="*50)
        
        # 기본 통계
        print(f"\n카테고리 트리 기본 정보:")
        print(f"   총 노드 수: {len(self.category_tree_df)}")
        print(f"   고유 카테고리 수: {self.category_tree_df['categoryid'].nunique()}")
        
        # 루트 노드 찾기 (parentid가 NaN인 경우)
        root_categories = self.category_tree_df[self.category_tree_df['parentid'].isna()]
        print(f"   루트 카테고리 수: {len(root_categories)}")
        
        # 트리 깊이 계산
        def calculate_depth(category_id, depth=0, visited=None):
            if visited is None:
                visited = set()
            if category_id in visited:
                return depth  # 순환 참조 방지
            
            visited.add(category_id)
            children = self.category_tree_df[self.category_tree_df['parentid'] == category_id]
            
            if len(children) == 0:
                return depth
            
            max_child_depth = 0
            for _, child in children.iterrows():
                child_depth = calculate_depth(child['categoryid'], depth + 1, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        # 각 루트에서 시작하여 최대 깊이 계산
        max_depths = []
        for _, root in root_categories.iterrows():
            depth = calculate_depth(root['categoryid'])
            max_depths.append(depth)
        
        if max_depths:
            print(f"   최대 트리 깊이: {max(max_depths)}")
            print(f"   평균 트리 깊이: {np.mean(max_depths):.2f}")
        
        # 각 카테고리의 자식 수
        children_counts = self.category_tree_df['parentid'].value_counts()
        print(f"\n카테고리별 자식 수 (상위 10개):")
        print(children_counts.head(10))
        
        # 결과 저장
        self.results['category_tree_analysis'] = {
            'total_nodes': len(self.category_tree_df),
            'unique_categories': self.category_tree_df['categoryid'].nunique(),
            'root_categories': len(root_categories),
            'max_depth': max(max_depths) if max_depths else 0,
            'avg_depth': np.mean(max_depths) if max_depths else 0,
            'children_counts': children_counts.head(10).to_dict()
        }
        
        return self.results['category_tree_analysis']
    
    def analyze_user_behavior_patterns(self):
        """5️⃣ 사용자 행동 패턴 분석 (추천 시스템 특화)"""
        print("\n" + "="*50)
        print("5. 사용자 행동 패턴 분석")
        print("="*50)
        
        # 사용자별 행동 패턴 분석
        user_behavior = self.events_df.groupby(['visitorid', 'event']).size().unstack(fill_value=0)
        
        # 행동 유형별 사용자 분류
        def classify_user_behavior(row):
            views = row.get('view', 0)
            carts = row.get('addtocart', 0)
            transactions = row.get('transaction', 0)
            
            if transactions > 0:
                return 'buyer'
            elif carts > 0:
                return 'cart_user'
            elif views > 0:
                return 'browser'
            else:
                return 'inactive'
        
        user_behavior['user_type'] = user_behavior.apply(classify_user_behavior, axis=1)
        user_type_counts = user_behavior['user_type'].value_counts()
        
        print(f"\n사용자 유형 분포:")
        for user_type, count in user_type_counts.items():
            percentage = (count / len(user_behavior)) * 100
            print(f"   {user_type}: {count:,} ({percentage:.2f}%)")
        
        # 사용자별 선호 카테고리 분석
        category_mapping = self.item_properties_df[
            self.item_properties_df['property'] == 'categoryid'
        ][['itemid', 'value']].rename(columns={'value': 'categoryid'})
        category_mapping['categoryid'] = category_mapping['categoryid'].astype(int)
        
        events_with_category = self.events_df.merge(category_mapping, on='itemid', how='left')
        
        # 사용자별 카테고리 선호도
        user_category_preference = events_with_category.groupby(['visitorid', 'categoryid']).size().reset_index(name='interaction_count')
        user_category_preference = user_category_preference.sort_values(['visitorid', 'interaction_count'], ascending=[True, False])
        
        # 각 사용자의 최고 선호 카테고리
        top_categories_per_user = user_category_preference.groupby('visitorid').first()
        
        print(f"\n사용자별 카테고리 선호도:")
        print(f"   카테고리 선호도가 있는 사용자: {len(top_categories_per_user):,}")
        
        # 결과 저장
        self.results['user_behavior_analysis'] = {
            'user_type_distribution': user_type_counts.to_dict(),
            'users_with_category_preference': len(top_categories_per_user),
            'user_category_data': user_category_preference
        }
        
        return self.results['user_behavior_analysis']
    
    def analyze_item_similarity_features(self):
        """6️⃣ 아이템 유사도 분석을 위한 특성 추출"""
        print("\n" + "="*50)
        print("6. 아이템 유사도 특성 분석")
        print("="*50)
        
        # 아이템별 상호작용 통계
        item_stats = self.events_df.groupby('itemid').agg({
            'visitorid': 'nunique',  # 고유 사용자 수
            'event': 'count',        # 총 상호작용 수
            'datetime': ['min', 'max']  # 첫/마지막 상호작용
        }).reset_index()
        
        item_stats.columns = ['itemid', 'unique_users', 'total_interactions', 'first_interaction', 'last_interaction']
        
        # 아이템별 이벤트 타입별 상호작용
        item_event_stats = self.events_df.groupby(['itemid', 'event']).size().unstack(fill_value=0)
        item_event_stats.columns = [f'{col}_count' for col in item_event_stats.columns]
        
        # 아이템 통계와 이벤트 통계 결합
        item_features = item_stats.merge(item_event_stats, left_on='itemid', right_index=True, how='left')
        
        # 아이템 인기도 점수 계산
        item_features['popularity_score'] = (
            item_features['view_count'] * 1 +
            item_features['addtocart_count'] * 3 +
            item_features['transaction_count'] * 10
        )
        
        # 아이템 전환율 계산
        item_features['conversion_rate'] = (
            item_features['transaction_count'] / item_features['view_count'].replace(0, 1) * 100
        )
        
        # 카테고리 정보 추가
        category_mapping = self.item_properties_df[
            self.item_properties_df['property'] == 'categoryid'
        ][['itemid', 'value']].rename(columns={'value': 'categoryid'})
        category_mapping['categoryid'] = category_mapping['categoryid'].astype(int)
        
        item_features = item_features.merge(category_mapping, on='itemid', how='left')
        
        print(f"\n아이템 특성 분석:")
        print(f"   분석된 아이템 수: {len(item_features):,}")
        print(f"   카테고리가 있는 아이템: {item_features['categoryid'].notna().sum():,}")
        
        # 인기 아이템 상위 20개
        top_items = item_features.nlargest(20, 'popularity_score')
        print(f"\n상위 20개 인기 아이템:")
        print(top_items[['itemid', 'popularity_score', 'conversion_rate', 'categoryid']].head(10))
        
        # 카테고리별 아이템 특성
        if 'categoryid' in item_features.columns:
            category_stats = item_features.groupby('categoryid').agg({
                'popularity_score': 'mean',
                'conversion_rate': 'mean',
                'unique_users': 'mean',
                'itemid': 'count'
            }).reset_index()
            category_stats.columns = ['categoryid', 'avg_popularity', 'avg_conversion_rate', 'avg_unique_users', 'item_count']
            
            print(f"\n카테고리별 평균 특성 (상위 10개):")
            top_categories = category_stats.nlargest(10, 'avg_popularity')
            print(top_categories)
        
        # 결과 저장
        self.results['item_similarity_analysis'] = {
            'total_items_analyzed': len(item_features),
            'items_with_category': item_features['categoryid'].notna().sum(),
            'top_items': top_items[['itemid', 'popularity_score', 'conversion_rate', 'categoryid']].to_dict('records'),
            'item_features': item_features
        }
        
        return self.results['item_similarity_analysis']
    
    def analyze_session_based_recommendations(self):
        """7️⃣ 세션 기반 추천을 위한 세션 분석"""
        print("\n" + "="*50)
        print("7. 세션 기반 추천 분석")
        print("="*50)
        
        # 세션 정의: 같은 visitorid의 연속 이벤트를 30분 기준으로 끊기
        session_data = self.events_df.copy()
        session_data = session_data.sort_values(['visitorid', 'datetime'])
        
        # 시간 차이 계산 (분 단위)
        session_data['time_diff'] = session_data.groupby('visitorid')['datetime'].diff().dt.total_seconds() / 60
        
        # 세션 시작점 식별 (첫 이벤트이거나 30분 이상 간격)
        session_data['is_session_start'] = (
            session_data['time_diff'].isna() | 
            (session_data['time_diff'] > 30)
        )
        
        # 세션 ID 생성
        session_data['session_id'] = session_data.groupby('visitorid')['is_session_start'].cumsum()
        
        # 세션별 통계
        session_stats = session_data.groupby(['visitorid', 'session_id']).agg({
            'datetime': ['min', 'max', 'count'],
            'itemid': ['nunique', lambda x: list(x)],
            'event': lambda x: list(x)
        }).reset_index()
        
        session_stats.columns = ['visitorid', 'session_id', 'session_start', 'session_end', 'event_count', 'unique_items', 'item_sequence', 'event_sequence']
        
        # 세션 길이 (분)
        session_stats['session_duration_minutes'] = (
            session_stats['session_end'] - session_stats['session_start']
        ).dt.total_seconds() / 60
        
        # 세션별 행동 패턴 분석
        def analyze_session_pattern(events):
            events_str = ' '.join(events)
            if 'transaction' in events_str:
                return 'conversion'
            elif 'addtocart' in events_str:
                return 'cart_only'
            else:
                return 'view_only'
        
        session_stats['session_pattern'] = session_stats['event_sequence'].apply(analyze_session_pattern)
        
        # 세션 길이별 분포
        session_length_distribution = session_stats['event_count'].value_counts().sort_index()
        
        print(f"\n세션 통계:")
        print(f"   총 세션 수: {len(session_stats):,}")
        print(f"   평균 세션 길이: {session_stats['event_count'].mean():.2f} 이벤트")
        print(f"   평균 세션 시간: {session_stats['session_duration_minutes'].mean():.2f} 분")
        print(f"   평균 세션당 고유 아이템: {session_stats['unique_items'].mean():.2f}")
        
        # 세션 패턴 분포
        pattern_counts = session_stats['session_pattern'].value_counts()
        print(f"\n세션 패턴 분포:")
        for pattern, count in pattern_counts.items():
            percentage = (count / len(session_stats)) * 100
            print(f"   {pattern}: {count:,} ({percentage:.2f}%)")
        
        # 세션 길이별 전환율 분석
        conversion_by_length = session_stats.groupby('event_count')['session_pattern'].apply(
            lambda x: (x == 'conversion').mean() * 100
        ).reset_index()
        conversion_by_length.columns = ['session_length', 'conversion_rate']
        
        print(f"\n세션 길이별 전환율 (상위 10개):")
        print(conversion_by_length.head(10))
        
        # 아이템 시퀀스 분석 (연관성 분석을 위해)
        def extract_item_transitions(item_sequence):
            transitions = []
            for i in range(len(item_sequence) - 1):
                transitions.append((item_sequence[i], item_sequence[i+1]))
            return transitions
        
        # 모든 세션의 아이템 전환 추출
        all_transitions = []
        for item_seq in session_stats['item_sequence']:
            if len(item_seq) > 1:
                transitions = extract_item_transitions(item_seq)
                all_transitions.extend(transitions)
        
        # 가장 빈번한 아이템 전환 (상위 20개)
        if all_transitions:
            transition_counts = pd.Series(all_transitions).value_counts().head(20)
            print(f"\n가장 빈번한 아이템 전환 (상위 10개):")
            for (from_item, to_item), count in transition_counts.head(10).items():
                print(f"   {from_item} → {to_item}: {count}회")
        
        # 결과 저장
        self.results['session_based_analysis'] = {
            'total_sessions': len(session_stats),
            'avg_session_length': session_stats['event_count'].mean(),
            'avg_session_duration': session_stats['session_duration_minutes'].mean(),
            'avg_unique_items_per_session': session_stats['unique_items'].mean(),
            'session_patterns': pattern_counts.to_dict(),
            'conversion_by_length': conversion_by_length.head(10).to_dict('records'),
            'frequent_transitions': transition_counts.head(20).to_dict() if all_transitions else {}
        }
        
        return self.results['session_based_analysis']
    
    def detect_anomalies(self):
        """8️⃣ 이상 사용자 탐지"""
        print("\n" + "="*50)
        print("8. 이상 사용자 탐지")
        print("="*50)
        
        # 사용자별 통계
        user_stats = self.events_df.groupby('visitorid').agg({
            'event': 'count',
            'itemid': 'nunique',
            'datetime': ['min', 'max']
        }).reset_index()
        
        user_stats.columns = ['visitorid', 'total_events', 'unique_items', 'first_event', 'last_event']
        
        # 사용자 활동 기간 (일)
        user_stats['activity_days'] = (
            user_stats['last_event'] - user_stats['first_event']
        ).dt.total_seconds() / (24 * 3600)
        
        # 일평균 이벤트 수
        user_stats['daily_avg_events'] = user_stats['total_events'] / (user_stats['activity_days'] + 1)
        
        # 이상치 탐지 (IQR 방법)
        def detect_outliers_iqr(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data[column] < lower_bound) | (data[column] > upper_bound)
        
        # 각 지표별 이상치 탐지
        user_stats['outlier_total_events'] = detect_outliers_iqr(user_stats, 'total_events')
        user_stats['outlier_daily_avg_events'] = detect_outliers_iqr(user_stats, 'daily_avg_events')
        user_stats['outlier_unique_items'] = detect_outliers_iqr(user_stats, 'unique_items')
        
        # 종합 이상치 점수
        user_stats['anomaly_score'] = (
            user_stats['outlier_total_events'].astype(int) +
            user_stats['outlier_daily_avg_events'].astype(int) +
            user_stats['outlier_unique_items'].astype(int)
        )
        
        # 이상치 사용자 식별
        anomaly_users = user_stats[user_stats['anomaly_score'] >= 2]
        
        print(f"\n이상치 탐지 결과:")
        print(f"   총 사용자 수: {len(user_stats):,}")
        print(f"   이상치 사용자 수: {len(anomaly_users):,}")
        print(f"   이상치 비율: {(len(anomaly_users) / len(user_stats)) * 100:.2f}%")
        
        if len(anomaly_users) > 0:
            print(f"\n이상치 사용자 샘플 (상위 10개):")
            top_anomalies = anomaly_users.nlargest(10, 'total_events')
            print(top_anomalies[['visitorid', 'total_events', 'unique_items', 'daily_avg_events', 'anomaly_score']])
        
        # 결과 저장
        self.results['anomaly_detection'] = {
            'total_users': len(user_stats),
            'anomaly_users': len(anomaly_users),
            'anomaly_ratio': (len(anomaly_users) / len(user_stats)) * 100,
            'top_anomalies': top_anomalies[['visitorid', 'total_events', 'unique_items', 'daily_avg_events', 'anomaly_score']].to_dict('records') if len(anomaly_users) > 0 else []
        }
        
        return self.results['anomaly_detection']
    
    def create_visualizations(self):
        """9️⃣ 통합 시각화 (추천 시스템 특화)"""
        print("\n" + "="*50)
        print("9. 시각화 생성")
        print("="*50)
        
        # Plotly를 사용한 인터랙티브 시각화 (추천 시스템 특화)
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('이벤트 타입 분포\n\n', '사용자 유형 분포\n\n', 
                          '시간대별 이벤트 분포', '아이템 인기도 분포',
                          '전환 퍼널', '세션 패턴 분포',
                          '카테고리별 이벤트 수', '세션 길이별 전환율'),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "funnel"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. 이벤트 타입 분포 (파이 차트)
        event_counts = self.events_df['event'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=event_counts.index, 
                values=event_counts.values, 
                name="이벤트 타입",
                hovertemplate="<b>%{label}</b><br>개수: %{value:,}<br>비율: %{percent}<extra></extra>",
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        # 2. 사용자 유형 분포 (파이 차트)
        if 'user_behavior_analysis' in self.results:
            user_type_dist = self.results['user_behavior_analysis']['user_type_distribution']
        fig.add_trace(
                go.Pie(
                    labels=list(user_type_dist.keys()), 
                    values=list(user_type_dist.values()), 
                    name="사용자 유형",
                    hovertemplate="<b>%{label}</b><br>사용자 수: %{value:,}<br>비율: %{percent}<extra></extra>",
                    textinfo='label+percent'
                ),
            row=1, col=2
        )
        
        # 3. 시간대별 이벤트 분포
        hourly_events = self.events_df.groupby('hour')['event'].count()
        fig.add_trace(
            go.Bar(
                x=hourly_events.index, 
                y=hourly_events.values, 
                name="시간대별 이벤트",
                hovertemplate="<b>%{x}시</b><br>이벤트 수: %{y:,}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # 4. 아이템 인기도 분포 (로그 스케일)
        item_counts = self.events_df['itemid'].value_counts()
        fig.add_trace(
            go.Histogram(
                x=np.log10(item_counts.values), 
                name="아이템 인기도 (로그)",
                hovertemplate="<b>로그10(조회수)</b><br>아이템 수: %{y}<extra></extra>",
                nbinsx=30
            ),
            row=2, col=2
        )
        
        # 5. 전환 퍼널
        funnel_data = [
            ('View', event_counts.get('view', 0)),
            ('AddToCart', event_counts.get('addtocart', 0)),
            ('Transaction', event_counts.get('transaction', 0))
        ]
        fig.add_trace(
            go.Funnel(
                y=[x[0] for x in funnel_data], 
                x=[x[1] for x in funnel_data], 
                name="전환 퍼널",
                hovertemplate="<b>%{y}</b><br>사용자 수: %{x:,}<extra></extra>"
            ),
            row=3, col=1
        )
        
        # 6. 세션 패턴 분포
        if 'session_based_analysis' in self.results:
            session_patterns = self.results['session_based_analysis']['session_patterns']
            fig.add_trace(
                go.Bar(
                    x=list(session_patterns.keys()), 
                    y=list(session_patterns.values()), 
                    name="세션 패턴",
                    hovertemplate="<b>%{x}</b><br>세션 수: %{y:,}<extra></extra>"
                ),
                row=3, col=2
            )
        
        # 7. 카테고리별 이벤트 수 (상위 20개)
        category_mapping = self.item_properties_df[
            self.item_properties_df['property'] == 'categoryid'
        ][['itemid', 'value']].rename(columns={'value': 'categoryid'})
        category_mapping['categoryid'] = category_mapping['categoryid'].astype(int)
        
        events_with_category = self.events_df.merge(category_mapping, on='itemid', how='left')
        category_event_counts = events_with_category['categoryid'].value_counts().head(20)
        
        fig.add_trace(
            go.Bar(
                x=[str(x) for x in category_event_counts.index], 
                y=category_event_counts.values, 
                name="카테고리별 이벤트",
                hovertemplate="<b>카테고리 %{x}</b><br>이벤트 수: %{y:,}<extra></extra>"
            ),
            row=4, col=1
        )
        
        # 8. 세션 길이별 전환율
        if 'session_based_analysis' in self.results and 'conversion_by_length' in self.results['session_based_analysis']:
            conversion_data = self.results['session_based_analysis']['conversion_by_length']
            if conversion_data:
                session_lengths = [x['session_length'] for x in conversion_data[:10]]
                conversion_rates = [x['conversion_rate'] for x in conversion_data[:10]]
                fig.add_trace(
                    go.Scatter(
                        x=session_lengths, 
                        y=conversion_rates, 
                        mode='lines+markers', 
                        name="세션 길이별 전환율",
                        hovertemplate="<b>세션 길이: %{x} 이벤트</b><br>전환율: %{y:.2f}%<extra></extra>",
                        line=dict(width=3),
                        marker=dict(size=8)
                    ),
                    row=4, col=2
                )
        
        # 전체 레이아웃 업데이트
        fig.update_layout(
            height=1600, 
            showlegend=False, 
            title_text="RetailRocket 추천 시스템 EDA 시각화",
            title_x=0.5,
            title_font_size=20
        )
        
        # 각 서브플롯에 축 레이블과 제목 추가
        fig.update_xaxes(title_text="시간 (시)", row=2, col=1)
        fig.update_yaxes(title_text="이벤트 수", row=2, col=1)
        
        fig.update_xaxes(title_text="로그10(조회수)", row=2, col=2)
        fig.update_yaxes(title_text="아이템 수", row=2, col=2)
        
        fig.update_xaxes(title_text="이벤트 수", row=3, col=1)
        fig.update_yaxes(title_text="사용자 수", row=3, col=1)
        
        fig.update_xaxes(title_text="세션 패턴", row=3, col=2)
        fig.update_yaxes(title_text="세션 수", row=3, col=2)
        
        fig.update_xaxes(title_text="카테고리 ID", row=4, col=1)
        fig.update_yaxes(title_text="이벤트 수", row=4, col=1)
        
        fig.update_xaxes(title_text="세션 길이 (이벤트 수)", row=4, col=2)
        fig.update_yaxes(title_text="전환율 (%)", row=4, col=2)
        
        return fig
    
    def create_category_tree_visualization(self):
        """카테고리 트리 네트워크 시각화"""
        print("\n" + "="*50)
        print("카테고리 트리 시각화 생성")
        print("="*50)
        
        # 카테고리 트리 데이터 준비
        category_tree = self.category_tree_df.copy()
        
        # NaN 값 제거 (루트 노드들)
        category_tree = category_tree.dropna()
        
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        
        # 엣지 추가 (parentid -> categoryid)
        for _, row in category_tree.iterrows():
            G.add_edge(row['parentid'], row['categoryid'])
        
        # 노드 수가 너무 많으면 상위 연결 노드들만 선택
        if len(G.nodes()) > 200:
            # 연결 수가 많은 상위 노드들 선택
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:150]
            top_node_ids = [node[0] for node in top_nodes]
            
            # 선택된 노드들과 연결된 서브그래프 생성
            subgraph_nodes = set(top_node_ids)
            for node in top_node_ids:
                subgraph_nodes.update(list(G.successors(node)))
                subgraph_nodes.update(list(G.predecessors(node)))
            
            G = G.subgraph(subgraph_nodes)
        
        # 레이아웃 계산 (계층적 레이아웃 사용)
        try:
            # 계층적 레이아웃 시도
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # 실패하면 spring layout 사용 (더 넓은 간격)
            pos = nx.spring_layout(G, k=5, iterations=100)
        
        # Plotly 네트워크 그래프 생성
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # 엣지 트레이스
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='rgba(0,0,0,0.6)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 노드 트레이스
        node_x = []
        node_y = []
        node_text = []
        node_hovertext = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 노드 정보 계산
            neighbors = list(G.neighbors(node))
            children = list(G.successors(node))
            parents = list(G.predecessors(node))
            degree = len(neighbors)
            
            # 노드 크기 설정 (연결 수에 따라, 최소/최대 크기 제한)
            node_size = max(8, min(25, 8 + degree * 0.5))
            node_sizes.append(node_size)
            
            # 중요한 노드만 텍스트 표시
            if degree >= 5:  # 연결이 많은 노드만 라벨 표시
                node_text.append(str(node))
            else:
                node_text.append("")
            
            hovertext = f"카테고리 ID: {node}<br>"
            hovertext += f"자식 카테고리: {len(children)}개<br>"
            hovertext += f"부모 카테고리: {len(parents)}개<br>"
            hovertext += f"총 연결: {len(neighbors)}개"
            
            node_hovertext.append(hovertext)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertext=node_hovertext,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=False,
                color=[],
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title="연결 수",
                    xanchor="left",
                    len=0.5
                ),
                line=dict(width=1, color='white'),
                opacity=0.8
            )
        )
        
        # 노드 색상 설정 (연결 수에 따라)
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
        
        node_trace.marker.color = node_adjacencies
        
        # 그래프 생성
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text='RetailRocket 카테고리 트리 네트워크',
                    x=0.5,
                    font=dict(size=18)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=40,l=20,r=20,t=60),
                annotations=[ dict(
                    text=f"총 {len(G.nodes())}개 카테고리, {len(G.edges())}개 연결<br>연결이 많은 주요 카테고리만 라벨 표시",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1,
                    xanchor="center", yanchor="top",
                    font=dict(color="gray", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                width=1200,
                height=900
            )
        )
        
        return fig
    
    def generate_html_report(self):
        """🔟 통합 HTML 리포트 생성 (추천 시스템 특화)"""
        print("\n" + "="*50)
        print("10. 통합 HTML 리포트 생성")
        print("="*50)
        
        # 먼저 시각화 생성
        fig = self.create_visualizations()
        
        # 카테고리 트리 시각화 생성
        category_fig = self.create_category_tree_visualization()
        
        # HTML 템플릿
        html_template = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🛍️ RetailRocket 추천 시스템 EDA 리포트</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #2c3e50;
                    margin-top: 25px;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .conversion-funnel {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .funnel-step {{
                    background: #3498db;
                    color: white;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    text-align: center;
                    position: relative;
                }}
                .funnel-step::after {{
                    content: '';
                    position: absolute;
                    bottom: -10px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 0;
                    height: 0;
                    border-left: 10px solid transparent;
                    border-right: 10px solid transparent;
                    border-top: 10px solid #3498db;
                }}
                .funnel-step:last-child::after {{
                    display: none;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .data-table th, .data-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .data-table th {{
                    background-color: #3498db;
                    color: white;
                }}
                .data-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .warning-box {{
                    background: #fdf2e9;
                    border-left: 4px solid #e67e22;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .code-block {{
                    background: #2c3e50;
                    color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    font-family: 'Courier New', monospace;
                    overflow-x: auto;
                }}
                .visualization-container {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    border: 1px solid #e9ecef;
                }}
                .plotly-graph-div {{
                    width: 100%;
                    height: 800px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>RetailRocket 추천 시스템 EDA 리포트</h1>
                <p style="text-align: center; color: #7f8c8d; font-size: 1.1em;">
                    <strong>목표:</strong> 사용자 행동 기반 상품 추천 시스템 구축을 위한 데이터 분석<br>
                    <strong>핵심 기능:</strong> 클릭/장바구니/구매 이력을 기반으로 한 비슷한 제품 추천
                </p>
                
                {content}
                
                <div style="text-align: center; margin-top: 50px; color: #7f8c8d;">
                    <p>📊 리포트 생성 시간: {timestamp}</p>
                    <p>🔍 분석 도구: Python, Pandas, Plotly</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 리포트 내용 생성
        content = self._generate_report_content()
        
        # 시각화 HTML 생성
        visualization_html = fig.to_html(include_plotlyjs=True, div_id="main-visualization")
        
        # 카테고리 트리 시각화 HTML 생성
        category_html = category_fig.to_html(include_plotlyjs=False, div_id="category-tree")
        
        # 상세 분석 결과 텍스트 추가
        detailed_analysis = self._generate_detailed_analysis_text()
        
        # 시각화를 리포트에 통합
        content_with_viz = content + f"""
        <h2>📊 6. 상세 분석 결과</h2>
        <div class="code-block">
            <pre>{detailed_analysis}</pre>
        </div>
        
        <h2>📈 7. 통합 시각화</h2>
        <div class="visualization-container">
            <h3>추천 시스템 분석 시각화</h3>
            <div class="insight-box">
                <strong>📊 시각화 설명:</strong><br>
                • <strong>이벤트 타입 분포:</strong> 사용자 행동 유형별 비율 (view, addtocart, transaction)<br>
                • <strong>사용자 유형 분포:</strong> 행동 패턴에 따른 사용자 분류 (browser, cart_user, buyer)<br>
                • <strong>시간대별 이벤트 분포:</strong> 하루 중 사용자 활동이 가장 활발한 시간대 파악<br>
                • <strong>아이템 인기도 분포:</strong> 상품별 조회수 분포 (로그 스케일로 Long-tail 현상 확인)<br>
                • <strong>전환 퍼널:</strong> View → AddToCart → Transaction 단계별 전환율<br>
                • <strong>세션 패턴 분포:</strong> 세션별 행동 패턴 유형 (view_only, cart_only, conversion)<br>
                • <strong>카테고리별 이벤트 수:</strong> 가장 인기 있는 상품 카테고리 Top 20<br>
                • <strong>세션 길이별 전환율:</strong> 세션 길이에 따른 구매 전환율 변화 추이
            </div>
            {visualization_html}
        </div>
        
        <h2>🌳 8. 카테고리 트리 네트워크</h2>
        <div class="visualization-container">
            <h3>전체 카테고리 계층 구조</h3>
            <div class="insight-box">
                <strong>🌳 카테고리 트리 설명:</strong><br>
                • <strong>노드 필터링:</strong> 연결이 많은 주요 카테고리들만 표시<br>
                • <strong>라벨 최적화:</strong> 연결이 많은 노드(5개 이상)만 라벨 표시하여 텍스트 겹침 방지<br>
                • <strong>색상 개선:</strong> Viridis 색상 스케일로 연결 수에 따른 색상 구분<br>
                • <strong>크기 조정:</strong> 연결 수에 따라 노드 크기가 동적으로 조정 (8-25px 범위)<br>
                • <strong>엣지 투명도:</strong> 얇고 투명한 엣지로 시각적 복잡성 감소<br>
                • <strong>호버 정보:</strong> 각 카테고리의 자식/부모 카테고리 수와 총 연결 수 표시<br>
                • <strong>추천 활용:</strong> 카테고리 기반 추천 시스템에서 상위/하위 카테고리 관계 파악에 활용
            </div>
            {category_html}
        </div>
        """
        
        # HTML 파일 생성
        html_content = html_template.format(
            content=content_with_viz,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open("retailrocket_recommendation_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("통합 HTML 리포트가 'retailrocket_recommendation_report.html'에 저장되었습니다.")
        
    def _generate_report_content(self):
        """리포트 내용 생성"""
        content = ""
        
        # 1. 데이터 기본 구조
        content += "<h2>📊 1. 데이터 기본 구조</h2>"
        content += "<div class='stats-grid'>"
        
        for dataset in self.results['basic_overview']:
            content += f"""
            <div class='metric-card'>
                <div class='metric-value'>{dataset['Rows']:,}</div>
                <div class='metric-label'>{dataset['Dataset']} 행 수</div>
            </div>
            """
        
        content += "</div>"
        
        # 2. 사용자 행동 패턴 분석
        if 'user_behavior_analysis' in self.results:
            user_behavior_data = self.results['user_behavior_analysis']
            content += "<h2>👥 2. 사용자 행동 패턴 분석</h2>"
            
            # 사용자 유형 분포
            content += "<div class='conversion-funnel'>"
            content += "<h3>👤 사용자 유형 분포</h3>"
            
            user_type_dist = user_behavior_data['user_type_distribution']
            for user_type, count in user_type_dist.items():
                percentage = (count / sum(user_type_dist.values())) * 100
            content += f"""
            <div class='funnel-step'>
                    <strong>{user_type}:</strong> {count:,}명 ({percentage:.2f}%)
            </div>
            """
            content += "</div>"
            
            # 사용자 통계
            content += "<div class='stats-grid'>"
            content += f"""
            <div class='metric-card'>
                <div class='metric-value'>{sum(user_type_dist.values()):,}</div>
                <div class='metric-label'>총 사용자</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{user_behavior_data['users_with_category_preference']:,}</div>
                <div class='metric-label'>카테고리 선호도 보유 사용자</div>
            </div>
            """
            content += "</div>"
        
        # 3. 아이템 유사도 분석
        if 'item_similarity_analysis' in self.results:
            item_data = self.results['item_similarity_analysis']
            content += "<h2>🎯 3. 아이템 유사도 특성 분석</h2>"
            
            content += "<div class='stats-grid'>"
            content += f"""
            <div class='metric-card'>
                <div class='metric-value'>{item_data['total_items_analyzed']:,}</div>
                <div class='metric-label'>분석된 아이템 수</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{item_data['items_with_category']:,}</div>
                <div class='metric-label'>카테고리 정보 보유 아이템</div>
            </div>
            """
            content += "</div>"
        
            # 상위 인기 아이템
            content += "<h3>🏆 상위 인기 아이템 (Top 10)</h3>"
            content += "<table class='data-table'>"
            content += "<tr><th>아이템 ID</th><th>인기도 점수</th><th>전환율 (%)</th><th>카테고리</th></tr>"
            
            for item in item_data['top_items'][:10]:
                content += f"""
                <tr>
                    <td>{item['itemid']}</td>
                    <td>{item['popularity_score']:.1f}</td>
                    <td>{item['conversion_rate']:.2f}</td>
                    <td>{item['categoryid'] if pd.notna(item['categoryid']) else 'N/A'}</td>
                </tr>
                """
            content += "</table>"
        
        # 4. 세션 기반 추천 분석
        if 'session_based_analysis' in self.results:
            session_data = self.results['session_based_analysis']
            content += "<h2>📊 4. 세션 기반 추천 분석</h2>"
            
            content += "<div class='stats-grid'>"
            content += f"""
            <div class='metric-card'>
                <div class='metric-value'>{session_data['total_sessions']:,}</div>
                <div class='metric-label'>총 세션 수</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{session_data['avg_session_length']:.1f}</div>
                <div class='metric-label'>평균 세션 길이</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{session_data['avg_unique_items_per_session']:.1f}</div>
                <div class='metric-label'>세션당 평균 고유 아이템</div>
            </div>
            """
            content += "</div>"
            
            # 세션 패턴 분포
            content += "<h3>🔄 세션 패턴 분포</h3>"
            content += "<div class='conversion-funnel'>"
            session_patterns = session_data['session_patterns']
            for pattern, count in session_patterns.items():
                percentage = (count / sum(session_patterns.values())) * 100
                content += f"""
                <div class='funnel-step'>
                    <strong>{pattern}:</strong> {count:,}개 세션 ({percentage:.2f}%)
                </div>
                """
            content += "</div>"
        
        return content
    
    def _generate_detailed_analysis_text(self):
        """상세 분석 결과 텍스트 생성"""
        analysis_text = ""
        
        # 1. 데이터 기본 구조
        analysis_text += "="*50 + "\n"
        analysis_text += "1. 데이터 기본 구조 파악\n"
        analysis_text += "="*50 + "\n\n"
        
        for dataset in self.results['basic_overview']:
            analysis_text += f"{dataset['Dataset'].upper()}\n"
            analysis_text += f"   Shape: ({dataset['Rows']}, {dataset['Columns']})\n"
            analysis_text += f"   Memory: {dataset['Memory (MB)']:.2f} MB\n"
            analysis_text += f"   Columns: {dataset['Columns']}\n\n"
        
        # 2. 이벤트 분석
        if 'events_analysis' in self.results:
            events_data = self.results['events_analysis']
            analysis_text += "="*50 + "\n"
            analysis_text += "2. 이벤트 로그 분석\n"
            analysis_text += "="*50 + "\n\n"
            
            analysis_text += "이벤트 타입 분포:\n"
            for event, count in events_data['event_counts'].items():
                analysis_text += f"{event}: {count:,}\n"
            
            analysis_text += "\n전환 퍼널:\n"
            funnel = events_data['conversion_funnel']
            analysis_text += f"   View → AddToCart: {funnel['view_to_cart_rate']:.2f}%\n"
            analysis_text += f"   AddToCart → Transaction: {funnel['cart_to_transaction_rate']:.2f}%\n"
            analysis_text += f"   View → Transaction: {funnel['view_to_transaction_rate']:.2f}%\n\n"
            
            user_stats = events_data['user_stats']
            analysis_text += "사용자 행동 수준:\n"
            analysis_text += f"   고유 방문자 수: {user_stats['unique_visitors']:,}\n"
            analysis_text += f"   총 이벤트 수: {user_stats['total_events']:,}\n"
            analysis_text += f"   방문자당 평균 이벤트 수: {user_stats['avg_events_per_visitor']:.2f}\n\n"
        
        # 3. 사용자 행동 패턴
        if 'user_behavior_analysis' in self.results:
            user_data = self.results['user_behavior_analysis']
            analysis_text += "="*50 + "\n"
            analysis_text += "3. 사용자 행동 패턴 분석\n"
            analysis_text += "="*50 + "\n\n"
            
            analysis_text += "사용자 유형 분포:\n"
            for user_type, count in user_data['user_type_distribution'].items():
                percentage = (count / sum(user_data['user_type_distribution'].values())) * 100
                analysis_text += f"   {user_type}: {count:,} ({percentage:.2f}%)\n"
            
            analysis_text += f"\n카테고리 선호도가 있는 사용자: {user_data['users_with_category_preference']:,}\n\n"
        
        # 4. 아이템 유사도 분석
        if 'item_similarity_analysis' in self.results:
            item_data = self.results['item_similarity_analysis']
            analysis_text += "="*50 + "\n"
            analysis_text += "4. 아이템 유사도 특성 분석\n"
            analysis_text += "="*50 + "\n\n"
            
            analysis_text += f"분석된 아이템 수: {item_data['total_items_analyzed']:,}\n"
            analysis_text += f"카테고리가 있는 아이템: {item_data['items_with_category']:,}\n\n"
            
            analysis_text += "상위 10개 인기 아이템:\n"
            for i, item in enumerate(item_data['top_items'][:10], 1):
                analysis_text += f"{i:2d}. 아이템 {item['itemid']}: 인기도 {item['popularity_score']:.1f}, 전환율 {item['conversion_rate']:.2f}%\n"
            analysis_text += "\n"
        
        # 5. 세션 기반 분석
        if 'session_based_analysis' in self.results:
            session_data = self.results['session_based_analysis']
            analysis_text += "="*50 + "\n"
            analysis_text += "5. 세션 기반 추천 분석\n"
            analysis_text += "="*50 + "\n\n"
            
            analysis_text += "세션 통계:\n"
            analysis_text += f"   총 세션 수: {session_data['total_sessions']:,}\n"
            analysis_text += f"   평균 세션 길이: {session_data['avg_session_length']:.2f} 이벤트\n"
            analysis_text += f"   평균 세션 시간: {session_data['avg_session_duration']:.2f} 분\n"
            analysis_text += f"   평균 세션당 고유 아이템: {session_data['avg_unique_items_per_session']:.2f}\n\n"
            
            analysis_text += "세션 패턴 분포:\n"
            for pattern, count in session_data['session_patterns'].items():
                percentage = (count / sum(session_data['session_patterns'].values())) * 100
                analysis_text += f"   {pattern}: {count:,} ({percentage:.2f}%)\n"
            analysis_text += "\n"
        
        # 6. 이상치 탐지
        if 'anomaly_detection' in self.results:
            anomaly_data = self.results['anomaly_detection']
            analysis_text += "="*50 + "\n"
            analysis_text += "6. 이상 사용자 탐지\n"
            analysis_text += "="*50 + "\n\n"
            
            analysis_text += "이상치 탐지 결과:\n"
            analysis_text += f"   총 사용자 수: {anomaly_data['total_users']:,}\n"
            analysis_text += f"   이상치 사용자 수: {anomaly_data['anomaly_users']:,}\n"
            analysis_text += f"   이상치 비율: {anomaly_data['anomaly_ratio']:.2f}%\n\n"
        
        return analysis_text
    
    def run_complete_eda(self):
        """전체 EDA 실행 (추천 시스템 특화)"""
        print("RetailRocket 추천 시스템 EDA 시작!")
        print("="*60)
        
        # 1. 데이터 로딩
        self.load_data()
        
        # 2. 기본 구조 파악
        self.basic_data_overview()
        
        # 3. 이벤트 분석
        self.analyze_events()
        
        # 4. 아이템 속성 분석
        self.analyze_item_properties()
        
        # 5. 카테고리 트리 분석
        self.analyze_category_tree()
        
        # 6. 사용자 행동 패턴 분석 (추천 시스템 특화)
        self.analyze_user_behavior_patterns()
        
        # 7. 아이템 유사도 특성 분석 (추천 시스템 특화)
        self.analyze_item_similarity_features()
        
        # 8. 세션 기반 추천 분석 (추천 시스템 특화)
        self.analyze_session_based_recommendations()
        
        # 9. 이상치 탐지
        self.detect_anomalies()
        
        # 10. 통합 HTML 리포트 생성 (시각화 포함)
        self.generate_html_report()
        
        print("\n" + "="*60)
        print("추천 시스템 EDA 완료!")
        print("생성된 파일:")
        print("   - EDA/retailrocket_recommendation_report.html")
        print("="*60)
        
        return self.results

# 실행
if __name__ == "__main__":
    eda = RetailRocketEDA()
    results = eda.run_complete_eda()
