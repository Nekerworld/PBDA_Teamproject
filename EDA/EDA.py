"""
RetailRocket 데이터셋 EDA (탐색적 데이터 분석)
목표: implicit feedback 기반 추천 모델을 위한 데이터 성격 파악
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
        self.events_df = pd.read_csv('events.csv')
        
        # Item properties 데이터 로딩 (part1, part2 합치기)
        item_props1 = pd.read_csv('item_properties_part1.csv')
        item_props2 = pd.read_csv('item_properties_part2.csv')
        self.item_properties_df = pd.concat([item_props1, item_props2], ignore_index=True)
        
        # Category tree 데이터 로딩
        self.category_tree_df = pd.read_csv('category_tree.csv')
        
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
    
    def analyze_sessions(self):
        """5️⃣ 세션 분석"""
        print("\n" + "="*50)
        print("5. 세션 분석")
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
            'itemid': 'nunique',
            'event': lambda x: list(x)
        }).reset_index()
        
        session_stats.columns = ['visitorid', 'session_id', 'session_start', 'session_end', 'event_count', 'unique_items', 'events']
        
        # 세션 길이 (분)
        session_stats['session_duration_minutes'] = (
            session_stats['session_end'] - session_stats['session_start']
        ).dt.total_seconds() / 60
        
        print(f"\n세션 통계:")
        print(f"   총 세션 수: {len(session_stats):,}")
        print(f"   평균 세션 길이: {session_stats['event_count'].mean():.2f} 이벤트")
        print(f"   평균 세션 시간: {session_stats['session_duration_minutes'].mean():.2f} 분")
        print(f"   평균 세션당 고유 아이템: {session_stats['unique_items'].mean():.2f}")
        
        # 세션별 행동 패턴 분석
        def analyze_session_pattern(events):
            events_str = ' '.join(events)
            if 'transaction' in events_str:
                return 'conversion'
            elif 'addtocart' in events_str:
                return 'cart_only'
            else:
                return 'view_only'
        
        session_stats['session_pattern'] = session_stats['events'].apply(analyze_session_pattern)
        pattern_counts = session_stats['session_pattern'].value_counts()
        
        print(f"\n세션 패턴 분포:")
        for pattern, count in pattern_counts.items():
            percentage = (count / len(session_stats)) * 100
            print(f"   {pattern}: {count:,} ({percentage:.2f}%)")
        
        # 결과 저장
        self.results['session_analysis'] = {
            'total_sessions': len(session_stats),
            'avg_session_length': session_stats['event_count'].mean(),
            'avg_session_duration': session_stats['session_duration_minutes'].mean(),
            'avg_unique_items_per_session': session_stats['unique_items'].mean(),
            'session_patterns': pattern_counts.to_dict()
        }
        
        return self.results['session_analysis']
    
    def detect_anomalies(self):
        """6️⃣ 이상 사용자 탐지"""
        print("\n" + "="*50)
        print("6. 이상 사용자 탐지")
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
        """7️⃣ 통합 시각화"""
        print("\n" + "="*50)
        print("7. 시각화 생성")
        print("="*50)
        
        # Plotly를 사용한 인터랙티브 시각화
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('이벤트 타입 분포', '시간대별 이벤트 분포', 
                          '요일별 이벤트 분포', '아이템 인기도 분포',
                          '전환 퍼널', '카테고리별 이벤트 수'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "funnel"}, {"type": "bar"}]]
        )
        
        # 1. 이벤트 타입 분포 (파이 차트)
        event_counts = self.events_df['event'].value_counts()
        fig.add_trace(
            go.Pie(labels=event_counts.index, values=event_counts.values, name="이벤트 타입"),
            row=1, col=1
        )
        
        # 2. 시간대별 이벤트 분포
        hourly_events = self.events_df.groupby('hour')['event'].count()
        fig.add_trace(
            go.Bar(x=hourly_events.index, y=hourly_events.values, name="시간대별 이벤트"),
            row=1, col=2
        )
        
        # 3. 요일별 이벤트 분포
        daily_events = self.events_df.groupby('day_name')['event'].count()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_events = daily_events.reindex(day_order)
        fig.add_trace(
            go.Bar(x=daily_events.index, y=daily_events.values, name="요일별 이벤트"),
            row=2, col=1
        )
        
        # 4. 아이템 인기도 분포 (로그 스케일)
        item_counts = self.events_df['itemid'].value_counts()
        fig.add_trace(
            go.Histogram(x=np.log10(item_counts.values), name="아이템 인기도 (로그)"),
            row=2, col=2
        )
        
        # 5. 전환 퍼널
        funnel_data = [
            ('View', event_counts.get('view', 0)),
            ('AddToCart', event_counts.get('addtocart', 0)),
            ('Transaction', event_counts.get('transaction', 0))
        ]
        fig.add_trace(
            go.Funnel(y=[x[0] for x in funnel_data], x=[x[1] for x in funnel_data], name="전환 퍼널"),
            row=3, col=1
        )
        
        # 6. 카테고리별 이벤트 수 (상위 20개)
        category_mapping = self.item_properties_df[
            self.item_properties_df['property'] == 'categoryid'
        ][['itemid', 'value']].rename(columns={'value': 'categoryid'})
        category_mapping['categoryid'] = category_mapping['categoryid'].astype(int)
        
        events_with_category = self.events_df.merge(category_mapping, on='itemid', how='left')
        category_event_counts = events_with_category['categoryid'].value_counts().head(20)
        
        fig.add_trace(
            go.Bar(x=[str(x) for x in category_event_counts.index], 
                   y=category_event_counts.values, name="카테고리별 이벤트"),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=False, title_text="RetailRocket 데이터셋 EDA 시각화")
        
        # 시각화 저장
        fig.write_html("retailrocket_eda_visualizations.html")
        print("시각화가 'retailrocket_eda_visualizations.html'에 저장되었습니다.")
        
        return fig
    
    def generate_html_report(self):
        """8️⃣ HTML 리포트 생성"""
        print("\n" + "="*50)
        print("8. HTML 리포트 생성")
        print("="*50)
        
        # HTML 템플릿
        html_template = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RetailRocket 데이터셋 EDA 리포트</title>
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
                .insight-box {{
                    background: #e8f5e8;
                    border-left: 4px solid #27ae60;
                    padding: 15px;
                    margin: 15px 0;
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
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛍️ RetailRocket 데이터셋 EDA 리포트</h1>
                <p style="text-align: center; color: #7f8c8d; font-size: 1.1em;">
                    <strong>목표:</strong> Implicit Feedback 기반 추천 모델을 위한 데이터 성격 파악
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
        
        # HTML 파일 생성
        html_content = html_template.format(
            content=content,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open("retailrocket_eda_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("HTML 리포트가 'retailrocket_eda_report.html'에 저장되었습니다.")
        
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
        
        # 2. 이벤트 분석 결과
        if 'events_analysis' in self.results:
            events_data = self.results['events_analysis']
            content += "<h2>🎯 2. 이벤트 분석 결과</h2>"
            
            # 전환 퍼널
            content += "<div class='conversion-funnel'>"
            content += "<h3>🔄 전환 퍼널</h3>"
            
            funnel_data = events_data['conversion_funnel']
            content += f"""
            <div class='funnel-step'>
                <strong>View → AddToCart:</strong> {funnel_data['view_to_cart_rate']:.2f}%
            </div>
            <div class='funnel-step'>
                <strong>AddToCart → Transaction:</strong> {funnel_data['cart_to_transaction_rate']:.2f}%
            </div>
            <div class='funnel-step'>
                <strong>View → Transaction:</strong> {funnel_data['view_to_transaction_rate']:.2f}%
            </div>
            """
            content += "</div>"
            
            # 사용자 통계
            user_stats = events_data['user_stats']
            content += "<div class='stats-grid'>"
            content += f"""
            <div class='metric-card'>
                <div class='metric-value'>{user_stats['unique_visitors']:,}</div>
                <div class='metric-label'>고유 방문자</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{user_stats['total_events']:,}</div>
                <div class='metric-label'>총 이벤트</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{user_stats['avg_events_per_visitor']:.1f}</div>
                <div class='metric-label'>방문자당 평균 이벤트</div>
            </div>
            """
            content += "</div>"
        
        # 3. 인사이트 및 권장사항
        content += "<h2>💡 3. 주요 인사이트 및 권장사항</h2>"
        
        insights = self._generate_insights()
        for insight in insights:
            if insight['type'] == 'insight':
                content += f"<div class='insight-box'><strong>💡 {insight['title']}</strong><br>{insight['content']}</div>"
            else:
                content += f"<div class='warning-box'><strong>⚠️ {insight['title']}</strong><br>{insight['content']}</div>"
        
        return content
    
    def _generate_insights(self):
        """인사이트 생성"""
        insights = []
        
        if 'events_analysis' in self.results:
            events_data = self.results['events_analysis']
            
            # 전환율 분석
            view_to_transaction = events_data['conversion_funnel']['view_to_transaction_rate']
            if view_to_transaction < 1:
                insights.append({
                    'type': 'warning',
                    'title': '낮은 전환율',
                    'content': f'전체 전환율이 {view_to_transaction:.2f}%로 매우 낮습니다. 이는 일반적인 e-commerce 사이트보다 낮은 수준입니다.'
                })
            
            # 사용자 행동 패턴
            avg_events = events_data['user_stats']['avg_events_per_visitor']
            if avg_events > 10:
                insights.append({
                    'type': 'insight',
                    'title': '높은 사용자 참여도',
                    'content': f'방문자당 평균 {avg_events:.1f}개의 이벤트로 높은 참여도를 보입니다. 이는 충분한 행동 데이터가 있음을 의미합니다.'
                })
            
            # Long-tail 분포
            concentration = events_data['item_stats']['concentration_ratio']
            if concentration > 20:
                insights.append({
                    'type': 'warning',
                    'title': '높은 집중도',
                    'content': f'상위 1% 아이템이 전체 조회의 {concentration:.1f}%를 차지합니다. 이는 매우 불균형한 분포입니다.'
                })
        
        if 'anomaly_detection' in self.results:
            anomaly_data = self.results['anomaly_detection']
            if anomaly_data['anomaly_ratio'] > 5:
                insights.append({
                    'type': 'warning',
                    'title': '높은 이상치 비율',
                    'content': f'{anomaly_data["anomaly_ratio"]:.1f}%의 사용자가 이상치로 분류되었습니다. 봇이나 크롤러 제거를 고려해야 합니다.'
                })
        
        # 추천 모델 방향성
        insights.append({
            'type': 'insight',
            'title': '추천 모델 방향성',
            'content': 'Implicit feedback 기반이므로 협업 필터링과 행동 기반 추천이 적합합니다. 세션 기반 추천도 고려할 수 있습니다.'
        })
        
        return insights
    
    def run_complete_eda(self):
        """전체 EDA 실행"""
        print("RetailRocket 데이터셋 EDA 시작!")
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
        
        # 6. 세션 분석
        self.analyze_sessions()
        
        # 7. 이상치 탐지
        self.detect_anomalies()
        
        # 8. 시각화 생성
        self.create_visualizations()
        
        # 9. HTML 리포트 생성
        self.generate_html_report()
        
        print("\n" + "="*60)
        print("EDA 완료!")
        print("생성된 파일:")
        print("   - retailrocket_eda_visualizations.html (시각화)")
        print("   - retailrocket_eda_report.html (종합 리포트)")
        print("="*60)
        
        return self.results

# 실행
if __name__ == "__main__":
    eda = RetailRocketEDA()
    results = eda.run_complete_eda()
