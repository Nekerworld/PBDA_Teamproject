"""
추천 시스템 모델 파이프라인

이 모듈은 다음 단계로 구성된 추천 시스템을 구현합니다:
1. IsolationForestPreprocessor: 이상치 탐지 및 데이터 전처리
2. ALSRecommender: ALS(Alternating Least Squares) 기반 협업 필터링
3. GNNEmbeddingGenerator: 그래프 신경망 기반 임베딩 생성
4. ReRanker: ALS와 GNN 결과를 결합한 리랭킹
5. TestSetEvaluator: 테스트 세트 평가 및 성능 지표 계산

각 컴포넌트는 독립적으로 실행 가능하며, 순차적으로 실행되어 최종 추천 결과를 생성합니다.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

try:
    from .data_protocol import (UnifiedPreprocessor, DEFAULT_DATA, DEFAULT_PROCESSED,
        require_protocol, load_shared_split, require_artifacts, stamp_artifacts, dump_json,
        positive_sets)
    from .ranking_evaluation import TestSetEvaluator, ranking_metrics
    from .recommendation_scoring import (FactorScorer, GNNScorer, minmax,
        top_indices, append_ranking, recommendation_frame, hybrid_scores, hybrid_rerank_scores)
except ImportError:
    from data_protocol import (UnifiedPreprocessor, DEFAULT_DATA, DEFAULT_PROCESSED,
        require_protocol, load_shared_split, require_artifacts, stamp_artifacts, dump_json,
        positive_sets)
    from ranking_evaluation import TestSetEvaluator, ranking_metrics
    from recommendation_scoring import (FactorScorer, GNNScorer, minmax,
        top_indices, append_ranking, recommendation_frame, hybrid_scores, hybrid_rerank_scores)


# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# 전역 시드 고정 (재현가능성 보장)
GLOBAL_SEED = 42

def load_without_adjacency(model, state):
    """Load a checkpoint saved without the fixed graph; any other key mismatch is an error."""
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) - {'adjacency'}:
        raise RuntimeError(f'Checkpoint mismatch: missing={result.missing_keys} unexpected={result.unexpected_keys}')
    return result


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """
    모든 랜덤 시드를 고정하여 재현가능성을 보장합니다.
    
    Args:
        seed: 고정할 시드 값 (기본값: 42)
    """
    # Python random 모듈
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (가능한 경우)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경
        # 재현가능성을 위한 추가 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch가 설치되지 않은 경우 무시

# 전역 시드 설정 실행
set_global_seed(GLOBAL_SEED)

# 이벤트 타입을 숫자 코드로 매핑 (view=0, addtocart=1, transaction=2)
EVENT_TO_CODE: Dict[str, int] = {"view": 0, "addtocart": 1, "transaction": 2}

# 숫자 추출을 위한 정규표현식 패턴 (음수 표기 "n" 또는 "-" 지원)
NUMERIC_PATTERN = re.compile(r"(?P<sign>n|-)?(?P<number>\d+(?:\.\d+)?)")

# 이벤트 타입별 가중치 (transaction > addtocart > view)
EVENT_WEIGHT: Dict[str, float] = {"view": 1.0, "addtocart": 6.0, "transaction": 12.0}

# ALS 전용 가중치·설정 (als_debug 실험 반영, NDCG/Recall 향상)
ALS_WEIGHT_MAP: Dict[str, float] = {"view": 1.0, "addtocart": 15.0, "transaction": 32.0}
USE_WEIGHT_LOG_SCALE = False
ENSEMBLE_ALPHA = 0.5
BPR_REGULARIZATION = 0.03


def _learning():
    try:
        from . import learning
    except ImportError:
        import learning
    return learning


def _ndcg_at_k(recommendations, positives, k, score_col="score", ascending=False):
    return ranking_metrics(recommendations, positives, k, score_col, ascending)[0]["ndcg"]


def _recall_at_k(recommendations, positives, k, score_col="score", ascending=False):
    return ranking_metrics(recommendations, positives, k, score_col, ascending)[0]["recall"]


def _extract_numeric(value: Optional[str]) -> Optional[float]:
    """
    문자열에서 숫자 값을 추출합니다.
    
    "n" 또는 "-" 접두사가 있으면 음수로 처리합니다.
    예: "n123" -> -123.0, "456.7" -> 456.7
    
    Args:
        value: 숫자를 추출할 문자열 (None 가능)
        
    Returns:
        추출된 숫자 값 (float), 추출 실패 시 None
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    match = NUMERIC_PATTERN.search(str(value))
    if not match:
        return None
    sign = match.group("sign")
    number = match.group("number")
    prefix = "-" if sign == "n" or sign == "-" else ""
    try:
        return float(f"{prefix}{number}")
    except ValueError:
        return None


IsolationForestPreprocessor = UnifiedPreprocessor
ALSPreprocessor = UnifiedPreprocessor
GNNPreprocessor = UnifiedPreprocessor

@dataclass
class GNNEmbeddingGenerator:
    processed_dir: Path = DEFAULT_PROCESSED
    data_dir: Path = DEFAULT_DATA
    embedding_dim: int = 8
    layers: int = 2
    epochs: int = 200
    batch_size: int = 65536
    learning_rate: float = .001
    reg: float = .0001
    seed: int = 42
    device: Optional[str] = None
    early_stopping_patience: Optional[int] = 25
    min_epochs: int = 1
    min_delta: float = 1e-6
    behavior_weighted: bool = True
    property_mode: str = 'values'
    metadata_policy: str = 'latest'
    validation_k: int = 50
    validation_interval: int = 5
    validation_metric: str = 'ndcg'
    propagation: str = 'unweighted'
    edge_dropout: float = 0.0
    negative_mode: str = 'uniform'
    item_item_edges: bool = False
    time_decay: bool = False
    half_life_days: float = 7.0
    graph_mode: str = 'bipartite'
    loss_mode: str = 'bpr'
    n_negatives: int = 1
    temperature: float = 0.2

    def __post_init__(self):
        import torch
        self.processed_dir=Path(self.processed_dir)
        self.device=self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.propagation not in {'unweighted', 'weighted'}:
            raise ValueError('propagation must be unweighted or weighted')
        if self.graph_mode not in {'bipartite', 'heterogeneous'}:
            raise ValueError('graph_mode must be bipartite or heterogeneous')
        if self.loss_mode not in {'bpr', 'infonce'}:
            raise ValueError('loss_mode must be bpr or infonce')
        if self.n_negatives < 1:
            raise ValueError('n_negatives must be positive')
        if self.min_epochs < 1:
            raise ValueError('min_epochs must be positive')
        if not 0 <= self.edge_dropout < 1:
            raise ValueError('edge_dropout must be in [0, 1)')
        if self.validation_metric not in {'ndcg', 'ndcg_recall_mean'}:
            raise ValueError('validation_metric must be ndcg or ndcg_recall_mean')

    def _load_inputs_from(self,directory):
        require_protocol(directory)
        return {'events':pd.read_csv(Path(directory)/'events_train_clean.csv'),
                'item_properties':pd.read_csv(Path(directory)/'item_properties_train_clean.csv',dtype={'property':str,'value':str}),
                'category_tree':pd.read_csv(Path(directory)/'category_tree.csv')}

    def _load_inputs(self):
        return self._load_inputs_from(self.processed_dir)

    def _prepare_graph_structures(self, datasets):
        learning = _learning()
        events = datasets['events']
        pairs = learning.interaction_table(events, self.behavior_weighted, self.time_decay, self.half_life_days)
        user_ids = sorted(map(int, events.visitorid.unique()))
        item_ids = sorted(map(int, events.itemid.unique()))
        user_map = {u: i for i, u in enumerate(user_ids)}
        item_map = {u: i for i, u in enumerate(item_ids)}
        use_side = self.graph_mode == 'heterogeneous'
        props = datasets['item_properties'].copy()
        props = props.loc[props.itemid.isin(item_ids)]
        props = learning.encode_properties(props, self.property_mode, self.metadata_policy) if use_side else props.iloc[0:0]
        category = props.loc[props.property == 'categoryid'].copy() if use_side else props.iloc[0:0]
        if len(category):
            category['categoryid'] = pd.to_numeric(category.value, errors='coerce')
            category = category.dropna(subset=['categoryid'])
            category['categoryid'] = category.categoryid.astype(int)
        tree = datasets['category_tree'].copy() if use_side else pd.DataFrame(columns=['categoryid', 'parentid'])
        if use_side:
            categories = set(tree.categoryid.dropna().astype(int)) | set(tree.parentid.dropna().astype(int))
            if len(category):
                categories |= set(category.categoryid.astype(int))
            categories = sorted(categories)
        else:
            categories = []
        cat_map = {v: i for i, v in enumerate(categories)}
        # Inventory is applied only by the reranker, so available is not a graph node.
        properties = props.loc[~props.property.isin(['categoryid', 'available'])].copy() if use_side else props.iloc[0:0]
        property_ids = sorted(properties.node_key.unique()) if use_side else []
        prop_map = {v: i for i, v in enumerate(property_ids)}
        counts = {'user': len(user_ids), 'item': len(item_ids), 'property': len(property_ids), 'category': len(categories)}
        offsets, total = {}, 0
        for key, value in counts.items():
            offsets[key] = total
            total += value
        edge_sources, edge_targets, edge_weights = [], [], []

        def add_undirected(src, dst, weight):
            edge_sources.extend([int(src), int(dst)])
            edge_targets.extend([int(dst), int(src)])
            edge_weights.extend([float(weight), float(weight)])

        interaction = np.column_stack([pairs.visitorid.map(user_map), pairs.itemid.map(item_map)]).astype(np.int64)
        pair_weights = pairs.weight.to_numpy(dtype=np.float64)
        for (user_idx, item_idx), weight in zip(interaction, pair_weights):
            value = weight if self.propagation == 'weighted' else 1.0
            add_undirected(user_idx, item_idx + offsets['item'], value)
        if use_side and len(properties):
            edges = properties[['itemid', 'node_key']].drop_duplicates()
            for item_idx, prop_idx in zip(edges.itemid.map(item_map), edges.node_key.map(prop_map)):
                add_undirected(item_idx + offsets['item'], prop_idx + offsets['property'], 1.0)
        if use_side and len(category):
            edges = category[['itemid', 'categoryid']].drop_duplicates()
            for item_idx, cat_idx in zip(edges.itemid.map(item_map), edges.categoryid.map(cat_map)):
                add_undirected(item_idx + offsets['item'], cat_idx + offsets['category'], 1.0)
        if use_side:
            tree_edges = tree.dropna(subset=['categoryid', 'parentid']).copy()
            if len(tree_edges):
                tree_edges['categoryid'] = tree_edges.categoryid.astype(int)
                tree_edges['parentid'] = tree_edges.parentid.astype(int)
                for cat_idx, parent_idx in zip(tree_edges.categoryid.map(cat_map), tree_edges.parentid.map(cat_map)):
                    add_undirected(cat_idx + offsets['category'], parent_idx + offsets['category'], 1.0)
        if use_side and self.item_item_edges:
            co_edges, co_weights = learning.item_cooccurrence_edges(pairs, item_map)
            for (left, right), weight in zip(co_edges, co_weights):
                add_undirected(left + offsets['item'], right + offsets['item'], weight)
        # Aggregate duplicate undirected edges created by unique mappings.
        frame = pd.DataFrame({'s': edge_sources, 't': edge_targets, 'w': edge_weights})
        frame = frame.groupby(['s', 't'], as_index=False).w.sum()
        loss_weights = pair_weights.astype(np.float32)
        loss_weights = loss_weights / loss_weights.mean()
        return {'edge_sources': frame.s.to_numpy(dtype=np.int64), 'edge_targets': frame.t.to_numpy(dtype=np.int64),
            'edge_weights': frame.w.to_numpy(dtype=np.float32),
            'total_nodes': total, 'user_ids': user_ids, 'item_ids': item_ids, 'property_ids': property_ids,
            'category_ids': categories, 'user_map': user_map, 'item_map': item_map, 'offsets': offsets,
            'interactions': interaction, 'weights': loss_weights,
            **{f'num_{key}s': value for key, value in counts.items()},
            'num_properties': len(property_ids), 'num_categories': len(categories),
            'graph_mode': self.graph_mode}

    def _fit_graph(self, graph, epochs, torch, validation=None):
        learning = _learning()
        NegativeSampler = learning.NegativeSampler
        weighted_pairwise_loss = learning.weighted_pairwise_loss
        info_nce_loss = learning.info_nce_loss
        sample_negatives_matrix = learning.sample_negatives_matrix
        set_global_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('GNN was asked to use cuda but torch.cuda.is_available() is False')
        logger.info(
            'GNN device=%s cuda_available=%s nodes=%d pairs=%d dim=%d layers=%d mode=%s loss=%s n_neg=%d',
            self.device, torch.cuda.is_available(), graph['total_nodes'],
            len(graph['interactions']), self.embedding_dim, self.layers,
            self.graph_mode, self.loss_mode, self.n_negatives)
        adjacency=self._build_normalized_adjacency(graph['edge_sources'],graph['edge_targets'],graph['total_nodes'],
                                                   graph.get('edge_weights'))
        adjacency = adjacency.to(self.device)
        model=self._build_model(adjacency,torch)
        if next(model.parameters()).device.type != ('cuda' if self.device == 'cuda' else 'cpu'):
            raise RuntimeError(f'GNN parameters stayed on {next(model.parameters()).device}, expected {self.device}')
        if model.adjacency.device.type != ('cuda' if self.device == 'cuda' else 'cpu'):
            raise RuntimeError(f'GNN adjacency stayed on {model.adjacency.device}, expected {self.device}')
        logger.info('GNN tensors on embedding=%s adjacency=%s', next(model.parameters()).device, model.adjacency.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=self.learning_rate)
        interactions=graph['interactions']
        popularity=np.bincount(interactions[:,1], minlength=graph['num_items']).astype(np.float64)
        if self.negative_mode == 'mf_mixed':
            sampler=learning.MFMixedNegativeSampler(interactions,graph['num_items'],
                self.processed_dir, graph['user_ids'], graph['item_ids'], seed=self.seed)
        else:
            sampler=NegativeSampler(interactions,graph['num_items'], mode=self.negative_mode, item_popularity=popularity)
        best_score=-np.inf; best_epoch=0; best_state=None; history=[]
        item_offset = graph['offsets']['item']
        n_neg = self.n_negatives if self.loss_mode == 'infonce' else 1
        min_epochs = min(self.min_epochs, epochs)
        for epoch in range(1,epochs+1):
            if n_neg == 1:
                negatives = sampler.sample(interactions[:,0], rng)
                valid = negatives >= 0
            else:
                negatives = sample_negatives_matrix(sampler, interactions[:,0], rng, n_neg)
                valid = (negatives >= 0).all(axis=1)
            order=rng.permutation(np.flatnonzero(valid))
            if not len(order):
                raise ValueError('No unobserved items remain for GNN training')
            total=0.; count=0
            for start in range(0,len(order),self.batch_size):
                idx=order[start:start+self.batch_size]
                u=torch.as_tensor(interactions[idx,0],device=self.device)
                p=torch.as_tensor(interactions[idx,1]+item_offset,device=self.device)
                w=torch.as_tensor(graph['weights'][idx],device=self.device)
                model.train(); optimizer.zero_grad()
                vectors=model(self.edge_dropout if model.training else 0.0)
                if self.loss_mode == 'infonce' and n_neg > 1:
                    neg_idx = torch.as_tensor(negatives[idx] + item_offset, device=self.device)
                    neg_vectors = vectors[neg_idx.reshape(-1)].view(len(idx), n_neg, -1)
                    loss = info_nce_loss(vectors[u], vectors[p], neg_vectors, w, self.reg, self.temperature)
                else:
                    n=torch.as_tensor(np.asarray(negatives[idx]).reshape(-1)+item_offset,device=self.device)
                    loss=weighted_pairwise_loss(vectors[u],vectors[p],vectors[n],w,self.reg)
                if not torch.isfinite(loss): raise ValueError('Non-finite GNN loss')
                loss.backward(); optimizer.step()
                total+=float(loss.detach())*len(idx); count+=len(idx)
            skipped = int((~valid).sum())
            row={'epoch':epoch,'train_loss':total/count,'pairs':count,'saturated_pairs_skipped':skipped}
            if validation is not None and (epoch==1 or epoch%self.validation_interval==0 or epoch==epochs):
                model.eval()
                score=self._validation_score(model,graph,validation)
                row['validation_original_ndcg']=score
                observer = getattr(self, '_checkpoint_observer', None)
                if observer is not None:
                    row.update(observer(model, graph, epoch))
                eligible = epoch >= min_epochs
                if eligible and score>best_score+self.min_delta:
                    best_score=score; best_epoch=epoch
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k!='adjacency'}
                if best_epoch >= min_epochs and epoch-best_epoch>=self.early_stopping_patience:
                    history.append(row); break
            history.append(row)
            if epoch==1 or epoch%5==0:
                logger.info('GNN epoch %d/%d %s',epoch,epochs,row)
        if validation is not None and best_epoch < 1:
            raise RuntimeError('Validation did not select a GNN epoch')
        if best_state is not None:
            load_without_adjacency(model, best_state)
        return model, best_epoch if validation is not None else epochs, history

    def _publish(self,model,graph,history,chosen,start,selection=None):
        import torch
        model.eval()
        with torch.no_grad(): self._save_embeddings(model(0.0).detach().cpu().numpy(),graph)
        info={'chosen_epochs':chosen,'refit_history':history,**(selection or {})}
        dump_json(self.processed_dir/'gnn_training_history.json',info)
        stamp_artifacts(self.processed_dir, 'gnn', ['gnn_user_embeddings.csv', 'gnn_item_embeddings.csv', 'gnn_training_history.json'],
            embedding_dim=self.embedding_dim, layers=self.layers, chosen_epochs=chosen,
            seed=self.seed, behavior_weighted=self.behavior_weighted, property_mode=self.property_mode,
            metadata_policy=self.metadata_policy, training_seconds=time.perf_counter()-start,
            negative_sampling=f'{self.negative_mode} excluding all train observations; saturated rows skipped',
            propagation=f'{self.propagation} LightGCN; edge_dropout={self.edge_dropout}',
            pairwise_loss=('behavior-weighted InfoNCE' if self.behavior_weighted else 'binary InfoNCE')
                if self.loss_mode == 'infonce'
                else ('behavior-weighted BPR' if self.behavior_weighted else 'binary pair BPR'),
            inventory_property_in_graph=False, property_nodes=len(graph['property_ids']),
            item_item_edges=self.item_item_edges and self.graph_mode == 'heterogeneous',
            time_decay=self.time_decay,
            graph_mode=self.graph_mode, loss_mode=self.loss_mode, n_negatives=self.n_negatives,
            temperature=self.temperature, min_epochs=self.min_epochs,
            device=self.device,
            selection_metric=('validation original NDCG' if self.validation_metric == 'ndcg'
                              else 'validation original mean(NDCG, Recall)'), validation_k=self.validation_k,
            validation_protocol_id=(selection or {}).get('validation_protocol_id'))

    def run(self):
        import torch
        start = time.perf_counter()
        directory = self.processed_dir
        chosen=self.epochs; selection={'mode':'fixed_epochs'}
        if self.early_stopping_patience is not None:
            valdir=directory/'validation'
            graph=self._prepare_graph_structures(self._load_inputs_from(valdir))
            positives=positive_sets(pd.read_csv(valdir/'events_test_original.csv'))
            model,chosen,history=self._fit_graph(graph,self.epochs,torch,positives)
            self.processed_dir=valdir
            self._publish(model,graph,history,chosen,start,{'mode':'validation_selection'})
            self.processed_dir=directory
            selection={'mode':'chronological_validation_then_refit','selection_history':history,
                'validation_protocol_id':require_protocol(valdir,verify=False)['protocol_id']}
            del model,graph
            if self.device=='cuda': torch.cuda.empty_cache()
        graph=self._prepare_graph_structures(self._load_inputs())
        model,_,history=self._fit_graph(graph,chosen,torch)
        self._publish(model,graph,history,chosen,start,selection)

    def _build_normalized_adjacency(
        self,
        sources,
        targets,
        total_nodes: int,
        weights=None,
    ) -> Any:
        sources = np.asarray(sources, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        if len(sources) == 0 or len(targets) == 0:
            raise ValueError("그래프에 유효한 엣지가 없습니다.")
        data = np.ones(len(sources), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        if data.shape != sources.shape:
            raise ValueError('Edge weights must match edge endpoints')
        adjacency = sp.coo_matrix((data, (sources, targets)), shape=(total_nodes, total_nodes)).tocsr()
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            deg_inv_sqrt = np.power(degrees, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        deg_inv_sqrt[np.isnan(deg_inv_sqrt)] = 0.0
        deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
        normalized = (deg_inv_sqrt_mat @ adjacency @ deg_inv_sqrt_mat).tocoo()
        indices = np.vstack((normalized.row, normalized.col))
        import torch
        indices_tensor = torch.from_numpy(indices).long()
        values_tensor = torch.from_numpy(normalized.data.astype(np.float32))
        adjacency_tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, torch.Size(normalized.shape))
        return adjacency_tensor.coalesce()

    def _build_model(self, adjacency: Any, torch_module: Any) -> Any:
        class LightGCN(torch_module.nn.Module):
            def __init__(self, num_nodes: int, embedding_dim: int, n_layers: int, adj: Any) -> None:
                super().__init__()
                self.embedding = torch_module.nn.Embedding(num_nodes, embedding_dim)
                torch_module.nn.init.xavier_uniform_(self.embedding.weight)
                self.n_layers = n_layers
                self.register_buffer("adjacency", adj.coalesce())

            def forward(self, edge_dropout: float = 0.0) -> torch_module.Tensor:
                adjacency = self.adjacency
                if self.training and edge_dropout > 0:
                    values = adjacency.values()
                    keep = (torch_module.rand(values.shape, device=values.device) >= edge_dropout).to(values.dtype)
                    values = values * keep / (1.0 - edge_dropout)
                    adjacency = torch_module.sparse_coo_tensor(
                        adjacency.indices(), values, adjacency.size(), device=values.device).coalesce()
                embeddings = self.embedding.weight
                all_embeddings = [embeddings]
                for _ in range(self.n_layers):
                    embeddings = torch_module.sparse.mm(adjacency, embeddings)
                    all_embeddings.append(embeddings)
                stacked = torch_module.stack(all_embeddings, dim=0)
                return stacked.mean(dim=0)

        model = LightGCN(adjacency.shape[0], self.embedding_dim, self.layers, adjacency)
        return model.to(torch_module.device(self.device))

    def _validation_score(self, model, graph, positives):
        import torch
        model.eval()
        with torch.no_grad():
            vectors = model(0.0)
        start = graph['offsets']['item']
        item_vectors = vectors[start:start + graph['num_items']]
        ids = np.asarray(graph['item_ids'])
        eligible = [uid for uid in positives if uid in graph['user_map']]
        records = []
        if eligible:
            index = torch.as_tensor([graph['user_map'][uid] for uid in eligible], device=vectors.device)
            users = vectors.index_select(0, index)
            for offset in range(0, len(eligible), 256):
                chunk = users[offset:offset + 256]
                scores = (item_vectors @ chunk.T).detach().cpu().numpy()
                for column, uid in enumerate(eligible[offset:offset + 256]):
                    append_ranking(records, uid, ids, scores[:, column], self.validation_k)
        metrics = ranking_metrics(recommendation_frame(records), positives, self.validation_k)[0]
        if self.validation_metric == 'ndcg_recall_mean':
            return (metrics['ndcg'] + metrics['recall']) / 2
        return metrics['ndcg']

    def _save_embeddings(self, all_embeddings: np.ndarray, graph_data: Dict[str, Any]) -> None:
        """
        학습된 임베딩을 CSV 파일로 저장합니다.
        
        사용자와 아이템 임베딩만 추출하여 저장합니다.
        
        Args:
            all_embeddings: 모든 노드의 임베딩 행렬 (shape: [total_nodes, embedding_dim])
            graph_data: 그래프 구조 정보 딕셔너리
            
        저장되는 파일:
        - gnn_user_embeddings.csv: 사용자 임베딩
        - gnn_item_embeddings.csv: 아이템 임베딩
        """
        user_offset = graph_data["offsets"]["user"]
        item_offset = graph_data["offsets"]["item"]

        user_indices = np.arange(graph_data["num_users"], dtype=np.int64)
        item_indices = np.arange(graph_data["num_items"], dtype=np.int64)

        user_global = user_offset + user_indices
        item_global = item_offset + item_indices

        user_embeddings = all_embeddings[user_global]
        item_embeddings = all_embeddings[item_global]

        user_df = pd.DataFrame(user_embeddings, columns=[f"embedding_{i}" for i in range(self.embedding_dim)])
        user_df.insert(0, "visitorid", graph_data["user_ids"])

        item_df = pd.DataFrame(item_embeddings, columns=[f"embedding_{i}" for i in range(self.embedding_dim)])
        item_df.insert(0, "itemid", graph_data["item_ids"])

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        user_df.to_csv(self.processed_dir / "gnn_user_embeddings.csv", index=False)
        item_df.to_csv(self.processed_dir / "gnn_item_embeddings.csv", index=False)



@dataclass
class ALSRecommender:
    """Fit ALS and BPR once; save all three baselines and real retrieval scores."""
    processed_dir: Path = DEFAULT_PROCESSED
    factors: int = 128
    regularization: float = 0.03
    bpr_regularization: float = 0.03
    iterations: int = 256
    alpha: float = 1.5
    top_k: int = 200
    random_state: int = 42
    weight_map: Optional[Dict[str, float]] = None
    use_weight_log_scale: bool = False
    ensemble_alpha: float = 0.5
    behavior_weighted: bool = True
    bpr_epochs: int = 128
    device: str = 'cpu'
    als_threads: int = 8
    time_decay: bool = False
    half_life_days: float = 7.0
    negative_mode: str = 'uniform'

    def run(self):
        try:
            from . import als_debug
        except ImportError:
            import als_debug
        if self.top_k < 1 or not 0 <= self.ensemble_alpha <= 1:
            raise ValueError('Invalid recommendation depth or ensemble alpha')
        learning = _learning()
        out = self.processed_dir = Path(self.processed_dir)
        train, _, _, _ = load_shared_split(out)
        weight_map = self.weight_map or learning.BEHAVIOR_WEIGHTS
        if self.time_decay:
            pairs = learning.interaction_table(train, self.behavior_weighted, True, self.half_life_days)
            if self.use_weight_log_scale:
                pairs = pairs.copy()
                pairs['weight'] = 1.0 + np.log1p(pairs.weight.to_numpy(dtype=float))
            user_ids, unique_users = pd.factorize(pairs.visitorid, sort=True)
            item_ids, unique_items = pd.factorize(pairs.itemid, sort=True)
            matrix = sp.coo_matrix((pairs.weight.astype(float), (user_ids, item_ids)),
                                   shape=(len(unique_users), len(unique_items))).tocsr()
            users = unique_users.to_numpy() if hasattr(unique_users, 'to_numpy') else np.asarray(unique_users)
            items = unique_items.to_numpy() if hasattr(unique_items, 'to_numpy') else np.asarray(unique_items)
            user_map = {int(uid): idx for idx, uid in enumerate(users)}
        else:
            matrix, users, items, user_map, *_ = als_debug.build_user_item_matrix(
                train, weight_map if self.behavior_weighted else {k: 1.0 for k in weight_map},
                self.use_weight_log_scale)
        unchanged = matrix.data.copy()
        start = time.perf_counter()
        als = als_debug.train_als(matrix, self.factors, self.regularization,
                                  self.iterations, self.random_state, self.alpha,
                                  num_threads=self.als_threads, use_gpu=False)
        if not np.array_equal(matrix.data, unchanged):
            raise RuntimeError('ALS confidence conversion changed the behavior-weight matrix used by BPR')
        als_seconds = time.perf_counter() - start
        start = time.perf_counter()
        bpr = learning.train_weighted_bpr(matrix, self.factors, self.bpr_regularization,
            self.bpr_epochs, self.random_state, self.device, self.behavior_weighted,
            negative_mode=self.negative_mode)
        bpr_seconds = time.perf_counter() - start
        dump_json(out/'bpr_training_history.json',bpr.history)
        files = ['bpr_training_history.json']
        for prefix, fitted in [('als', als), ('bpr', bpr)]:
            for entity in ['user', 'item']:
                name = f'{prefix}_{entity}_factors.npy'
                np.save(out / name, getattr(fitted, f'{entity}_factors'))
                files.append(name)
        for entity, ids, column in [('user', users, 'visitorid'), ('item', items, 'itemid')]:
            name = f'als_{entity}_mapping.csv'
            pd.DataFrame({f'{entity}_index': range(len(ids)), column: ids}).to_csv(out / name, index=False)
            files.append(name)
        records = {'ALS': [], 'BPR': [], 'ALS+BPR': []}
        durations = {name: 0. for name in records}
        eval_users = pd.read_csv(out / 'evaluation_users.csv').visitorid.astype(int)
        for uid in eval_users:
            if uid not in user_map:
                continue  # Filtered reference user: kept in evaluation with zero recommendations.
            row = user_map[uid]
            start = time.perf_counter()
            als_scores = als.item_factors @ als.user_factors[row]
            als_score_time = time.perf_counter() - start
            append_ranking(records['ALS'], uid, items, als_scores, self.top_k)
            als_time = time.perf_counter() - start
            start = time.perf_counter()
            bpr_scores = bpr.item_factors @ bpr.user_factors[row]
            bpr_score_time = time.perf_counter() - start
            append_ranking(records['BPR'], uid, items, bpr_scores, self.top_k)
            bpr_time = time.perf_counter() - start
            start = time.perf_counter()
            scores = self.ensemble_alpha * minmax(als_scores) + (1-self.ensemble_alpha) * minmax(bpr_scores)
            append_ranking(records['ALS+BPR'], uid, items, scores, self.top_k)
            durations['ALS'] += als_time
            durations['BPR'] += bpr_time
            durations['ALS+BPR'] += als_score_time + bpr_score_time + time.perf_counter() - start
        for label, filename in [('ALS', 'als_recommendations.csv'), ('BPR', 'bpr_recommendations.csv'),
                                 ('ALS+BPR', 'als_bpr_recommendations.csv')]:
            recommendation_frame(records[label]).to_csv(out / filename, index=False)
            files.append(filename)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}
        stamp_artifacts(out, 'als', files, top_k=self.top_k, bpr_used=True,
            bpr_implementation='torch weighted pairwise loss; implicit BPR is not used',
            als_device='cpu', als_use_gpu=False, als_threads=self.als_threads,
            als_confidence='1 + alpha * behavior weight',
            behavior_weights=weight_map, time_decay=self.time_decay,
            negative_mode=self.negative_mode,
            config=config, ensemble_alpha=self.ensemble_alpha,
            training_seconds={'ALS': als_seconds, 'BPR': bpr_seconds, 'ALS+BPR': als_seconds + bpr_seconds},
            recommendation_seconds=durations)


@dataclass
class ReRanker:
    """ALS+BPR retrieval -> ALS/LightGCN fusion -> inventory adjustment -> sort."""
    processed_dir: Path = DEFAULT_PROCESSED
    data_dir: Path = DEFAULT_DATA
    top_k: int = 200
    als_candidate_k: int = 500
    als_weight: float = 0.4
    inventory_gamma: float = 1.5
    fusion: str = 'minmax'
    rerank_m: Optional[int] = None
    rrf_k: int = 60
    mf_score_source: str = 'ensemble'

    def run(self):
        out = self.processed_dir = Path(self.processed_dir)
        require_protocol(out)
        als_meta = require_artifacts(out, 'als')
        require_artifacts(out, 'gnn')
        if self.top_k < 1 or self.als_candidate_k < self.top_k:
            raise ValueError('Candidate depth must be at least the requested Top-K')
        factors = FactorScorer(out, als_meta['ensemble_alpha'])
        gnn = GNNScorer(out, factors.item_ids)
        users = pd.read_csv(out / 'evaluation_users.csv').visitorid.astype(int)
        features = pd.read_csv(out / 'feature_train_inliers.csv').set_index('itemid')
        available = features.reindex(factors.item_ids).has_available.eq(1).to_numpy()
        records, candidate_records = [], []
        retrieval_seconds = rerank_seconds = 0.
        for uid in users:
            if uid not in factors.users:
                continue
            start = time.perf_counter()
            als, _, retrieval = factors.scores(uid)
            candidates = top_indices(retrieval, factors.item_ids, self.als_candidate_k)
            retrieval_seconds += time.perf_counter() - start
            candidate_ids = factors.item_ids[candidates]
            start = time.perf_counter()
            graph_scores = gnn.scores(uid, candidates)
            if self.mf_score_source not in {'ensemble', 'als'}:
                raise ValueError('mf_score_source must be ensemble or als')
            mf_scores = retrieval if self.mf_score_source == 'ensemble' else als
            scores = hybrid_rerank_scores(mf_scores[candidates], graph_scores, available[candidates],
                                          self.als_weight, self.inventory_gamma, self.fusion,
                                          self.rrf_k, self.rerank_m)
            append_ranking(records, uid, candidate_ids, scores, self.top_k)
            rerank_seconds += time.perf_counter() - start
            candidate_records.extend((uid, int(factors.item_ids[i]), rank, float(retrieval[i]))
                for rank, i in enumerate(candidates, 1))
        recommendation_frame(records).to_csv(out / 'final_recommendations.csv', index=False)
        pd.DataFrame(candidate_records, columns=['visitorid', 'itemid', 'rank', 'retrieval_score']).to_csv(
            out / 'hybrid_candidates.csv', index=False)
        stamp_artifacts(out, 'rerank', ['final_recommendations.csv', 'hybrid_candidates.csv'],
            top_k=self.top_k, candidate_k=self.als_candidate_k, als_weight=self.als_weight,
            inventory_gamma=self.inventory_gamma, fusion=self.fusion, rerank_m=self.rerank_m,
            retrieval='ALS+BPR', mf_score_source=self.mf_score_source, inventory_before_sort=True,
            retrieval_seconds=retrieval_seconds, rerank_seconds=rerank_seconds)


@dataclass
class TopKExperiment:
    processed_dir: Path = DEFAULT_PROCESSED
    k_values: Tuple[int, ...] = (10, 20, 30, 50, 100, 150, 200)

    def run(self):
        return TestSetEvaluator(self.processed_dir, max(self.k_values)).run(k_values=self.k_values)


def main(argv=None):
    import argparse
    from datetime import datetime, timezone
    parser = argparse.ArgumentParser(description='Shared 80:20 warm-start recommendation experiment')
    parser.add_argument('stage', nargs='?', default='all',
                        choices=['all', 'preprocess', 'als', 'gnn', 'rerank', 'evaluate', 'topk'])
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--fixed-epochs', action='store_true')
    parser.add_argument('--iterations', type=int, default=256)
    parser.add_argument('--factors', type=int, default=128)
    parser.add_argument('--embedding-dim', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--bpr-epochs',type=int,default=128)
    parser.add_argument('--use-if',action='store_true')
    parser.add_argument('--core-mode',choices=['events','neighbors'],default='events')
    parser.add_argument('--binary-pairwise',action='store_true')
    parser.add_argument('--property-mode',choices=['values','names'],default='values')
    parser.add_argument('--metadata-policy',choices=['latest','history'],default='latest')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top-k', type=int, default=50, help='Primary figure/evaluation K')
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 20, 30, 50, 100, 150, 200])
    parser.add_argument('--candidate-k', type=int, default=500)
    parser.add_argument('--ensemble-alpha', type=float, default=.5)
    parser.add_argument('--als-weight', type=float, default=.4)
    parser.add_argument('--inventory-gamma', type=float, default=1.5)
    parser.add_argument('--if-contamination', default='auto',
                        help='Predeclared setting: auto or a fraction in (0, .5]')
    args = parser.parse_args(argv)
    depth = max([args.top_k, *args.k_values])
    if min([args.top_k, *args.k_values]) < 1 or args.candidate_k < depth:
        parser.error('Positive K values and candidate-k >= max(K) are required')
    if args.epochs < 1 or args.iterations < 1 or args.factors < 1 or args.embedding_dim < 1 or args.batch_size < 1:
        parser.error('Model dimensions, iterations and batch size must be positive')
    if not 0 <= args.ensemble_alpha <= 1 or not 0 <= args.als_weight <= 1 or args.inventory_gamma <= 0:
        parser.error('Weights must be in [0,1] and inventory gamma must be positive')
    contamination = args.if_contamination if args.if_contamination == 'auto' else float(args.if_contamination)
    if contamination != 'auto' and not 0 < contamination <= .5:
        parser.error('IF contamination must be auto or in (0, .5]')
    if args.stage in ['all', 'preprocess']:
        manifest = UnifiedPreprocessor(args.data_dir, args.processed_dir,
            random_state=args.seed, contamination=contamination,use_if=args.use_if,core_mode=args.core_mode).run()
        logger.info('Preprocessing: %d evaluation users; positive test coverage %.2f%%',
                    manifest['evaluation_users'], 100 * manifest['test_positive_coverage'])
    else:
        require_protocol(args.processed_dir)
    # Keep an append-only invocation record; artifact stamps contain actual model settings.
    import json
    invocation = vars(args) | {'data_dir': str(args.data_dir), 'processed_dir': str(args.processed_dir),
                              'time_utc': datetime.now(timezone.utc).isoformat()}
    with (args.processed_dir / 'commands.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(invocation) + '\n')
    if args.stage in ['all', 'als']:
        ALSRecommender(args.processed_dir, factors=args.factors, iterations=args.iterations,
            top_k=depth, random_state=args.seed, ensemble_alpha=args.ensemble_alpha,
            bpr_epochs=args.bpr_epochs,device=args.device or 'cpu',behavior_weighted=not args.binary_pairwise).run()
    if args.stage in ['all', 'gnn']:
        GNNEmbeddingGenerator(args.processed_dir, args.data_dir, embedding_dim=args.embedding_dim,
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device,
            early_stopping_patience=None if args.fixed_epochs else 25,
            behavior_weighted=not args.binary_pairwise,property_mode=args.property_mode,
            metadata_policy=args.metadata_policy).run()
    if args.stage in ['all', 'rerank']:
        ReRanker(args.processed_dir, args.data_dir, depth, args.candidate_k,
                 args.als_weight, args.inventory_gamma).run()
    if args.stage in ['all', 'evaluate', 'topk']:
        TestSetEvaluator(args.processed_dir, args.top_k).run(k_values=args.k_values)


if __name__ == '__main__':
    main()

