"""Create the Korean PDF briefing from measured experiment outputs.

Run with the Codex bundled document Python. Charts are generated separately by
plot_hybrid_results.py using the experiment environment's matplotlib.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether


def main(experiment,materials):
    root=Path(experiment);out=Path(materials);fig=out/'figures'
    result=json.loads((root/'result.json').read_text(encoding='utf-8'))
    selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    evidence=json.loads((out/'evidence.json').read_text(encoding='utf-8'))
    diagnosis=json.loads((out/'if_diagnosis.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(root/'model_summary.csv')
    primary=summary.loc[summary.k==50].set_index('model')
    metrics=pd.read_csv(root/'experiment_metrics.csv')
    pair=pd.read_csv(root/'paired_differences.csv')
    runtime=pd.read_csv(root/'runtime_benchmark.csv')
    components=pd.read_csv(root/'component_ablation.csv')
    complement=pd.read_csv(root/'complementarity_summary.csv')
    order=['ALS','BPR','ALS+BPR','GNN','Hybrid']
    success=result['success']
    robust=bool(selection.get('validation_seeds'))
    status='Hybrid가 두 지표의 세 seed 평균 모두 가장 높았습니다' if success else 'Hybrid의 두 지표 동시 1위 목표는 달성하지 못했습니다'
    hybrid=primary.loc['Hybrid']
    best={n:float(primary.loc[order[:-1],n+'_mean'].max()) for n in ['ndcg','recall']}
    lift={n:float(hybrid[n+'_mean']/best[n]-1)*100 for n in best}
    users=int(metrics.loc[metrics.model=='Hybrid','users'].iloc[0])
    missing=int(metrics.loc[metrics.model=='Hybrid','missing_users'].iloc[0])
    chosen=selection['hybrid'];cfg=chosen['config'];trial=chosen['trial']
    epoch=trial['baseline_epochs'] if chosen['checkpoint']=='baseline' else trial['hybrid_epochs']
    conclusion=f'{status}. RetailRocket의 공통 시간순 80:20 분할에서 seed 42·43·44의 평균을 비교했습니다. Hybrid의 NDCG@50은 {hybrid.ndcg_mean:.6f}, Recall@50은 {hybrid.recall_mean:.6f}입니다.'
    follow=f'각 지표의 가장 강한 비교 모델 대비 상대 차이는 NDCG {lift["ndcg"]:+.2f}%, Recall {lift["recall"]:+.2f}%입니다. 이 문서의 결과는 기존 논문의 점수를 재사용하지 않고 이번 실행에서 산출한 값입니다.'
    pdfmetrics.registerFont(TTFont('Malgun',r'C:\Windows\Fonts\malgun.ttf'))
    pdfmetrics.registerFont(TTFont('MalgunBold',r'C:\Windows\Fonts\malgunbd.ttf'))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('KO',fontName='Malgun',fontSize=9.2,leading=15,spaceAfter=7,wordWrap='CJK'))
    styles.add(ParagraphStyle('KOTitle',fontName='MalgunBold',fontSize=21,leading=29,spaceAfter=12,textColor=colors.black,wordWrap='CJK'))
    styles.add(ParagraphStyle('KOH',fontName='MalgunBold',fontSize=13,leading=20,spaceBefore=6,spaceAfter=9,textColor=colors.black,wordWrap='CJK'))
    styles.add(ParagraphStyle('KOSmall',fontName='Malgun',fontSize=7.8,leading=12,spaceAfter=6,wordWrap='CJK'))
    styles.add(ParagraphStyle('KOCell',fontName='Malgun',fontSize=8.2,leading=12,wordWrap='CJK'))
    story=[];md=[]
    def para(text,style='KO'):
        story.append(Paragraph(text,styles[style]));md.append(text.replace('<b>','').replace('</b>',''))
    def heading(text):para(text,'KOH');md.append('')
    def table(data,widths=None):
        cells=[[Paragraph(str(c),styles['KOCell']) for c in row] for row in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef2f5')),
            ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8193a2')),
            ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce2e7')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,9)])
        md.extend([' | '.join(map(str,row)) for row in data]);md.append('')
    def image(name,width=480):
        from PIL import Image as PILImage
        path=fig/(name+'.png')
        with PILImage.open(path) as im:w,h=im.size
        story.extend([Image(str(path),width=width,height=width*h/w),Spacer(1,6)])
        md.append(f'![{name}](figures/{name}.png)')
    def page():story.append(PageBreak());md.append('\n---\n')
    para('Hybrid 추천 모델 개정 실험 보고','KOTitle')
    date=pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y년 %m월 %d일')
    para(f'권유희 교수님 검토용 · {date} · 공통 시간순 분할과 실제 재실험 결과','KOSmall')
    para(conclusion);para(follow)
    recall_ci=pair.loc[(pair.comparison=='Hybrid - GNN')&(pair.metric=='recall')].iloc[0]
    uncertain_recall=recall_ci.ci_low<=0<=recall_ci.ci_high
    caution=(f'LightGCN 대비 Recall 평균 차이는 {recall_ci.difference:+.6f}이지만 사용자 bootstrap 95% 신뢰구간 [{recall_ci.ci_low:+.6f}, {recall_ci.ci_high:+.6f}]은 0을 포함합니다. 평균 1위와 통계적으로 확실한 Recall 우위는 구분해야 합니다.' if uncertain_recall else '')
    if caution:para(caution,'KOSmall')
    data=[['모델','NDCG@50 평균 ± SD','Recall@50 평균 ± SD']]
    for model in order:
        row=primary.loc[model]
        data.append([model,f'{row.ndcg_mean:.6f} ± {row.ndcg_std:.6f}',f'{row.recall_mean:.6f} ± {row.recall_std:.6f}'])
    table(data,[100,190,190])
    image('model_comparison')
    para(f'정답은 시험 구간의 장바구니·구매 상품 집합입니다. 모든 모델의 성적표 분모는 동일한 {users:,}명이며, Hybrid에서 추천 목록이 없는 {missing:,}명도 0점으로 포함했습니다. ±는 세 seed의 표본 표준편차이며 신뢰구간과 다릅니다.','KOSmall')
    para('평가 대상은 필터 전 학습에서 관측된 사용자 중 시험 양성 상품이 하나 이상 필터 전 학습 catalog에 있는 사람입니다. 그 사용자의 새 상품을 포함한 전체 시험 양성을 original 정답으로 둡니다. 전체 신규 사용자까지 포함한 서비스 이용자 전체의 결과는 아닙니다.','KOSmall')
    difference='두 모델이 맞히는 대상이 다르다는 점을 사용자별 추천 목록에서 확인했습니다.' if complement.mf_only_user.sum()+complement.gnn_only_user.sum()>0 else '두 모델의 정답 포함 사용자 집합은 같았으며 순위와 추가 정답 여부를 별도로 봐야 합니다.'
    para(f'각 seed에서 평균적으로 MF만 정답을 포함한 사용자는 {complement.mf_only_user.mean():.1f}명, GNN만 정답을 포함한 사용자는 {complement.gnn_only_user.mean():.1f}명이었습니다. {difference} 이 차이가 실제 Hybrid 이득으로 이어지는지는 뒤의 통제 비교와 신뢰구간을 함께 봐야 합니다.','KOSmall')
    page()
    heading('분할과 구매 제거 문제를 확인한 결과')
    para('학습 구간에서 validation을 분리하고 설정 선택 후 재학습하는 것은 정상입니다. 이번 분할은 상품 ID가 아닌 이벤트 기준입니다. 학습에 등장했던 상품만 점수를 낼 수 있으며, 미학습 상품은 original 정답에서 실패로 남깁니다.')
    split=evidence['splits']
    table([['구간','필터 전 이벤트','core 이후 학습 이벤트','용도'],
        ['내부 학습',f'{int(split["inner"]["train_before_filter"]["events"]):,}',f'{int(split["inner"]["train_final"]["events"]):,}','모델 학습'],
        ['내부 validation',f'{int(split["inner"]["test_before_filter"]["events"]):,}','해당 없음','epoch와 설정 선택'],
        ['최종 재학습',f'{int(split["outer"]["train_before_filter"]["events"]):,}',f'{int(split["outer"]["train_final"]["events"]):,}','설정 확정 후 재학습'],
        ['바깥 test',f'{int(split["outer"]["test_before_filter"]["events"]):,}','해당 없음','최종 성능 계산']], [105,110,130,135])
    legacy=evidence['legacy_reconstruction']
    support=evidence['evaluation_support']
    para(f'원본 코드 재구성 자료에서는 학습 pair와 validation pair의 직접 중복은 {legacy["train_validation_pair_overlap"]}개였습니다. 그러나 validation {legacy["pair_validation_rows"]:,}개가 모두 분할 전에 생성한 전파 그래프에 들어 있었습니다. 이번 구현은 내부 학습 구간만으로 그래프를 만듭니다.')
    para(f'original 평가자는 {support["reference_users"]}명이고, 사용자 벡터가 남는 사람은 {support["retained_user_vectors"]}명입니다. 정답 상품까지 남는 conditional 평가자 {support["users_with_reachable_truth"]}명과는 다릅니다. 현재 지원 범위에서 전체 카탈로그를 이상적으로 정렬한 Recall@50 상한은 {support["full_catalog_recall50_ceiling"]:.4f}입니다.','KOSmall')
    final_stats=split['outer']['train_final']
    para(f'최종 학습은 사용자 {int(final_stats["users"]):,}명, 상품 {int(final_stats["items"]):,}개, 이벤트 {int(final_stats["events"]):,}행입니다. 고유 사용자·상품 쌍은 {int(final_stats["unique_user_item_pairs"]):,}개이며 이 쌍을 기준으로 한 행렬 희소도는 {float(final_stats["sparsity_percent"]):.5f}%입니다. 반복 이벤트 행 수와 고유 쌍 수를 구분했습니다.','KOSmall')
    image('preprocessing_comparison')
    para('동일 8:2 구간에서 IF를 적용하면 학습 구매 17,864건이 IF 단계에서 모두 제거됐습니다. IF 없이 core만 적용하면 17,305건이 남습니다. 원본 코드 재구성에서도 학습 구매는 0건이므로 8:2 통일 뒤 새로 생긴 문제라고 단정하지 않습니다. 재구성 자료는 투고 당시 원본 실행 기록을 대신하지 않습니다.','KOSmall')
    ablation={r['condition']:r for r in diagnosis['results']}
    table([['학습 구간 IF 특징 비교','IF 이후 남은 구매'],
        ['원래 9개 특징',str(ablation['original_9']['kept_purchases'])],
        ['행동코드 평균과 표준편차 제외',f'{ablation["without_behavior_code"]["kept_purchases"]:,}'],
        ['행동코드 평균과 표준편차만 사용',str(ablation['behavior_code_only']['kept_purchases'])]], [350,130])
    para(f'원래 설정을 재적합한 라벨은 저장 라벨과 {ablation["original_9"]["agreement_with_cached_labels"]*100:.1f}% 일치했습니다. 행동코드 특징을 제외하면 구매 일부가 복구되지만 대부분은 여전히 제거됩니다. 따라서 행동코드만의 단독 문제로 설명할 수 없습니다.','KOSmall')
    page()
    heading('코드에서 바꾼 부분과 선택 근거')
    para('<b>MF 신호 보존</b> 원고 식 8은 후보 점수인 ALS+BPR과 GNN 점수를 결합합니다. 기존 코드는 최종 결합에 ALS만 사용했습니다. 이를 수정해 BPR이 후보 검색 이후에도 최종 순위에 기여하도록 했습니다.')
    para('<b>후보 검색</b> MF 결합 점수로 검색하는 방식과 ALS·BPR 후보를 균형 있게 합치는 방식을 비교했습니다. GNN으로 전체 카탈로그를 검색하는 방식으로 바꾸지는 않았습니다.')
    para('<b>공동 선택 기준</b> validation NDCG만 최대화하는 대신 NDCG와 Recall의 상대 개선율 중 작은 값을 최대화했습니다. 한 지표가 좋아지는 대신 다른 지표가 나빠지는 설정을 피하려는 기준입니다.')
    if robust:
        para('첫 v3 실행은 NDCG는 1위였지만 Recall은 LightGCN보다 낮았습니다. 이를 확인한 후 추가 연구로 seed 42·43·44의 내부 validation 평균으로 설정을 다시 선택했습니다. 기존 3층 LightGCN 비교 모델은 유지했습니다. 추가 설정 선택 함수에는 바깥 test 지표를 입력하지 않았습니다.','KOSmall')
    para('<b>GNN 학습</b> 64차원 2층·3층 LightGCN에서 행동 가중 InfoNCE를 사용했습니다. 음성 수는 32개, 온도는 0.2입니다. 균등·MF 기반 혼합 표본을 비교하고 학습 이력 상품을 음성에서 제외했습니다. 학습 MF도 같은 학습 구간만 사용합니다. 원고의 GNN BPR 손실 설명은 실제 목적함수에 맞춰 수정해야 합니다.')
    if cfg['fusion']=='rrf':
        para('<b>최종 융합</b> 선택된 RRF는 점수 크기 대신 후보 안에서의 MF 순위와 GNN 순위를 사용합니다. 각각 60을 더한 순위의 역수에 가중치를 곱해 합칩니다. 원고 식 8의 min-max 가중합을 그대로 사용했다고 쓰지 않고 이 방식으로 수정해야 합니다.')
    else:
        para('<b>최종 융합</b> 후보 안에서 MF와 GNN 점수를 각각 min-max 정규화한 뒤 가중합합니다. MF 점수는 ALS 단독이 아니라 ALS+BPR 결합 점수입니다.')
    image('pipeline')
    image('validation_tradeoff',width=370)
    para('점선은 각 지표의 가장 강한 validation 비교 모델을 나타냅니다. 별표는 공동 기준으로 선택한 Hybrid입니다. 점과 색은 실제 설정 탐색 CSV에서 생성했습니다.'+(' 이 그림은 세 validation seed의 평균입니다.' if robust else ''),'KOSmall')
    page()
    heading('학습과 하이퍼파라미터 선택 결과')
    table([['항목','선택값'],['GNN 조건',f'{trial["variant"]["layers"]}층 · {trial["variant"]["negative_mode"]} · 64차원'],
        ['Hybrid 학습 길이',str(epoch)+' epoch'],['후보 검색',cfg['strategy']],['후보 수',str(cfg['candidate_k'])],
        ['ALS 비중과 MF 융합 비중',f'후보 alpha {cfg["alpha"]} · 최종 omega {cfg["omega"]}'],
        ['최종 융합과 재고 가중',f'{cfg["fusion"]} · gamma {cfg["gamma"]}']], [180,300])
    image('validation_training')
    image('gamma_sensitivity')
    para('학습 곡선은 seed 42에서 epoch를 선택한 초기 내부 validation 이력입니다. gamma 민감도는 '+('세 validation seed 평균' if robust else '내부 validation')+'입니다. 최종 test 곡선으로 gamma나 epoch를 선택하지 않았습니다. gamma가 1이면 재고 가중은 중립입니다.','KOSmall')
    para('재고 가중은 학습 종료 전에 관측한 마지막 available 값을 사용하고 미상 상품에는 부스팅을 적용하지 않습니다. 시험 기간의 실시간 재고나 매출 효과를 검증한 결과는 아닙니다.','KOSmall')
    inventory=components.loc[(components.role=='validation')&components.model.isin(['Hybrid','Hybrid_no_inventory'])].set_index('model')
    para(f'seed 42 validation에서 실제 출력된 추천 항목 중 알려진 가용 상품 비율은 중립 가중에서 {inventory.loc["Hybrid_no_inventory","top50_available_share"]*100:.1f}%, 선택한 가중에서 {inventory.loc["Hybrid","top50_available_share"]*100:.1f}%였습니다. 이 항목 비율의 분모는 출력된 추천 수이며 성능 지표의 사용자 분모와 다릅니다.','KOSmall')
    page()
    heading('효과의 범위와 교수님 질문에 대한 답변')
    image('paired_differences')
    positive=pair.loc[pair.ci_low>0]
    para(f'사용자 단위 paired bootstrap 95% 개별 신뢰구간을 계산했으며 다중비교 보정은 하지 않았습니다. 8개 비교 중 하한이 0보다 큰 비교는 {len(positive)}개입니다. 이 구간은 평가한 세 seed에 조건부인 사용자 불확실성이며, 다른 시점이나 다른 데이터셋에 대한 일반화 증거는 아닙니다.')
    image('bpr_signal_ablation')
    controlled=components.loc[components.role=='test'].groupby('model')[['ndcg','recall']].mean()
    labels=[('Hybrid','Hybrid 전체'),('Hybrid_no_inventory','재고 가중 없음'),
        ('MF_only_same_inventory','GNN 점수 없음 같은 재고 가중'),
        ('GNN_candidates_same_inventory','MF 점수 없음 같은 후보와 재고 가중'),
        ('GNN_full_same_weights','동일 GNN 가중치 전체 상품 평가')]
    table([['통제 조건','NDCG@50','Recall@50']]+[[text,f'{controlled.loc[key,"ndcg"]:.6f}',f'{controlled.loc[key,"recall"]:.6f}'] for key,text in labels],[280,100,100])
    para('<b>Validation이 학습에 중복됐는가</b> 원본의 문제는 파일 pair 중복이 아니라 전파 그래프에 검증 관계가 남은 것이었습니다. 이번 검증 단계와 최종 재학습 단계는 분리되어 있습니다.')
    para('<b>왜 구매가 제거됐는가</b> 구매 이력이 있는 상품들이 IF 이상치로 분류되면서 해당 상품의 모든 이벤트가 제거됐습니다. 행동코드 두 특징을 제외하면 구매 3,476건이 남지만 14,388건은 여전히 제거됩니다. 행동코드와 나머지 특징들의 영향을 함께 봐야 하며, 봇을 식별한 결과로 설명하지 않습니다.')
    para('<b>원고에서 수정할 내용</b> 공통 8:2와 정답 집합, 식 6~9의 실제 점수 계산, 이분그래프 구조, 데이터 손실 통계, 성능과 실행시간 표, 그림 번호와 해상도를 함께 수정해야 합니다. 속성·카테고리 그래프 및 cold-start 해결 주장은 현재 구현에 맞게 조정해야 합니다.')
    page()
    heading('재현 방법과 판단의 한계')
    para('최종 설정을 selection.json으로 확정한 뒤 데이터·수치 학습 조건·seed·epoch가 맞는 모델을 연결했습니다. 이미 완료된 동일 조건의 재학습 모델은 재사용했으며 새로 학습한 것처럼 계산하지 않았습니다. 각 seed의 완료 파일은 새 설정 해시를 참조하고 추천·평가 CSV는 새로 생성했습니다. synthetic 검사는 실제 데이터 성능을 대신하지 않습니다.')
    para('재사용 모듈의 selection_metric 문구는 고정값이어서 실제 선택 방식과 다를 수 있습니다. 원본 메타데이터를 보존하고 effective_training_protocol.json에 실제 선택·고정 epoch 이력·재사용 출처를 연결했습니다. 여섯 모델 기록의 epoch·모드·분할이 새 설정과 같은지 독립 검산했습니다.','KOSmall')
    para(f'이번 선택 및 최종 평가 단계의 기록 시간은 {result["seconds_total"]/3600:.2f}시간입니다. 재사용 모델의 앞선 생성 시간은 이 값에 포함되지 않습니다. 학습 환경과 장비는 experiment_plan.json, 코드 및 데이터 식별자는 각 manifest에 기록했습니다.')
    para('과거 프로젝트의 test 결과와 v3의 Recall 미달 결과를 이미 검토한 후 추가 연구를 수행했습니다. 설정의 수치 선택에는 내부 validation만 사용했지만, 연구 방향 결정까지 포함해 시험과 완전히 독립적이었다고 주장하지 않습니다. 단일 데이터셋·단일 시간 구간·세 seed의 정정 평가라는 범위를 명시합니다.' if robust else '과거 프로젝트의 test 결과는 이번 작업 전에 이미 검토된 상태입니다. 이번 설정 탐색은 내부 validation에서만 진행했지만, 이 자료를 연구 전체에서 한 번도 관찰하지 않은 독립 시험이라고 부르지는 않습니다. 단일 데이터셋·단일 시간 구간·세 seed의 정정 평가라는 범위를 명시합니다.')
    para('투고 원고의 NDCG 0.3710과 Recall 0.7113은 이번과 데이터 분할·후보 처리·평가 분모가 다릅니다. 해당 숫자와 이번 결과를 같은 조건의 전후 성능으로 비교하거나 상대 개선율을 계산하지 않습니다.')
    para('데이터는 2015년 5~9월의 단일 전자상거래 서비스 로그입니다. 행동·상품·재고 분포가 다른 서비스나 현재 시점의 데이터로 일반화됐다는 증거는 아니며, 신규 사용자와 미관측 상품의 cold-start 성능도 입증하지 않았습니다.','KOSmall')
    endpoint=controlled.loc['GNN_candidates_same_inventory']
    if endpoint.recall>hybrid.recall_mean:
        para(f'동일 후보·재고 가중에서 MF 점수를 뺀 통제 조건의 Recall은 {endpoint.recall:.6f}로 Hybrid보다 높았지만 NDCG는 {endpoint.ndcg:.6f}로 낮았습니다. 따라서 모든 구성요소가 모든 지표를 높인다고 주장하지 않고, 주요 비교 모델 대비 평균 성능과 지표 사이의 균형을 구분해 해석합니다.','KOSmall')
    means=runtime.groupby('model')[['training_seconds','inference_batch_seconds','amortized_user_ms']].mean()
    table([['모델','학습 초','배치 추천 초','사용자당 환산 ms']]+[[m,f'{means.loc[m,"training_seconds"]:.1f}',f'{means.loc[m,"inference_batch_seconds"]:.3f}',f'{means.loc[m,"amortized_user_ms"]:.2f}'] for m in order], [100,110,130,140])
    active_users=int(runtime.active_users.iloc[0])
    para(f'추천 시간은 모델 로딩을 제외하고 실제 점수를 계산하는 {active_users}명의 배치를 CPU에서 3회 실행한 중앙값을 seed별로 구한 뒤 평균한 값입니다. 사용자당 값은 이 배치 시간을 {active_users}명으로 나눈 환산치이며 온라인 응답 지연시간이 아닙니다. MF 기반 음성 채굴을 사용한 GNN의 학습 시간에는 MF 의존 비용을 포함합니다.','KOSmall')
    different=int((runtime.total_lists-runtime.exact_matching_lists).sum())
    if different:
        delta=max(float(runtime.ndcg_delta_vs_saved.abs().max()),float(runtime.recall_delta_vs_saved.abs().max())) if 'ndcg_delta_vs_saved' in runtime else float('nan')
        para(f'후보 제한 행렬 계산과 저장 평가의 전체 행렬 계산 사이에 부동소수점 반올림으로 순위 목록 {different}개가 달랐습니다. 실행시간 측정 목록의 NDCG·Recall과 저장 평가 차이의 최대 절댓값은 {delta:.3g}입니다. 상세 목록은 runtime_rounding_differences.json에 보존했고, 본문 성능은 독립 검산한 저장 추천 CSV 기준입니다.','KOSmall')
    ratio=float(means.loc['Hybrid','inference_batch_seconds']/means.loc['GNN','inference_batch_seconds'])
    if ratio>1:
        para(f'이 측정에서 Hybrid의 배치 추천 시간은 GNN 단독의 {ratio:.2f}배였습니다. GNN 점수를 계산하는 후보 수가 줄어도 MF 검색 비용이 추가되므로 전체 속도 향상을 보장하지는 않습니다. 원고의 계산 효율 우위 주장은 이 측정 범위에 맞춰 수정해야 합니다.','KOSmall')
    else:
        para(f'이 측정에서 Hybrid의 배치 추천 시간은 GNN 단독의 {ratio:.2f}배였습니다. 이 비율은 저장된 임베딩을 이용한 해당 CPU 배치 조건의 결과이며, 온라인 서비스 전체 비용으로 일반화하지 않습니다.','KOSmall')
    folder='results/'+root.name
    filemap=[['파일 또는 폴더','내용'],['EXECUTION_PLAN_KO.md','실행 전 고정한 목표와 절차'],
        [folder+'/experiment_metrics.csv','모델별 seed별 Top-K 실제 지표'],
        [folder+'/selection.json','validation에서 선택한 설정과 hash'],
        [folder+'/runs','재학습 결과와 사용자별 평가'],
        ['professor_materials/evidence.json','분할 통계와 원본 문제의 근거'],
        ['professor_materials/figures','고해상도 PNG와 벡터 PDF SVG'],
        ['regression_final.log','최종 회귀검사 결과']]
    md.extend([' | '.join(row) for row in filemap])
    para('재현 명령과 파일별 위치는 같은 폴더의 README_KO.md와 상위 폴더의 README_HYBRID_KO.md에 정리했습니다. 결과 CSV, 실행 로그, 선택 근거와 소스는 함께 보존했습니다.','KOSmall')
    para('재실행 예시: '+('robust_hybrid.py에 --parent로 보존된 v3 결과 경로를 전달하고 --output-dir에 새 빈 폴더를 지정합니다. 최초 준비부터의 절차는 README_HYBRID_KO.md에 있습니다.' if robust else '지정 Python 환경에서 improve_hybrid.py에 --source-dir로 공통 no_if_events 준비 데이터 경로를 전달하고 --output-dir에 새 빈 결과 폴더를 지정합니다.')+' 기존 결과 디렉터리에 다른 코드나 설정으로 이어 쓰지 않습니다.','KOSmall')
    para('제안된 알고리즘 수정과 결과는 연구팀의 검토를 위한 자료입니다. 교수님께 직접 전송하거나 논문을 재투고하는 작업은 수행하지 않았습니다.','KOSmall')
    page()
    heading('부록: 전처리 수치와 실제 IF 입력')
    data=[['조건과 단계','사용자','상품','이벤트','구매']]
    for condition,label in [('if_events','IF 적용'),('no_if_events','IF 없음')]:
        stats={r['stage']:r for r in evidence['conditions'][condition]['statistics']}
        for stage,text in [('train_before_filter','필터 전'),('train_after_if','IF 단계 후'),('train_final','core 후')]:
            row=stats[stage]
            data.append([label+' '+text,*[f'{int(row[key]):,}' for key in ['users','items','events','transaction_events']]])
    table(data,[150,85,85,90,70])
    manifest=evidence['conditions']['if_events']['manifest']
    para(f'IF에서 제거된 상품은 {manifest["if_item_count"]:,}개로 학습 상품의 {manifest["if_item_percent_of_train_items"]:.2f}%입니다. 제거 이벤트는 {manifest["if_removed_events"]:,}행으로 필터 전 학습의 {manifest["if_event_percent_of_prefilter_train"]:.2f}%, 원시 전체의 {manifest["if_event_percent_of_raw_input"]:.2f}%입니다. 상품 비율과 이벤트 비율의 분모가 다릅니다.')
    para('이번 revision 구현과 재적합 진단의 IF 입력 9개: event_count, view_count, unique_visitors, event_code_mean, event_code_std, first_event_hour, last_event_hour, event_duration_hours, property_count. 원고의 17열 특징 표와 구분합니다. 투고 당시 원본 실행 파일은 보존되지 않아 당시 입력도 완전히 같았다고 단정하지 않습니다.')
    para('IF 설정: n_estimators=200, contamination=auto, max_samples=auto, random_state=42. 원래 9개 특징을 같은 학습 구간에 재적합해 저장 라벨과 일치하는지 확인한 뒤 특징 제외 실험을 수행했습니다.')
    para('MF 후보 점수는 사용자별 전체 학습 catalog에서 ALS와 BPR를 각각 min-max 정규화한 뒤 alpha 비중으로 합칩니다. 최종 minmax 결합은 후보 안에서 다시 정규화하며, RRF를 선택한 경우 후보 내 순위의 역수를 사용합니다. 재고 가중은 최종 결합 점수에 적용합니다.','KOSmall')
    para('공통 학습 설정: 행동 가중 view/addtocart/transaction=1/6/12, 시간 감쇠 반감기 7일. MF 128차원, ALS 256회·alpha 2·정규화 0.02, BPR 128epoch. GNN 64차원·학습률 0.001·정규화 0.0001·edge dropout 0.1·InfoNCE 음성 32개·온도 0.2. 세부 설정과 선택한 epoch는 소스와 selection.json에 보존했습니다.','KOSmall')
    para('core-3은 서로 다른 이웃 3개가 아닌 이벤트 행 3개 기준입니다. 학습에서 관측했던 상품의 재추천을 모든 모델에 공통으로 허용하며, 시험 양성은 사용자별 상품 집합으로 중복 제거해 평가합니다.','KOSmall')
    para('시간 경계 UTC: 내부 validation '+split['inner']['cutoff_utc']+' / 바깥 test '+split['outer']['cutoff_utc']+'. 특징·속성·전파 그래프와 음성 채굴 모델은 해당 학습 구간에서 구성했습니다.','KOSmall')
    audit_path=out/'legacy_9010_integrity.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        para(f'인접 90:10 로그는 실제 학습 {audit["train_events"]:,}행, 시험 {audit["test_events"]:,}행인데 manifest.config.test_size에는 {audit["manifest_config_test_size"]}가 남아 있습니다. 실제 건수와 설정 메타데이터가 불일치합니다. 이번 80:20과 섞지 않고 그대로 보존했습니다. 해당 로그의 학습 구매도 필터 전 {audit["train_purchases_before_filter"]:,}건에서 최종 {audit["train_purchases_after_filter"]}건입니다.')
    if robust:
        para('선택 과정: v1/v2는 내부 검증 학습과 후보 합집합 수정, v3는 후보 상한 확장과 첫 최종 시험, v4/v5는 첫 시험의 Recall 미달을 기록한 뒤 세 seed validation 평균으로 선택한 추가 실험입니다. v5는 v4와 같은 설정·선택 기준에서 중복 계산만 줄였습니다. 실패한 v3 결과와 소스도 보존했습니다.')
    parity_path=out/'fast_validation_parity.json'
    if parity_path.exists():
        parity=json.loads(parity_path.read_text(encoding='utf-8'))
        error=max(parity['maximum_metric_error'].values())
        para(f'검증 계산 가속은 실제 {parity["real_configurations"]}설정 전체와 대조했으며 지표 최대 차이는 {error:.2g}였습니다. 별도 확인한 사용자별 Top-50 {parity["exact_top50_lists_checked"]}개는 원래 전체 정렬과 일치했습니다. 최종 추천 CSV 생성과 독립 검산에는 기존 평가 계산을 사용했습니다.','KOSmall')
    def footer(canvas,doc):
        canvas.setFont('Malgun',7.5);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(42,24,'Hybrid 추천 모델 개정 실험 보고')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    pdf=out/'professor_briefing_ko.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=40,
        title='Hybrid 추천 모델 개정 실험 보고',author='PBDA 연구팀').build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'professor_briefing_ko.md').write_text('\n\n'.join(md),encoding='utf-8')
    index=f'# 교수님 설명 자료\n\n{conclusion}\n\n{follow}\n\n- professor_briefing_ko.pdf: 회의용 보고서\n- professor_briefing_ko.md: 수정 가능한 본문\n- figures/: 실제 지표 기반 그림\n- evidence.json: 원본 확인 근거\n'
    (out/'README_KO.md').write_text(index,encoding='utf-8')
    meeting=(f'# 교수님께 설명드릴 핵심 요약\n\n교수님, 기존 코드와 재구성 데이터를 확인하고 공통 시간순 8:2 조건에서 다시 실험했습니다. {conclusion}\n\n'
        '1. Validation을 train에서 분리한 것 자체가 오류는 아니었습니다. 기존 GNN은 분리 전에 전파 그래프를 만들어 validation 관계 139,306개가 그래프에 남아 있었습니다. 이번에는 선택 단계의 그래프를 내부 학습 구간만으로 구성했습니다.\n\n'
        '2. IF가 구매를 모두 제거하는 현상을 재현했습니다. 저장 라벨과 재적합 라벨이 100% 일치했습니다. 행동코드 평균·표준편차를 빼면 구매 3,476건은 남지만 14,388건은 여전히 제거됩니다. 따라서 봇 제거라고 설명하거나 행동코드 두 특징만의 문제라고 단정할 수 없습니다. 본실험은 IF 없이 공통 core-3을 적용해 구매 17,305건을 보존했습니다.\n\n'
        '3. 원고 식 8과 달리 최종 융합에서 BPR 신호가 빠지던 점을 수정했습니다. 후보 수·MF/GNN 비중·재고 가중은 내부 validation의 NDCG와 Recall 공동 기준으로 선택했습니다. 첫 v3는 Recall이 부족해 추가로 세 validation seed 평균에서 설정을 고정했고, 기존 단독 LightGCN은 유지했습니다. 재사용 가능한 모델은 보존하고 필요한 모델만 재학습했습니다.\n\n'
        f'4. {follow} {caution}\n\n'
        '5. 원고는 새 데이터 분할과 평가 분모, 이분그래프, 실제 점수 계산, 재고 관측 시점과 시간 측정 조건에 맞춰 수정해야 합니다. 기존 0.7대 Recall과 이번 값을 동일 조건의 전후 성능으로 비교하지 않았습니다. 과거 시험 결과를 이미 확인한 연구 과정이라는 한계도 기록했습니다.\n\n'
        '통제 비교에서는 MF 점수를 뺀 조건의 Recall이 전체 Hybrid보다 높고 NDCG는 낮았습니다. 모든 구성요소가 모든 지표를 높인다는 주장과, 주요 baseline 대비 두 지표 평균 1위라는 결과를 구분해야 합니다.\n\n'
        '자세한 표·그림과 통제 비교는 professor_briefing_ko.pdf에 정리했습니다. 이 문서는 전달용 초안이며 메일이나 카카오톡으로 전송하지 않았습니다.\n')
    (out/'MEETING_NOTES_KO.md').write_text(meeting,encoding='utf-8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file() and p.name!='delivery_manifest.json'}
    (out/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pdf)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--materials',required=True)
    a=p.parse_args();main(a.experiment,a.materials)
