"""Seal only verified, visually reviewed deliverables and copy convenient outputs."""
import argparse,hashlib,json,shutil,zipfile,re
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(root,outputs):
    root=Path(root);outputs=Path(outputs);outputs.mkdir(parents=True,exist_ok=True)
    experiment=root/'results/hybrid_joint_8020_v5'
    materials=root/'professor_materials'
    verification=json.loads((materials/'verification.json').read_text(encoding='utf-8'))
    assert verification['verified'] and verification['metric_pairs_replayed']==90
    assert verification['selection_hash']==sha(experiment/'selection.json')
    assert verification['result_hash']==sha(experiment/'result.json')
    assert verification['effective_training_protocol_hash']==sha(experiment/'effective_training_protocol.json')
    result=json.loads((experiment/'result.json').read_text(encoding='utf-8'))
    assert verification['success']==result['success']
    qa=json.loads((materials/'qa/visual_review.json').read_text(encoding='utf-8'))
    assert qa['approved'] and qa['pdf_sha256']==sha(materials/'professor_briefing_ko.pdf')
    render=json.loads((materials/'qa/render_manifest.json').read_text(encoding='utf-8'))
    assert qa['reviewed_pages']==list(range(1,render['pages']+1))
    for item in render['images']:
        image=Path(item['image'])
        if not image.is_absolute():image=root/image
        assert qa['page_images'][str(item['page'])]==sha(image)
    figures=json.loads((materials/'figures/figure_manifest.json').read_text(encoding='utf-8'))
    assert set(qa['figures'])==set(figures['figures'])
    for name,digest in qa['figures'].items():assert digest==sha(materials/'figures'/(name+'.png'))
    for name,digest in figures['sources'].items():
        path=Path(name)
        if not path.is_absolute():path=root/path
        assert sha(path)==digest,f'Stale plot source: {name}'
    plan=json.loads((experiment/'experiment_plan.json').read_text(encoding='utf-8'))
    assert set(plan['code'])=={p.name for p in root.glob('*.py')},'Untracked training source added'
    for name,digest in plan['code'].items():
        assert sha(root/name)==digest,f'Production source changed after experiment start: {name}'
    testlog=(root/'regression_final.log').read_text(encoding='utf-8',errors='replace')
    matched=re.search(r'(\d+) passed',testlog)
    assert matched and int(matched.group(1))>=46 and 'FAILED ' not in testlog
    metrics={row['model']:row for row in result['primary']}
    hybrid=metrics['Hybrid']
    status='두 지표의 세 seed 평균이 모든 비교 모델보다 높았다.' if result['success'] else '두 지표 동시 1위 목표는 미달했다.'
    index=('# Hybrid 개정 실험 최종 자료\n\n'
        '실제 코드 수정, RetailRocket 재실험, 시각화, 교수님 설명 자료를 생성하고 검산·시각 검수를 마쳤다. '+status+'\n\n'
        f'공통 시간순 80:20, seed 42/43/44, original 평가 사용자 {verification["users"]}명. Hybrid NDCG@50 {hybrid["ndcg_mean"]:.6f}, Recall@50 {hybrid["recall_mean"]:.6f}.\n\n'
        '- professor_materials/professor_briefing_ko.pdf: 교수님 검토용 보고서\n'
        '- professor_materials/MEETING_NOTES_KO.md: 설명 요약 초안\n'
        '- professor_materials/figures: 실제 수치의 PNG/PDF/SVG\n'
        '- results/hybrid_joint_8020_v5: 최종 설정, 지표, 사용자별 추천·검산 입력\n'
        '- README_HYBRID_KO.md: 재실행 방법, 조건과 한계\n'
        '- CODE_CHANGES_KO.md / CODE_CHANGES.patch: 실제 변경 검토\n\n'
        f'회귀검사 {matched.group(1)}개 통과, 추천 CSV의 90개 지표 쌍 독립 재계산, 원래 ALS/BPR/GNN baseline 추천 파일 일치 확인.\n\n'
        '첫 v3의 Recall 미달 후 추가 연구를 수행한 사실을 공개했다. 단일 시기·단일 데이터셋의 정정 평가이며 완전히 미관찰 시험이나 일반화 우위를 주장하지 않는다. 이전 소스와 실패 결과를 보존했다.\n\n'
        '큰 모델 배열·원시 학습 데이터는 프로젝트에 남겨 두었고 ZIP에서는 제외했다. ZIP의 경량 추천·정답 CSV로 독립 지표 검산은 가능하다. 교수님께 전송하거나 논문을 재투고하지 않았다.\n')
    (root/'README.md').write_text(index,encoding='utf-8')
    files=[*root.glob('*.py'),root/'README.md',root/'EXECUTION_PLAN_KO.md',root/'README_HYBRID_KO.md',root/'SEARCH_EXTENSION_KO.md',root/'ROBUST_SELECTION_KO.md',root/'SCORING_OPTIMIZATION_KO.md',root/'requirements.txt',root/'regression_final.log',
        *list((root/'report_tools').glob('*.py'))]
    for p in materials.rglob('*'):
        if p.is_file() and p.name!='final_delivery_manifest.json' and 'qa' not in p.relative_to(materials).parts and p.suffix.lower() in ['.pdf','.png','.svg','.csv','.json','.md']:
            files.append(p)
    for p in experiment.glob('*'):
        if p.is_file() and p.suffix.lower() in ['.csv','.json']:files.append(p)
    for p in (experiment/'runs').glob('*/complete.json'):files.append(p)
    # Keep the complete light-weight metric replay inputs in the ZIP. Large
    # factor/embedding arrays and raw training events stay at the project.
    for directory in (experiment/'runs').glob('seed_*'):
        for pattern in ['recommendations_*.csv','metrics_*.csv','manifest.json','evaluation_users.csv','candidate_items.csv','events_test_original.csv']:
            files.extend(directory.glob(pattern))
        files.extend(directory.glob('*/gnn_artifacts.json'))
        files.extend(directory.glob('*/gnn_training_history.json'))
    for p in (experiment/'trials').glob('*/trial_complete.json'):files.append(p)
    for p in (experiment/'trials').glob('*/*.json'):files.append(p)
    for parent,archive in [('hybrid_joint_8020_v1','source_v1_before_union_fix'),
                           ('hybrid_joint_8020_v2','source_v2_before_boundary_expansion'),
                           ('hybrid_joint_8020_v3','source_v3_before_robust_selection'),
                           ('hybrid_joint_8020_v4','source_v4_before_scoring_optimization')]:
        directory=root/'results'/parent
        archived=root/archive
        parent_plan=json.loads((directory/'experiment_plan.json').read_text(encoding='utf-8'))
        for name,expected in parent_plan['code'].items():
            assert sha(archived/name)==expected,f'Historical source mismatch: {parent}/{name}'
        files.extend(archived.glob('*.py'))
        files.append(directory/'experiment_plan.json')
        if (directory/'SUPERSEDED.json').exists():files.append(directory/'SUPERSEDED.json')
        for name in ['result.json','selection.json','model_summary.csv','experiment_metrics.csv','paired_differences.csv','effective_training_protocol.json']:
            if (directory/name).exists():files.append(directory/name)
        if parent=='hybrid_joint_8020_v3':
            for seed in [42,43,44]:
                run=directory/'runs'/f'seed_{seed}'
                for pattern in ['recommendations_*.csv','metrics_*.csv','manifest.json','evaluation_users.csv','candidate_items.csv','events_test_original.csv','complete.json']:
                    files.extend(run.glob(pattern))
                files.extend(run.glob('*/gnn_artifacts.json'));files.extend(run.glob('*/gnn_training_history.json'))
    for p in (root/'report_history/v3').glob('*'):
        if p.is_file():files.append(p)
    files.extend((root/'source_before_hybrid_20261017').glob('*.py'))
    for name in ['CODE_CHANGES.patch','CODE_CHANGES_KO.md','CODE_CHANGES_MANIFEST.json']:
        if (root/name).exists():files.append(root/name)
    for p in experiment.glob('validation_seed_*/*/*_artifacts.json'):files.append(p)
    for p in experiment.glob('validation_seed_*/*/gnn_training_history.json'):files.append(p)
    for p in experiment.glob('validation_seed_*/*/bpr_training_history.json'):files.append(p)
    manifest={str(p.relative_to(root)):sha(p) for p in sorted(set(files))}
    (root/'code_manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in root.rglob('*.py')
        if 'source_' not in str(p.relative_to(root)) and 'test_artifacts' not in p.parts},indent=2),encoding='utf-8')
    (materials/'final_delivery_manifest.json').write_text(json.dumps({'verified':verification,'qa':qa,'files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    archive=outputs/'hybrid_revision_deliverables.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):
            # Prior analysis files remain at the project, but current delivery is explicit.
            z.write(p,str(p.relative_to(root)))
        z.write(root/'code_manifest.json','code_manifest.json')
        z.write(materials/'final_delivery_manifest.json','professor_materials/final_delivery_manifest.json')
        license_path=root.parents[1]/'LICENSE'
        if license_path.exists():z.write(license_path,'LICENSE')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    for src,name in [(materials/'professor_briefing_ko.pdf','Hybrid_Revision_Report_KO.pdf'),
        (materials/'MEETING_NOTES_KO.md','Professor_Meeting_Notes_KO.md'),
        (materials/'figures/model_comparison.png','Hybrid_Model_Comparison.png'),
        (experiment/'model_summary.csv','Hybrid_Model_Summary.csv')]:
        shutil.copy2(src,outputs/name)
        assert sha(src)==sha(outputs/name)
    print(json.dumps({'outputs':[str(p) for p in outputs.iterdir() if p.is_file()],
        'archive_files':len(files),'success':verification['success']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--outputs',required=True)
    a=p.parse_args();main(a.root,a.outputs)
