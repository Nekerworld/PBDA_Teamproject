"""Plot only measured CSV/JSON evidence; no embedded model performance values."""
import argparse,json,os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER=['ALS','BPR','ALS+BPR','GNN','Hybrid']
COLORS=['#64748b','#2b8cbe','#41ab5d','#8064a2','#d45b42']


def main(experiment,evidence,output):
    root=Path(experiment);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():
        import sys,platform,importlib.metadata as metadata,torch
        versions={name:metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','implicit','matplotlib','pytest']}
        environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'versions':versions,'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
        (root/'runtime_environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    if (root/'result.json').exists() and (not (root/'component_ablation.csv').exists() or not (root/'complementarity_summary.csv').exists()):
        from component_ablation import main as component_controls
        component_controls(root)
    facts=json.loads(Path(evidence).read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest={'sources':{},'figures':[]}
    import hashlib
    def source(p):manifest['sources'][str(p)]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source(evidence)
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=400,bbox_inches='tight')
        plt.close(fig);manifest['figures'].append(name)

    conditions=facts['conditions']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    stages=['train_before_filter','train_after_if','train_final']
    labels=['Before filtering','After IF step','After core-3']
    for name,color,label in [('if_events','#8a8a8a','IF + core'),('no_if_events','#2b8cbe','Core only')]:
        stats={r['stage']:r for r in conditions[name]['statistics']}
        axes[0].plot(labels,[int(stats[s]['events']) for s in stages],marker='o',color=color,label=label)
        axes[1].plot(labels,[int(stats[s]['transaction_events']) for s in stages],marker='o',color=color,label=label)
    axes[0].set_ylabel('Training event rows');axes[1].set_ylabel('Training purchase event rows')
    axes[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    axes[0].legend();axes[1].legend()
    axes[0].set_title('Training data retained');axes[1].set_title('Purchase signal retained')
    fig.suptitle('One chronological 80:20 partition, identical core rule')
    save(fig,'preprocessing_comparison')

    fig,ax=plt.subplots(figsize=(11,2.7));ax.axis('off')
    boxes=[(.02,'Chronological split\n80% development / 20% test'),(.28,'MF candidate retrieval\nALS + BPR'),(.54,'Graph reranking\nMF + LightGCN'),(.80,'Inventory weighting\nFinal Top-K')]
    for x,text in boxes:ax.text(x,.5,text,ha='left',va='center',transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=.6',fc='#edf4f8',ec='#487b96'),fontsize=9)
    for x in [.235,.495,.755]:ax.annotate('',xy=(x+.035,.5),xytext=(x,.5),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#487b96'})
    ax.text(.5,.05,'Selection uses inner validation only; final models refit on the pre-test development window.',ha='center',transform=ax.transAxes,fontsize=9)
    save(fig,'pipeline')

    if (root/'model_summary.csv').exists():
        source(root/'model_summary.csv');source(root/'experiment_metrics.csv')
        summary=pd.read_csv(root/'model_summary.csv')
        primary=summary.loc[summary.k==50].set_index('model').loc[ORDER]
        has_runtime=(root/'runtime_benchmark.csv').exists()
        if has_runtime:
            fig=plt.figure(figsize=(10,5.2))
            grid=fig.add_gridspec(2,2,height_ratios=[3,1])
            axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
        else:
            fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        for ax,metric,title in zip(axes,['ndcg','recall'],['NDCG@50','Recall@50']):
            bars=ax.bar(ORDER,primary[metric+'_mean'],yerr=primary[metric+'_std'],capsize=3,color=COLORS)
            ax.bar_label(bars,fmt='%.4f',padding=5,fontsize=9)
            ax.set_ylim(0,float((primary[metric+'_mean']+primary[metric+'_std']).max())*1.25)
            ax.set_title(title);ax.set_ylabel('Original-cohort macro average')
        if has_runtime:
            source(root/'runtime_benchmark.csv')
            runtime=pd.read_csv(root/'runtime_benchmark.csv').groupby('model').inference_batch_seconds.agg(['mean','std']).loc[ORDER]
            ax=fig.add_subplot(grid[1,:]);ax.axis('off')
            cells=[[f'{runtime.loc[m,"mean"]:.3f}' for m in ORDER]]
            table=ax.table(cellText=cells,colLabels=ORDER,rowLabels=['CPU batch (s)'],loc='center',cellLoc='center')
            table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.45)
        fig.suptitle('Common 80:20 test, three seeds; mean and sample SD; batch time excludes model loading')
        save(fig,'model_comparison')
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for model,color in zip(ORDER,COLORS):
            sub=summary.loc[summary.model==model].sort_values('k')
            for ax,metric in zip(axes,['ndcg','recall']):ax.plot(sub.k,sub[metric+'_mean'],marker='o',color=color,label=model)
        for ax,metric in zip(axes,['NDCG@K','Recall@K']):ax.set_xlabel('K');ax.set_ylabel(metric);ax.set_xticks([10,20,30,40,50]);ax.grid(alpha=.2)
        axes[1].legend(fontsize=8);save(fig,'topk_comparison')
        ab=summary.loc[(summary.k==50)&summary.model.isin(['Hybrid_ALS_only','Hybrid'])].set_index('model')
        fig,axes=plt.subplots(1,2,figsize=(8.5,3.4))
        for ax,metric in zip(axes,['ndcg','recall']):
            values=ab.loc[['Hybrid_ALS_only','Hybrid'],metric+'_mean']
            bars=ax.bar(['ALS-only fusion','ALS+BPR fusion'],values,color=['#94a3b8',COLORS[-1]])
            ax.bar_label(bars,fmt='%.4f',padding=4);ax.set_title(metric.upper()+'@50');ax.set_ylim(0,float(values.max())*1.2)
        fig.suptitle('Controlled scoring ablation: same trained models and candidate set')
        save(fig,'bpr_signal_ablation')
        source(root/'paired_differences.csv')
        paired=pd.read_csv(root/'paired_differences.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for ax,metric in zip(axes,['ndcg','recall']):
            sub=paired.loc[paired.metric==metric]
            means=sub.difference.to_numpy()
            ax.errorbar(means,np.arange(len(sub)),xerr=np.vstack([means-sub.ci_low,sub.ci_high-means]),fmt='o',color=COLORS[-1],capsize=4)
            ax.set_yticks(np.arange(len(sub)),sub.comparison.str.replace('Hybrid - ','',regex=False))
            ax.axvline(0,color='grey',ls='--');ax.set_xlabel('Hybrid minus baseline');ax.set_title(metric.upper()+'@50')
        fig.suptitle('Paired user bootstrap 95% CI, conditional on the three evaluated seeds')
        save(fig,'paired_differences')
    curves=[]
    for p in sorted((root/'trials').glob('*/gnn_training_history.json')):
        source(p)
        h=json.loads(p.read_text(encoding='utf-8'))
        for row in h['refit_history']:
            if 'validation_original_ndcg' in row:curves.append(dict(row,trial=p.parent.name))
    if curves:
        frame=pd.DataFrame(curves);frame.to_csv(out/'validation_training_curves.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for name,sub in frame.groupby('trial'):
            axes[0].plot(sub.epoch,sub.validation_original_ndcg,label=name)
            axes[1].plot(sub.epoch,sub.validation_original_recall,label=name)
        axes[0].set_ylabel('Validation original NDCG@50');axes[1].set_ylabel('Validation original Recall@50')
        for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        axes[1].legend(fontsize=7);save(fig,'validation_training')
    if (root/'selection.json').exists():
        selection=json.loads((root/'selection.json').read_text(encoding='utf-8'));source(root/'selection.json')
        entry=selection['hybrid'];p=root/f'validation_fusion_{entry["name"]}_{entry["checkpoint"]}.csv'
        source(p);frame=pd.read_csv(p)
        fig,ax=plt.subplots(figsize=(6.5,4.2))
        dots=ax.scatter(frame.recall,frame.ndcg,c=frame.omega,cmap='viridis',s=12,alpha=.4)
        cfg=entry['config'];ref=selection['validation_reference']
        ax.scatter([cfg['recall']],[cfg['ndcg']],marker='*',s=220,c='#d45b42',edgecolors='black',label='Selected Hybrid')
        ax.axvline(ref['recall'],color='grey',ls='--',label='Best baseline per metric')
        ax.axhline(ref['ndcg'],color='grey',ls='--')
        ax.set_xlabel('Validation Recall@50');ax.set_ylabel('Validation NDCG@50');ax.legend(fontsize=8)
        fig.colorbar(dots,ax=ax,label='MF fusion weight');save(fig,'validation_tradeoff')
        mask=np.ones(len(frame),dtype=bool)
        for key in ['alpha','candidate_k','strategy','fusion','omega']:
            mask &= np.isclose(frame[key],cfg[key]) if key in ['alpha','omega'] else frame[key].eq(cfg[key])
        gamma=frame.loc[mask].drop_duplicates('gamma').sort_values('gamma')
        gamma.to_csv(out/'gamma_sensitivity.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(9,3.3))
        for ax,metric in zip(axes,['ndcg','recall']):
            ax.plot(gamma.gamma,gamma[metric],marker='o',color=COLORS[-1]);ax.axvline(cfg['gamma'],color='grey',ls='--')
            ax.set_xlabel('Inventory gamma');ax.set_ylabel('Validation '+metric.upper()+'@50');ax.grid(alpha=.2)
        save(fig,'gamma_sensitivity')
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Figures:',manifest['figures'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.experiment,a.evidence,a.output)
