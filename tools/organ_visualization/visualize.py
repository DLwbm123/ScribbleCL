import json
import sys
from pathlib import Path
import numpy as np

def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    out = Path(directory)
    data = np.load(out / 'selected_slices.npz')
    report = json.loads((out / 'selection.json').read_text())
    methods = report['config']['methods']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    names = ['Image', 'Ground truth'] + [m['name'].replace('-Sequential', '\nSequential') for m in methods]
    names[-1] = 'ScribbleCL (ours)\nZS-DER++'

    def overlay(ax, mask):
        mask = mask.astype(bool)
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = (*to_rgb('#ff657a'), .42)
        if mask.any() and not mask.all():
            ax.contour(mask, levels=[.5], colors=['#ff657a'], linewidths=1)
        ax.imshow(rgba, interpolation='nearest')

    def panel(rows, zoom, filename):
        fig, axes = plt.subplots(len(rows), len(methods)+2, figsize=(20, 2.8 * len(rows) + .5), squeeze=False)
        for r, row in enumerate(rows):
            code = row['task']; im = data[code + '_image']; gt = data[code + '_gt']
            lo, hi = np.percentile(im, [1, 99])
            for c, ax in enumerate(axes[r]):
                ax.imshow(im, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
                if c == 1:
                    overlay(ax, gt)
                elif c >= 2:
                    pred = data[code + '_pred_' + str(c-2)]
                    overlay(ax, pred)
                    ax.contour(gt, levels=[.5], colors=['white'], linewidths=.8, linestyles='dashed')
                    score = row['scores'][methods[c-2]['name']]
                    stage=row['checkpoint_stages'][methods[c-2]['name']]
                    label=f'Dice {score:.3f}\n'+('Final T4' if c==len(methods)+1 else f'After T{stage}')
                    ax.set_xlabel(label, fontsize=11, weight='bold', labelpad=6)
                if zoom:
                    y, x = np.where(gt)
                    size = max(int(max(x.max()-x.min(), y.max()-y.min()) * 1.7), 72)
                    cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
                    ax.set_xlim(max(0,cx-size/2), min(im.shape[1]-1,cx+size/2))
                    ax.set_ylim(min(im.shape[0]-1,cy+size/2), max(0,cy-size/2))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                if r == 0: ax.set_title(names[c], fontsize=11, weight='bold', pad=10)
            axes[r,0].set_ylabel(f'{code}: {row["organ"]}\nCase {row["patient_index"]+1}, slice {row["slice_in_patient"]}\nTest index {row["slice_index"]}', fontsize=10)
        fig.legend([Patch(color='#ff657a', alpha=.42), Line2D([0],[0],color='#555555',ls='--')],
                   ['Foreground', 'White dashed: ground truth'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5,.025))
        if len(rows) > 1:
            fig.suptitle('Organ-CL | Selected qualitative comparison', fontsize=16, weight='bold', y=.995)
        fig.text(.5,.009,'Selected for stronger ZS baselines below Ours | Baselines: after each task | Ours: final T4 | Same slice and display scale',ha='center',fontsize=9,color='#555555')
        fig.subplots_adjust(left=.067,right=.995,top=.92 if len(rows)>1 else .78,bottom=.12 if len(rows)>1 else .28,wspace=.035,hspace=.36)
        fig.savefig(out / (filename+'.png'), dpi=190, facecolor='white')
        fig.savefig(out / (filename+'.pdf'), facecolor='white')
        plt.close(fig)
    panel(report['selection'], False, 'organ_best_full')
    panel(report['selection'], True, 'organ_best_zoom')
    for row in report['selection']:
        panel([row], True, 'organ_' + row['task'])


def dice(p,y):
    p=p.astype(bool);y=y.astype(bool)
    return float((2*(p&y).sum()+1e-5)/(p.sum()+y.sum()+1e-5))


def choose(rows):
    valid=[r for r in rows if min(r['pixels'])>=32 and min(r['scores'])>=.01 and r['ours']>max(r['scores'][:-1])]
    if not valid:
        raise ValueError('No candidate meets visible foreground and Ours superiority; inspect candidate scores')
    return max(valid,key=lambda r:(min(r['scores'][2:5]),r['disagreement'],r['ours'],-r['index']))


def infer(config):
    import torch
    sys.path.insert(0,config['code'])
    import runner_core as core
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    out=Path(config['output']);out.mkdir(exist_ok=True,parents=True)
    methods=config['methods'];arrays={};selected=[];verified={}
    for task in config['tasks']:
        key=task['code'];stage=task['head']+1
        if hasattr(core.H5Slices,'cache_arrays'):core.H5Slices.cache_arrays=True
        ds=core.H5Slices(Path(config['data'])/task['file'],'test',label_shift=0)
        samples=[ds[i] for i in range(len(ds))]
        images=np.stack([x.numpy() for x,y in samples]);labels=np.stack([y.numpy() for x,y in samples])
        assert set(np.unique(labels))<={0,1}
        targets=labels.astype(bool);ends=np.asarray(ds.ends)
        assert int(ends[-1])+1==len(images)
        predictions=[];checkpoints={};stages={};aggregate={}
        for j,method in enumerate(methods):
            model_stage=4 if j==len(methods)-1 else stage
            ckpt=Path(method['checkpoint']).with_name(f's{model_stage:02}.pt')
            model=core._build_model('organ');model.activate_stage(model_stage-1)
            state=torch.load(ckpt,map_location='cpu');model.load_state_dict(state.get('model',state),strict=True)
            model.to('cuda:0').eval();del state
            pred=[]
            with torch.no_grad():
                for start in range(0,len(images),4):
                    values=core.zs_forward(model,torch.from_numpy(images[start:start+4]).cuda(),task['head'])['pred_masks']
                    assert values.shape[1]==2
                    pred.append(values.argmax(1).cpu().numpy().astype(bool))
            pred=np.concatenate(pred)
            del model;torch.cuda.empty_cache()
            predictions.append(pred)
            patient_scores=[dice(pred[a:b],targets[a:b]) for a,b in zip([0,*list(ends[:-1]+1)],ends+1)]
            aggregate[method['name']]=float(np.mean(patient_scores))
            expected=method['expected'][key]
            assert abs(aggregate[method['name']]-expected)<1e-5,(key,method['name'],aggregate[method['name']],expected)
            checkpoints[method['name']]=str(ckpt);stages[method['name']]=model_stage
            print('VERIFIED',key,method['name'],len(images),aggregate[method['name']],flush=True)
        np.savez_compressed(out/f'{key}_candidates.npz',images=images[:,0],gt=targets,predictions=np.stack(predictions),ends=ends)
        verified[key]=aggregate
        own=predictions[-1];eligible=[i for i,y in enumerate(targets) if y.sum()>=32]
        best=max(dice(own[i],targets[i]) for i in eligible)
        rows=[]
        for i in eligible:
            scores=[dice(p[i],targets[i]) for p in predictions]
            if scores[-1]<.70*best:continue
            yy,xx=np.where(targets[i]);size=max(int(max(xx.max()-xx.min(),yy.max()-yy.min())*1.7),72)
            cx=(xx.max()+xx.min())/2;cy=(yy.max()+yy.min())/2
            crop=(slice(max(0,int(cy-size/2)),min(256,int(cy+size/2)+1)),slice(max(0,int(cx-size/2)),min(256,int(cx+size/2)+1)))
            differences=[]
            for a in range(5):
                for b in range(a+1,5):
                    x=predictions[a][i][crop];y=predictions[b][i][crop]
                    differences.append(float((x!=y).sum()/max(int((x|y).sum()),1)))
            rows.append(dict(index=i,ours=scores[-1],scores=scores,pixels=[int(p[i].sum()) for p in predictions],disagreement=min(differences)))
        (out/f'{key}_scores.json').write_text(json.dumps(rows,indent=2))
        row=choose(rows);i=row['index'];patient=int(np.searchsorted(ends,i));start=0 if patient==0 else int(ends[patient-1])+1
        selected.append(dict(task=key,organ=task['organ'],slice_index=i,patient_index=patient,slice_in_patient=i-start,
                            scores=dict(zip([m['name'] for m in methods],row['scores'])),checkpoint_stages=stages,checkpoints=checkpoints,
                            ours_floor=.70*best,candidates=len(rows),baseline_min_disagreement=row['disagreement']))
        arrays[key+'_image']=images[i,0];arrays[key+'_gt']=targets[i]
        for j,p in enumerate(predictions):arrays[key+'_pred_'+str(j)]=p[i]
        print('SELECTED',key,i,row['scores'],flush=True)
    np.savez_compressed(out/'selected_slices.npz',**arrays)
    (out/'selection.json').write_text(json.dumps(dict(config=config,selection=selected,verified_foreground_patient_means=verified,
        selection_rule='GT >=32 pixels; Ours >=70% of best eligible slice; every method >=32 predicted pixels and Dice >=0.01; Ours strictly exceeds every baseline; maximize weakest ZS-Sequential/EWC/GPM Dice, ties use minimum pairwise baseline foreground disagreement, Ours Dice, lower index. Qualitative post-hoc selection.'),indent=2))
    render(out)


if __name__=='__main__':
    if sys.argv[1]=='render':render(sys.argv[2])
    elif sys.argv[1]=='infer':infer(json.loads(Path(sys.argv[2]).read_text()))
    elif sys.argv[1]=='self-check':
        a=np.array([[1,0],[0,0]],dtype=bool);assert dice(a,a)==1 and dice(a,~a)<1e-4
        rows=[dict(index=0,ours=.9,scores=[.5,.5,.6,.6,.6,.9],pixels=[50]*6,disagreement=.4),dict(index=1,ours=.9,scores=[.5,.5,.7,.7,.7,.9],pixels=[50]*6,disagreement=.1)]
        assert choose(rows)['index']==1
        rows[1]['scores'][0]=.95;assert choose(rows)['index']==0
        print('self-check passed')
