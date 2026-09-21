"""Final Class-CL models, evaluated on cumulative 3/5/7-class views."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

# Use verified H5 channel IDs; anatomy-name remapping needs the source conversion map.
NAMES = {c: f'C{c}' for c in range(1, 8)}
COLORS = {1: '#ff657a', 2: '#40c9ff', 3: '#ffc857', 4: '#b991ff',
          5: '#56df9a', 6: '#ff974f', 7: '#ed7ed6'}


def dice(pred, target, labels):
    return [float((2 * ((pred == c) & (target == c)).sum() + 1e-5) /
                  ((pred == c).sum() + (target == c).sum() + 1e-5)) for c in labels]


def eligible(target, labels):
    return all(np.any(target == c) for c in labels)


def cumulative_view(mask, last_class):
    """Filter a completed argmax; never reassign an excluded prediction."""
    return np.where(mask <= last_class, mask, 0).astype(np.uint8)


def infer(config):
    import torch
    sys.path.insert(0, config['code'])
    from runner_core import H5Slices, _build_model, _loader, zs_forward
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    model = _build_model('class'); model.activate_stage(2); model.to('cuda:0').eval()
    methods = config['methods']; last = len(methods) - 1

    def load(method):
        state = torch.load(method['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model', state), strict=True)

    load(methods[-1])
    rows, candidates, arrays, aggregate = [], [], {}, {}
    with torch.no_grad():
        ds = H5Slices(Path(config['data']) / 'MMWHS' / 'whole_heart_test.h5', 'test')
        predictions, targets = [], []
        for batch, (image, gt) in enumerate(_loader(ds, 4, False, 0, 0)):
            predictions.append(zs_forward(model, image.cuda(), None)['pred_masks'].argmax(1).cpu().numpy().astype(np.uint8))
            targets.append(gt.numpy().astype(np.uint8))
            if (batch+1) % 100 == 0:
                print('inferred', (batch+1)*4, '/', len(ds), flush=True)
        pred, target = np.concatenate(predictions), np.concatenate(targets)
        assert int(ds.ends[-1]) + 1 == len(ds), 'patient boundaries must cover every test slice'
        assert set(np.unique(target)) == set(range(8)), 'expected global background plus seven class IDs'
        starts = [0] + [int(x)+1 for x in ds.ends[:-1]]
        whole_per_class = np.mean([dice(pred[s:int(e)+1], target[s:int(e)+1], range(1,8))
                                   for s,e in zip(starts,ds.ends)], axis=0)
        assert np.allclose(whole_per_class, config['expected_ours_whole_per_class'], atol=1e-5, rtol=0), whole_per_class
        for task, limit in [('T1',3), ('T2',5), ('T3',7)]:
            labels = list(range(1,limit+1))
            aggregate[task] = float(whole_per_class[:limit].mean())
            ranked = []
            for i in range(len(ds)):
                if not eligible(target[i], labels):
                    continue
                scores = dice(pred[i], target[i], labels)
                patient = int(np.searchsorted(ds.ends, i))
                row = dict(task=task, patient_index=patient, slice_index=i,
                           slice_in_patient=i-starts[patient], foreground_pixels=int(((target[i]>0)&(target[i]<=limit)).sum()),
                           ours_dice=float(np.mean(scores)))
                candidates.append(dict(row)); ranked.append(row)
            if not ranked:
                raise ValueError(f'No slice contains all {limit} cumulative classes')
            best = max(ranked, key=lambda r:(r['ours_dice'],r['foreground_pixels'],-r['slice_index']))
            i = best['slice_index']; image,_ = ds[i]
            arrays[task+'_image'] = image.numpy()[0]
            arrays[task+'_raw_gt'] = target[i]
            arrays[task+'_gt'] = cumulative_view(target[i], limit)
            arrays[task+'_raw_pred_'+str(last)] = pred[i]
            arrays[task+'_pred_'+str(last)] = cumulative_view(pred[i], limit)
            best.update(classes=labels, scores={methods[-1]['name']:best['ours_dice']},
                        per_class={methods[-1]['name']:dice(pred[i],target[i],labels)}, eligible_slices=len(ranked))
            rows.append(best)
            print(task,'best',i,'Dice',best['ours_dice'],'cumulative whole-test',aggregate[task],flush=True)
        ds.close()
        for j, method in enumerate(methods[:-1]):
            load(method)
            for row in rows:
                k=row['task']; image=torch.from_numpy(arrays[k+'_image'][None,None]).cuda()
                pred=zs_forward(model,image,None)['pred_masks'].argmax(1)[0].cpu().numpy().astype(np.uint8)
                arrays[k+'_raw_pred_'+str(j)]=pred
                arrays[k+'_pred_'+str(j)]=cumulative_view(pred,max(row['classes']))
                scores=dice(pred,arrays[k+'_raw_gt'],row['classes'])
                row['per_class'][method['name']]=scores
                row['scores'][method['name']]=float(np.mean(scores))
            print(method['name'],'done',flush=True)
    for row in rows:
        k=row['task'];limit=max(row['classes'])
        assert set(np.unique(arrays[k+'_gt'])) - {0} == set(row['classes'])
        for j,method in enumerate(methods):
            shown=arrays[k+'_pred_'+str(j)]
            assert int(shown.max())<=limit
            assert np.isclose(np.mean(dice(shown,arrays[k+'_gt'],row['classes'])),row['scores'][method['name']])
    with (out/'candidate_slices.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
    np.savez_compressed(out/'selected_slices.npz',**arrays)
    (out/'selection.json').write_text(json.dumps(dict(config=config,selection=rows,
        verified_ours_whole_per_class=whole_per_class.tolist(),
        cumulative_foreground_patient_mean=aggregate,
        test_slices=len(target), ground_truth_source='MMWHS/whole_heart_test.h5',
        selection_rule='Post-hoc maximum cumulative macro foreground Dice for 3/5/7 classes among slices containing every class in that view; larger target area then smaller index break ties.',
        index_convention='zero-based; displayed case numbers are one-based',
        prediction_rule='All rows use the final s03.pt model and argmax over all eight channels. Display only: map labels above 3/5/7 to background after argmax, equally for predictions and GT. No restricted-logit argmax.'),indent=2))


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    out=Path(directory);d=np.load(out/'selected_slices.npz')
    report=json.loads((out/'selection.json').read_text());methods=report['config']['methods']
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})

    def overlay(ax,mask):
        rgba=np.zeros((*mask.shape,4))
        for c,color in COLORS.items():
            rgba[mask==c]=(*to_rgb(color),.42)
            if np.any(mask==c) and not np.all(mask==c):
                ax.contour(mask==c,levels=[.5],colors=[color],linewidths=1)
        ax.imshow(rgba,interpolation='nearest')

    def panel(rows,zoom,name):
        columns=len(methods)+2
        fig,axes=plt.subplots(len(rows),columns,figsize=(2.65*columns,3.2*len(rows)+.85),squeeze=False)
        titles=['Image','Ground truth']+[m['name'].replace('-Sequential','\nSequential') for m in methods]
        titles[-1]='ScribbleCL (ours)\nZS-DER++ + MiB'
        for r,row in enumerate(rows):
            k=row['task'];im=d[k+'_image'];gt=d[k+'_gt']
            lo,hi=np.percentile(im[gt>0],[1,99])
            padding=max(float(hi-lo)*.25,1e-6);lo-=padding;hi+=padding
            for c,ax in enumerate(axes[r]):
                ax.imshow(im,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
                if c==1:overlay(ax,gt)
                elif c>=2:
                    overlay(ax,d[k+'_pred_'+str(c-2)])
                    for label in row['classes']:
                        ax.contour(gt==label,levels=[.5],colors=['white'],linewidths=.8,linestyles='dashed')
                    value=row['scores'][methods[c-2]['name']]
                    label=f'Dice {value:.3f}'
                    if 'checkpoint_stages' in row:
                        stage=row['checkpoint_stages'][methods[c-2]['name']]
                        label+=f'\nFinal T3' if c==columns-1 else f'\nAfter T{stage}'
                    ax.set_xlabel(label,fontsize=11,weight='bold',labelpad=6)
                if zoom:
                    yy,xx=np.where(gt>0);size=max(max(xx.max()-xx.min(),yy.max()-yy.min())*1.55,86)
                    cx=(xx.max()+xx.min())/2;cy=(yy.max()+yy.min())/2
                    ax.set_xlim(max(0,cx-size/2),min(im.shape[1]-1,cx+size/2))
                    ax.set_ylim(min(im.shape[0]-1,cy+size/2),max(0,cy-size/2))
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if r==0:ax.set_title(titles[c],fontsize=12,weight='bold',pad=12)
            axes[r,0].set_ylabel(f'{k} cumulative: C1–C{max(row["classes"])}\nCase {row["patient_index"]+1}, slice {row["slice_in_patient"]}',fontsize=11)
        legend=[Patch(color=color,label=NAMES[c]) for c,color in COLORS.items()]
        legend.append(Line2D([0],[0],color='#555555',ls='--',label='White dashed: task GT'))
        fig.legend(handles=legend,loc='lower center',ncol=8,frameon=False,bbox_to_anchor=(.5,.035),fontsize=11)
        caption='Same final T3 model in every row | Cumulative 3 / 5 / 7 classes | Full argmax, then display filtering | Post-hoc best cases'
        if report.get('comparison_protocol')=='baselines_at_task_end_ours_final':
            caption='Baselines: after each row task | Ours: final T3 | Post-hoc slices favoring Ours; nonempty baseline foreground required'
        fig.text(.5,.014,caption,ha='center',fontsize=10,color='#444444')
        fig.subplots_adjust(left=.055,right=.997,top=.91 if len(rows)>1 else .77,bottom=.135 if len(rows)>1 else .30,wspace=.045,hspace=.34)
        fig.savefig(out/(name+'.png'),dpi=180,facecolor='white')
        fig.savefig(out/(name+'.pdf'),facecolor='white');plt.close(fig)
    panel(report['selection'],False,'class_best_full')
    panel(report['selection'],True,'class_best_zoom')
    for row in report['selection']:panel([row],True,'class_'+row['task'])


if __name__=='__main__':
    if sys.argv[1]=='self-check':
        target=np.array([[1,2],[0,0]],dtype=np.uint8)
        assert dice(target,target,[1,2])==[1.,1.]
        assert max(dice(np.array([[2,1],[0,0]]),target,[1,2]))<1e-4
        assert eligible(target,[1,2]) and not eligible(target,[1,2,3])
        # Later-class predictions remain errors for earlier task classes.
        assert max(dice(np.full_like(target,7),target,[1,2]))<1e-4
        raw=np.array([[1,4,7],[2,5,6]],dtype=np.uint8)
        assert np.array_equal(cumulative_view(raw,3),[[1,0,0],[2,0,0]])
        assert np.array_equal(cumulative_view(raw,5),[[1,4,0],[2,5,0]])
        assert np.array_equal(cumulative_view(raw,7),raw)
        for limit in (3,5,7):
            assert dice(raw,raw,range(1,limit+1))==dice(cumulative_view(raw,limit),cumulative_view(raw,limit),range(1,limit+1))
        print('self-check passed')
    elif sys.argv[1]=='infer':infer(json.loads(Path(sys.argv[2]).read_text()))
    elif sys.argv[1]=='render':render(sys.argv[2])
