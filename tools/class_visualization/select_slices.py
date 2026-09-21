"""Qualitative selection: acquisition baselines versus final Ours."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from visualize import dice, cumulative_view, render


def mask_disagreement(predictions, target, limit):
    """Class-aware foreground disagreement inside the displayed crop."""
    yy,xx=np.where((target>0)&(target<=limit))
    size=max(max(xx.max()-xx.min(),yy.max()-yy.min())*1.55,86)
    cx=(xx.max()+xx.min())/2;cy=(yy.max()+yy.min())/2
    crop=(slice(max(0,int(np.floor(cy-size/2))),min(target.shape[0],int(np.ceil(cy+size/2))+1)),
          slice(max(0,int(np.floor(cx-size/2))),min(target.shape[1],int(np.ceil(cx+size/2))+1)))
    pairs=[]
    for a in range(len(predictions)):
        for b in range(a+1,len(predictions)):
            x=cumulative_view(predictions[a],limit)[crop]
            y=cumulative_view(predictions[b],limit)[crop]
            foreground=(x>0)|(y>0)
            pairs.append(float(((x!=y)&foreground).sum()/max(int(foreground.sum()),1)))
    return pairs


def select_row(rows, mode="diverse"):
    feasible=[r for r in rows if min(r['baseline_foreground_pixels'])>=32 and min(r['baseline_scores'])>=.01]
    if not feasible:
        raise ValueError('No slice satisfies visible foreground and nonzero overlap for all baselines')
    if rows[0].get('task') in ('T2','T3'):
        feasible=[r for r in feasible if r['ours_dice']-max(r['baseline_scores'])>=.10]
        if not feasible:
            raise ValueError('No diverse-mode candidate retains an Ours advantage of 0.10')
        if mode=='stronger_zs':
            return max(feasible,key=lambda r:(min(r['baseline_scores'][2:5]),
                       min(r['baseline_pair_disagreement']),r['ours_dice'],-r['slice_index']))
        return max(feasible,key=lambda r:(min(r['baseline_pair_disagreement']),
                    float(np.mean(r['baseline_pair_disagreement'])),
                    r['ours_dice']-max(r['baseline_scores']),r['ours_dice'],-r['slice_index']))
    return max(feasible,key=lambda r:(r['ours_dice']-max(r['baseline_scores']),r['ours_dice'],-r['slice_index']))


def main(config_path):
    import h5py
    import torch
    config=json.loads(Path(config_path).read_text())
    sys.path.insert(0,config['code'])
    from runner_core import _build_model,zs_forward
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    out=Path(config['output']);out.mkdir(exist_ok=True,parents=True)
    with (Path(config['source_output'])/'candidate_slices.csv').open() as f:
        candidates=list(csv.DictReader(f))
    with h5py.File(Path(config['data'])/'MMWHS/whole_heart_test.h5','r') as h:
        images=np.ascontiguousarray(np.asarray(h['test_images'],dtype=np.float32).transpose(2,0,1)[:,None])
        targets=np.asarray(h['test_labels'],dtype=np.uint8).transpose(2,0,1)
        ends=np.asarray(h['patient_info_test'])
    model=_build_model('class');model.to('cuda:0').eval()
    methods=config['methods'];selected=[];arrays={};ranked_rows=[]
    def predict(method,stage,ids):
        checkpoint=Path(method['checkpoint']).with_name(f's{stage:02}.pt')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('model',state),strict=True);model.activate_stage(stage-1)
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(ids),4):
                x=torch.from_numpy(images[ids[start:start+4]]).cuda()
                output=zs_forward(model,x,None)['pred_masks']
                assert output.shape[1]==(4,6,8)[stage-1]
                predictions.append(output.argmax(1).cpu().numpy().astype(np.uint8))
        return np.concatenate(predictions)
    for stage,limit in enumerate((3,5,7),start=1):
        task=f'T{stage}';labels=list(range(1,limit+1))
        if stage==1 and config.get('preserve_t1_from'):
            previous=Path(config['preserve_t1_from'])
            report=json.loads((previous/'selection.json').read_text())
            row=next(r for r in report['selection'] if r['task']=='T1')
            assert report['config']['methods']==methods and report['config']['data']==config['data']
            with np.load(previous/'selected_slices.npz') as prior:
                arrays.update({k:prior[k] for k in prior.files if k.startswith('T1_')})
            selected.append(row)
            print('T1 preserved',row['slice_index'],flush=True)
            continue
        original=[r for r in candidates if r['task']==task]
        best_ours=max(float(r['ours_dice']) for r in original)
        qualified=[r for r in original if float(r['ours_dice'])>=config.get('ours_fraction',.90)*best_ours]
        ids=[int(r['slice_index']) for r in qualified]
        predictions=[]
        for method in methods[:-1]:
            predictions.append(predict(method,stage,ids))
            print(task,method['name'],'candidates',len(ids),flush=True)
        np.savez_compressed(out/f'candidate_predictions_{task}.npz',indices=np.array(ids),predictions=np.stack(predictions))
        rows=[]
        for i,r in enumerate(qualified):
            idx=ids[i]
            scores=[float(np.mean(dice(p[i],targets[idx],labels))) for p in predictions]
            pixels=[int(((p[i]>0)&(p[i]<=limit)).sum()) for p in predictions]
            pair_disagreement=mask_disagreement([p[i] for p in predictions],targets[idx],limit)
            row={'task':task,'slice_index':idx,'candidate_position':i,'ours_dice':float(r['ours_dice']),
                 'baseline_scores':scores,'baseline_foreground_pixels':pixels,
                 'baseline_pair_disagreement':pair_disagreement}
            rows.append(row)
            ranked_rows.append({'task':task,'slice_index':idx,'ours_dice':row['ours_dice'],
                'margin_over_strongest_baseline':row['ours_dice']-max(scores),
                'minimum_baseline_foreground_pixels':min(pixels),
                'baseline_min_disagreement':min(pair_disagreement),
                'baseline_mean_disagreement':float(np.mean(pair_disagreement)),
                **{m['name']:v for m,v in zip(methods[:-1],scores)}})
        chosen=select_row(rows,config.get('selection_mode','diverse'));i=chosen['candidate_position'];idx=chosen['slice_index']
        # Match the original full-test batch to avoid floating-point argmax ties.
        batch_start=(idx//4)*4
        own=predict(methods[-1],3,list(range(batch_start,min(batch_start+4,len(images)))))[idx-batch_start]
        print('verify',task,idx,chosen['ours_dice'],float(np.mean(dice(own,targets[idx],labels))),flush=True)
        assert abs(np.mean(dice(own,targets[idx],labels))-chosen['ours_dice'])<1e-5
        patient=int(np.searchsorted(ends,idx));start=0 if patient==0 else int(ends[patient-1])+1
        row={'task':task,'classes':labels,'slice_index':idx,'patient_index':patient,'slice_in_patient':idx-start,
             'ours_dice':chosen['ours_dice'],'scores':{},'per_class':{},'checkpoint_stages':{},'checkpoints':{},
             'eligible_ours_high_score_candidates':len(ids),'ours_score_floor':config.get('ours_fraction',.90)*best_ours,
             'margin_over_strongest_baseline':chosen['ours_dice']-max(chosen['baseline_scores']),
             'baseline_foreground_pixels':chosen['baseline_foreground_pixels']}
        row['baseline_pair_disagreement']=chosen['baseline_pair_disagreement']
        row['baseline_min_disagreement']=min(chosen['baseline_pair_disagreement'])
        row['baseline_mean_disagreement']=float(np.mean(chosen['baseline_pair_disagreement']))
        arrays[task+'_image']=images[idx,0];arrays[task+'_raw_gt']=targets[idx]
        arrays[task+'_gt']=cumulative_view(targets[idx],limit)
        for j,(method,p) in enumerate(zip(methods,[p[i] for p in predictions]+[own])):
            arrays[task+'_raw_pred_'+str(j)]=p;arrays[task+'_pred_'+str(j)]=cumulative_view(p,limit)
            scores=dice(p,targets[idx],labels);row['per_class'][method['name']]=scores
            row['scores'][method['name']]=float(np.mean(scores))
            checkpoint_stage=3 if j==len(methods)-1 else stage
            row['checkpoint_stages'][method['name']]=checkpoint_stage
            row['checkpoints'][method['name']]=str(Path(method['checkpoint']).with_name(f's{checkpoint_stage:02}.pt'))
        selected.append(row)
        print('SELECTED',task,idx,row['scores'],flush=True)
    np.savez_compressed(out/'selected_slices.npz',**arrays)
    with (out/'selection_candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(ranked_rows[0]));w.writeheader();w.writerows(ranked_rows)
    (out/'selection.json').write_text(json.dumps({'config':config,'selection':selected,
        'comparison_protocol':'baselines_at_task_end_ours_final',
        'selection_rule':config.get('selection_description','High-Ours candidates, visible baseline foreground, Ours margin >=0.10, then maximize minimum pairwise foreground disagreement.'),
        'pair_order':[[a['name'],b['name']] for i,a in enumerate(methods[:-1]) for b in methods[i+1:-1]],
        'ground_truth_source':'MMWHS/whole_heart_test.h5','test_slices':len(images),
        'index_convention':'zero-based; displayed cases are one-based'},indent=2))
    render(out)


if __name__=='__main__':
    if sys.argv[1]=='self-check':
        rows=[{'slice_index':1,'ours_dice':.9,'baseline_scores':[.8,.7],'baseline_foreground_pixels':[100,80]},
              {'slice_index':2,'ours_dice':.86,'baseline_scores':[.5,.4],'baseline_foreground_pixels':[90,70]},
              {'slice_index':3,'ours_dice':.99,'baseline_scores':[.01,.02],'baseline_foreground_pixels':[0,100]}]
        assert select_row(rows)['slice_index']==2
        gt=np.zeros((20,20),dtype=np.uint8);gt[5:15,5:15]=1
        other=gt.copy();other[5:15,5:15]=2
        assert mask_disagreement([gt,gt],gt,2)==[0.]
        assert mask_disagreement([gt,other],gt,2)==[1.]
        for row in rows[:2]:row.update(task='T2',baseline_pair_disagreement=[.1 if row['slice_index']==2 else .4])
        rows[0]['baseline_scores']=[.7,.6]
        assert select_row(rows[:2])['slice_index']==1
        stronger=[dict(rows[0],baseline_scores=[.4,.3,.2,.2,.2]),
                  dict(rows[1],baseline_scores=[.4,.3,.3,.3,.3])]
        assert select_row(stronger,'stronger_zs')['slice_index']==2
        print('self-check passed')
    else:main(sys.argv[1])
