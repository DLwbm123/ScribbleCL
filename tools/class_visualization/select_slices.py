"""Requested contrastive selection: acquisition baselines versus final Ours."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from visualize import dice, cumulative_view, render


def select_row(rows):
    feasible=[r for r in rows if min(r['baseline_foreground_pixels'])>=32 and min(r['baseline_scores'])>=.01]
    if not feasible:
        raise ValueError('No slice satisfies visible foreground and nonzero overlap for all baselines')
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
        original=[r for r in candidates if r['task']==task]
        best_ours=max(float(r['ours_dice']) for r in original)
        qualified=[r for r in original if float(r['ours_dice'])>=.90*best_ours]
        ids=[int(r['slice_index']) for r in qualified]
        predictions=[]
        for method in methods[:-1]:
            predictions.append(predict(method,stage,ids))
            print(task,method['name'],'candidates',len(ids),flush=True)
        rows=[]
        for i,r in enumerate(qualified):
            idx=ids[i]
            scores=[float(np.mean(dice(p[i],targets[idx],labels))) for p in predictions]
            pixels=[int(((p[i]>0)&(p[i]<=limit)).sum()) for p in predictions]
            row={'task':task,'slice_index':idx,'candidate_position':i,'ours_dice':float(r['ours_dice']),
                 'baseline_scores':scores,'baseline_foreground_pixels':pixels}
            rows.append(row)
            ranked_rows.append({'task':task,'slice_index':idx,'ours_dice':row['ours_dice'],
                'margin_over_strongest_baseline':row['ours_dice']-max(scores),
                'minimum_baseline_foreground_pixels':min(pixels),
                **{m['name']:v for m,v in zip(methods[:-1],scores)}})
        chosen=select_row(rows);i=chosen['candidate_position'];idx=chosen['slice_index']
        own=predict(methods[-1],3,[idx])[0]
        assert abs(np.mean(dice(own,targets[idx],labels))-chosen['ours_dice'])<1e-5
        patient=int(np.searchsorted(ends,idx));start=0 if patient==0 else int(ends[patient-1])+1
        row={'task':task,'classes':labels,'slice_index':idx,'patient_index':patient,'slice_in_patient':idx-start,
             'ours_dice':chosen['ours_dice'],'scores':{},'per_class':{},'checkpoint_stages':{},'checkpoints':{},
             'eligible_ours_high_score_candidates':len(ids),'ours_score_floor':.90*best_ours,
             'margin_over_strongest_baseline':chosen['ours_dice']-max(chosen['baseline_scores']),
             'baseline_foreground_pixels':chosen['baseline_foreground_pixels']}
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
        'selection_rule':'Keep Ours >= 90% of its best eligible slice Dice per cumulative view; require each baseline >=32 foreground pixels and Dice >=0.01; maximize Ours minus the strongest baseline. Post-hoc contrastive examples, not average performance.',
        'ground_truth_source':'MMWHS/whole_heart_test.h5','test_slices':len(images),
        'index_convention':'zero-based; displayed cases are one-based'},indent=2))
    render(out)


if __name__=='__main__':
    if sys.argv[1]=='self-check':
        rows=[{'slice_index':1,'ours_dice':.9,'baseline_scores':[.8,.7],'baseline_foreground_pixels':[100,80]},
              {'slice_index':2,'ours_dice':.86,'baseline_scores':[.5,.4],'baseline_foreground_pixels':[90,70]},
              {'slice_index':3,'ours_dice':.99,'baseline_scores':[.01,.02],'baseline_foreground_pixels':[0,100]}]
        assert select_row(rows)['slice_index']==2
        print('self-check passed')
    else:main(sys.argv[1])
