"""Audit whether final baseline models have foreground in cumulative views."""
import csv
import json
import sys
from pathlib import Path
import numpy as np


def class_counts(pred):
    return np.stack([(pred == c).sum(axis=(1, 2)) for c in range(8)], axis=1)


def main(config_path):
    import h5py
    import torch
    config=json.loads(Path(config_path).read_text())
    sys.path.insert(0,config['code'])
    from runner_core import _build_model,zs_forward
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    out=Path(config['output'])
    with h5py.File(Path(config['data'])/'MMWHS/whole_heart_test.h5','r') as f:
        images=np.ascontiguousarray(np.asarray(f['test_images'],dtype=np.float32).transpose(2,0,1)[:,None])
    with (out/'candidate_slices.csv').open() as f:candidates=list(csv.DictReader(f))
    model=_build_model('class');model.activate_stage(2);model.to('cuda:0').eval()
    records=[];all_counts=[];summary={}
    for method in config['methods'][:-1]:
        state=torch.load(method['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('model',state),strict=True)
        counts=[]
        with torch.no_grad():
            for start in range(0,len(images),4):
                x=torch.from_numpy(images[start:start+4]).cuda()
                pred=zs_forward(model,x,None)['pred_masks'].argmax(1).cpu().numpy()
                counts.append(class_counts(pred))
        counts=np.concatenate(counts)
        assert np.all(counts.sum(1)==images.shape[-1]*images.shape[-2])
        all_counts.append(counts)
        item={'checkpoint':method['checkpoint'],
              'total_pixels_per_class':counts.sum(0).tolist(),
              'slices_with_class':(counts>0).sum(0).tolist(),
              'slices_with_any_C1_C3':int((counts[:,1:4].sum(1)>0).sum()),
              'slices_with_any_C1_C5':int((counts[:,1:6].sum(1)>0).sum()),
              'slices_with_any_C1_C7':int((counts[:,1:8].sum(1)>0).sum())}
        summary[method['name']]=item
        for i,row in enumerate(counts):
            records.append({'method':method['name'],'slice_index':i,**{f'C{c}_pixels':int(row[c]) for c in range(8)}})
        print(method['name'],json.dumps(item),flush=True)
    all_counts=np.stack(all_counts)
    feasible={}
    for task,limit in [('T1',3),('T2',5),('T3',7)]:
        task_rows=[r for r in candidates if r['task']==task]
        valid=[r for r in task_rows if np.all(all_counts[:,int(r['slice_index']),1:limit+1].sum(1)>=32)]
        feasible[task]={'gt_eligible_candidates':len(task_rows),'all_five_baselines_have_at_least_32_foreground_pixels':len(valid),
                        'best_ours_candidate':max(valid,key=lambda r:float(r['ours_dice'])) if valid else None}
    result={'test_slices':len(images),'class_ids':list(range(8)),'methods':summary,
            'selection_criterion':'Every baseline has at least 32 predicted pixels in the cumulative class range; among feasible original candidates, maximize existing Ours Dice.',
            'feasible_candidates':feasible}
    (out/'foreground_audit.json').write_text(json.dumps(result,indent=2))
    with (out/'foreground_per_slice.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('FEASIBLE',json.dumps(feasible),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='self-check':
        x=np.array([[[0,1],[5,7]],[[0,0],[6,6]]])
        c=class_counts(x)
        assert c.tolist()==[[1,1,0,0,0,1,0,1],[2,0,0,0,0,0,2,0]]
        assert (c[:,1:4].sum(1)>0).tolist()==[True,False]
        assert (c[:,1:6].sum(1)>0).tolist()==[True,False]
        assert np.all(c.sum(1)==4)
        print('self-check passed')
    else:main(sys.argv[1])
