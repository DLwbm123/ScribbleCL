"""Small checks for dense targets and the fixed comparison budget."""
import tempfile
import json
import math
import os
import sys
from pathlib import Path
import h5py
import numpy as np
from runner_core import H5Slices, IGNORE_INDEX, organ_report_schedule
from run_comparisons import METHODS, job_arguments


def main():
    for method in METHODS:
        schedules = [organ_report_schedule(task, method.startswith('zs-')) for task in ('T1','T2','T3','T4')]
        assert [s['epochs'] for s in schedules] == [60,60,10,10]
        assert [s['lr'] for s in schedules] == [.03,.03,.06,.06]
        assert [s['spatial_weight'] for s in schedules] == ([.01,.01,0,0] if method.startswith('zs-') else [0]*4)
        assert schedules[0]['spatial_warmup'] == 28  # epoch index 29 is epoch 30
        args = job_arguments(Path('/data_nas/check'), method)
        assert '--organ-report-schedule' in args and '--setting-run' in args
        assert not any(flag in args for flag in ('--t2-from','--t3-from','--t4-from'))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        labels = np.zeros((256,256,2),dtype=np.int16); labels[50:100,50:100] = 1
        with h5py.File(root/'data.h5','w') as stream:
            stream['train_images'] = labels.astype('float32')
            stream['train_labels'] = labels
        sparse = np.full((2,256,256),IGNORE_INDEX,dtype=np.int16)
        sparse[:,60,60] = 1
        np.savez(root/'sparse.npz',annotations=sparse)
        for cached in (False,True):
            H5Slices.cache_arrays = cached
            dense = H5Slices(root/'data.h5','train',dense_supervision=True)
            weak = H5Slices(root/'data.h5','train',root/'sparse.npz')
            assert np.array_equal(dense[0][1].numpy(),labels[:,:,0])
            assert (weak[0][1].numpy()==IGNORE_INDEX).sum()==256*256-1
            dense.close(); weak.close()
        H5Slices.cache_arrays = False
    print('PASS: full dense targets, unchanged sparse targets, cache parity, 60/60/10/10 schedule and fresh jobs')


def smoke():
    import torch
    import runner_core as runner
    torch.set_num_threads(4)
    root = Path(os.environ['CHECK_ROOT'])
    method = os.environ['CHECK_METHOD']
    data = root/'inputs/data/Task_incre'; data.mkdir(parents=True,exist_ok=True)
    sparse_root = root/'inputs/sparse/organ'; sparse_root.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(42)
    labels = np.zeros((256,256,2),dtype=np.int16); labels[80:160,80:160] = 1
    for task in runner.TASKS['organ']:
        with h5py.File(data/task.filename,'w') as stream:
            for split in ('train','val','test'):
                stream[f'{split}_images'] = rng.normal(size=labels.shape).astype('float32')
                stream[f'{split}_labels'] = labels
                if split != 'train': stream[f'patient_info_{split}'] = [1]
        sparse = np.full((2,256,256),IGNORE_INDEX,dtype=np.int16)
        sparse[:,100:103,100:150] = 1; sparse[:,20:23,20:100] = 0
        np.savez(sparse_root/f'{task.code}_v2_s2_seed42.npz',annotations=sparse)
    args = job_arguments(root,method)
    for flag,value in {'--workers':'0','--batch-size':'2','--gpm-examples':'2',
                       '--gpm-max-patches-per-layer':'32','--gpm-max-matrix-elements':'32768',
                       '--fisher-batches':'1'}.items():
        args[args.index(flag)+1] = value
    args.remove('--setting-run')
    schedule = runner.organ_report_schedule
    def short_schedule(task,zs):
        return dict(schedule(task,zs),epochs=1,spatial_warmup=-1)
    runner.organ_report_schedule = short_schedule
    sys.argv = ['check',*args]
    runner.main('organ')
    output = root/'runs'/method
    result = json.loads((output/'summary.json').read_text())
    rows = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    trained = [row for row in rows if 'loss' in row]
    assert result['completed_stages']==4 and len(trained)==4
    assert all(math.isfinite(row['loss']) for row in trained)
    assert all(row['task_strategy'].get('freeze_backbone_bn') for row in result['stage_rows'][2:])
    if method == 'zs-gpm':
        assert all(row['gpm_gradient_ratio'] is not None for row in trained[1:])
    if method == 'zs-ewc':
        assert len(json.loads((output/'fisher.json').read_text()))==4
    (root/'passed.json').write_text(json.dumps(dict(method=method,completed_stages=4,status='PASS'))+'\n')
    print('PASS: synthetic four-stage GPU integration',method)


if __name__ == '__main__':
    smoke() if os.environ.get('CHECK_METHOD') else main()
