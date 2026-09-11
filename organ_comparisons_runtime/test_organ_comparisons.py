"""Small checks for dense targets and the fixed comparison budget."""
import tempfile
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


if __name__ == '__main__':
    main()
