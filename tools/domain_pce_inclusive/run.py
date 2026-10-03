"""Reevaluate saved Domain checkpoints on the unchanged test split.

Run as `python run.py` from a directory containing a private plan.json with
code, data_root, run_root, reference_root, seed and batch_size. Only scalar
scores are written. Original foreground scores must reproduce within 1e-5.
"""
import json
import random
import sys
import time
from pathlib import Path


def save(result):
    temporary = Path('results.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temporary.replace('results.json')


def main():
    plan = json.loads(Path('plan.json').read_text())
    sys.path.insert(0, plan['code'])
    import numpy as np
    import torch
    import runner_core as core

    torch.set_num_threads(4)
    seed = plan['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cuda:0')
    tasks = core.TASKS['domain']
    run = Path(plan['run_root'])
    reference = Path(plan['reference_root'])
    original = json.loads((run / 'metrics.json').read_text())
    assert len(tasks) == 6 and len(original['performance_matrix']) == 6
    assert json.loads((reference / 'independent_scores.json').read_text())['scores'] == original['independent_scores']
    result = dict(status='running', source_run=run.name, reference_run=reference.name,
                  seed=seed, split='test', classes=[0, 1], dice_includes_background=True,
                  aggregation='equal patient mean, equal class mean, equal task mean',
                  checkpoint_selection='preserved original checkpoints; no reselection',
                  started_at=time.time(), evaluations=[], matrices={}, metrics={})
    save(result)

    def evaluate(model, selected, expected, label):
        rows = []
        for index in selected:
            task = tasks[index]
            dataset = core.H5Slices(Path(plan['data_root']) / task.folder / task.filename, 'test')
            try:
                assert len(dataset.ends) and dataset.ends[-1] + 1 == len(dataset)
                assert np.all(np.diff(dataset.ends) > 0)
                score = core.evaluate(model, core._loader(dataset, plan['batch_size'], False, 0, seed),
                                      dataset.ends, device, None, (0, 1))
                bg, fg = score['per_class']
                inclusive = (bg + fg) / 2
                assert abs(score['benchmark_mean'] - inclusive) < 1e-12
                error = abs(fg - expected[index])
                if error > 1e-5:
                    raise ValueError(f'{label}/{task.code}: foreground mismatch {error}')
                assert np.isfinite([bg, fg]).all() and 0 <= inclusive <= 1
                rows.append(dict(task=task.code, foreground=fg, background=bg,
                                 inclusive=inclusive, foreground_abs_error=error,
                                 patients=len(dataset.ends), slices=len(dataset)))
            finally:
                dataset.close()
        result['evaluations'].append(dict(source=label, tasks=rows))
        save(result)
        print(json.dumps(dict(completed=label, evaluations=len(result['evaluations']),
                              elapsed_seconds=time.time()-result['started_at'])), flush=True)
        return rows

    model = core._build_model('domain').to(device)
    random_rows = evaluate(model, range(6), original['random_scores'], 'random_seed42')

    def load(path, stage):
        before = path.stat()
        state = torch.load(path, map_location='cpu', weights_only=False)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        model.load_state_dict(state.get('model', state), strict=True)
        model.activate_stage(stage)

    matrix_rows = []
    for stage in range(6):
        load(run / f's{stage+1:02d}.pt', stage)
        matrix_rows.append(evaluate(model, range(6), original['performance_matrix'][stage],
                                    f'{run.name}/s{stage+1:02d}.pt'))
    independent_rows = []
    for index in range(6):
        load(reference / f's{index+1:02d}.pt', index)
        independent_rows.extend(evaluate(model, [index], original['independent_scores'],
                                         f'{reference.name}/s{index+1:02d}.pt'))
    for metric in ['foreground', 'inclusive']:
        matrix = [[row[metric] for row in stage] for stage in matrix_rows]
        result['matrices'][metric] = matrix
        result['metrics'][metric] = core.domain_matrix_metrics(
            np.asarray(matrix), [row[metric] for row in random_rows],
            [row[metric] for row in independent_rows])
    for key, value in original['metrics'].items():
        assert abs(result['metrics']['foreground'][key] - value) < 1e-4
    result.update(status='complete', finished_at=time.time(),
                  foreground_parity_passed=True, evaluated_task_count=48,
                  gpu_peak_mib=torch.cuda.max_memory_reserved()/1024**2)
    assert sum(len(item['tasks']) for item in result['evaluations']) == 48
    save(result)
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if Path('results.json').exists():
            result = json.loads(Path('results.json').read_text())
            result.update(status='failed', finished_at=time.time())
            save(result)
        raise
