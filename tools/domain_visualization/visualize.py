"""Post-hoc best-slice visualization; no training or checkpoint selection."""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def dice(pred, target):
    return (2 * (pred & target).sum() + 1e-5) / (pred.sum() + target.sum() + 1e-5)


def infer(config):
    import torch
    sys.path.insert(0, config['code'])
    from runner_core import H5Slices, TASKS, _build_model, _loader, zs_forward
    torch.set_num_threads(4)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    model = _build_model('domain')
    for stage in range(6):
        model.activate_stage(stage)
    model.to('cuda:0').eval()

    def load(path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model', state))

    load(config['methods'][-1]['checkpoint'])
    selected, candidates, arrays, aggregate = [], [], {}, {}
    with torch.no_grad():
        for task in TASKS['domain']:
            ds = H5Slices(Path(config['data']) / task.folder / task.filename, 'test')
            predictions, targets = [], []
            for image, target in _loader(ds, 4, False, 0, 0):
                predictions.append(zs_forward(model, image.cuda(), None)['pred_masks'].argmax(1).cpu().numpy() == 1)
                targets.append(target.numpy() == 1)
            pred, gt = np.concatenate(predictions), np.concatenate(targets)
            starts = [0] + [int(x) + 1 for x in ds.ends[:-1]]
            aggregate[task.code] = float(np.mean([dice(pred[s:int(e)+1], gt[s:int(e)+1]) for s, e in zip(starts, ds.ends)]))
            ranked = []
            for i in range(len(ds)):
                if not gt[i].any():
                    continue
                patient = int(np.searchsorted(ds.ends, i))
                row = dict(domain=task.code, patient_index=patient, slice_index=i,
                           slice_in_patient=i-starts[patient], foreground_pixels=int(gt[i].sum()),
                           ours_dice=float(dice(pred[i], gt[i])))
                candidates.append(row)
                ranked.append(row)
            # Exact best foreground Dice; larger target then lower index break ties.
            best = max(ranked, key=lambda r: (r['ours_dice'], r['foreground_pixels'], -r['slice_index']))
            i = best['slice_index']
            image, _ = ds[i]
            arrays[task.code + '_image'] = image.numpy()[0]
            arrays[task.code + '_gt'] = gt[i]
            arrays[task.code + '_pred_4'] = pred[i]
            best['scores'] = {config['methods'][-1]['name']: best['ours_dice']}
            selected.append(best)
            ds.close()
            print(task.code, 'best', i, best['ours_dice'], 'patient mean', aggregate[task.code], flush=True)
        for j, method in enumerate(config['methods'][:-1]):
            load(method['checkpoint'])
            for row in selected:
                code = row['domain']
                x = torch.from_numpy(arrays[code + '_image'][None, None]).cuda()
                pred = zs_forward(model, x, None)['pred_masks'].argmax(1)[0].cpu().numpy() == 1
                arrays[code + '_pred_' + str(j)] = pred
                row['scores'][method['name']] = float(dice(pred, arrays[code + '_gt']))
            print(method['name'], 'done', flush=True)
    expected = config['expected_ours_foreground']
    assert all(abs(aggregate[k] - expected[k]) < 1e-5 for k in expected), (aggregate, expected)
    with (out / 'candidate_slices.csv').open('w') as f:
        fields = ['domain', 'patient_index', 'slice_index', 'slice_in_patient', 'foreground_pixels', 'ours_dice']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(candidates)
    np.savez_compressed(out / 'selected_slices.npz', **arrays)
    (out / 'selection.json').write_text(json.dumps(dict(config=config, selection=selected,
        verified_ours_foreground_patient_mean=aggregate, selection_rule='maximum foreground slice Dice among all nonempty test slices, independently per domain; post-hoc best cases',
        index_convention='all indices zero-based'), indent=2))


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = Path(directory)
    data = np.load(out / 'selected_slices.npz')
    report = json.loads((out / 'selection.json').read_text())
    methods = report['config']['methods']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    names = ['Image', 'Ground truth'] + [m['name'].replace('-Sequential', '\nSequential') for m in methods]
    names[-1] = 'ScribbleCL (ours)\nZS-DER++'

    def panel(rows, zoom, filename):
        fig, axes = plt.subplots(len(rows), 7, figsize=(17.5, 2.5 * len(rows)), squeeze=False)
        for r, row in enumerate(rows):
            code = row['domain']; im = data[code + '_image']; gt = data[code + '_gt']
            lo, hi = np.percentile(im, [1, 99])
            for c, ax in enumerate(axes[r]):
                ax.imshow(im, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
                if c >= 1:
                    ax.contour(gt, levels=[.5], colors=['#35e77d'], linewidths=1.15)
                if c >= 2:
                    pred = data[code + '_pred_' + str(c-2)]
                    if pred.any() and not pred.all():
                        ax.contour(pred, levels=[.5], colors=['#ff5d78'], linewidths=1.15)
                    score = row['scores'][methods[c-2]['name']]
                    ax.text(.5, .025, f'Dice {score:.3f}', transform=ax.transAxes, ha='center', va='bottom', color='white', fontsize=10,
                            bbox=dict(facecolor='black', edgecolor='none', alpha=.75, pad=2))
                if zoom:
                    y, x = np.where(gt)
                    size = max(int(max(x.max()-x.min(), y.max()-y.min()) * 1.7), 72)
                    cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
                    ax.set_xlim(max(0,cx-size/2), min(im.shape[1]-1,cx+size/2))
                    ax.set_ylim(min(im.shape[0]-1,cy+size/2), max(0,cy-size/2))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                if r == 0: ax.set_title(names[c], fontsize=11, weight='bold', pad=10)
            axes[r,0].set_ylabel(f'Domain {code}\nCase {row["patient_index"]+1}, slice {row["slice_in_patient"]}\nTest index {row["slice_index"]}', fontsize=10)
        fig.legend([Line2D([0],[0],color='#35e77d',lw=2),Line2D([0],[0],color='#ff5d78',lw=2)],
                   ['Ground truth', 'Prediction'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5,.015))
        if len(rows) > 1:
            fig.suptitle('Domain-CL | Best foreground-Dice slice per domain', fontsize=16, weight='bold', y=.995)
        fig.text(.5,.009,'Post-hoc best-case examples on test data • All models after A–F training • Same slice and display scale across methods',ha='center',fontsize=9,color='#555555')
        fig.subplots_adjust(left=.067,right=.995,top=.92 if len(rows)>1 else .78,bottom=.07 if len(rows)>1 else .2,wspace=.035,hspace=.08)
        fig.savefig(out / (filename+'.png'), dpi=190, facecolor='white')
        fig.savefig(out / (filename+'.pdf'), facecolor='white')
        plt.close(fig)
    panel(report['selection'], False, 'domain_best_full')
    panel(report['selection'], True, 'domain_best_zoom')
    for row in report['selection']:
        panel([row], True, 'domain_' + row['domain'])


if __name__ == '__main__':
    if sys.argv[1] == 'self-check':
        a = np.array([[True, False], [False, False]])
        assert dice(a,a) == 1 and dice(a,~a) < 1e-4
        print('self-check passed')
    elif sys.argv[1] == 'infer':
        infer(json.loads(Path(sys.argv[2]).read_text()))
    elif sys.argv[1] == 'render':
        render(sys.argv[2])
