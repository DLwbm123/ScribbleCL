"""One CPU self-check of bounded controls, isolation and observational gradients."""
import copy
import inspect
import random
from types import SimpleNamespace
import numpy as np
import torch
import runner_core as runner
from cl_methods import stable_backbone_features, freeze_batchnorm_stats, DarkExperienceReplayPlus
from organ_t3_retention_probe import (RetentionProbe, observation, gradient_probe,
    check_paired_restore, validate_controls)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(44)
    model = runner.OrganModel(); model.activate_stage(1)
    saved_model = copy.deepcopy(model.state_dict())
    replay = DarkExperienceReplayPlus(128, 4, .05, .5)
    state = replay.state_dict()
    state.update(examples=torch.randn(128,1,2,2), feature_targets=torch.randn(128,2,2,2),
        sparse_labels=torch.zeros(128,2,2,dtype=torch.int16), task_ids=torch.tensor([0]*76+[1]*52),
        class_counts=torch.full((128,),2), num_seen_examples=8801)
    replay.load_state_dict(state)
    check_paired_restore(model,replay,dict(model=saved_model,continual=state))
    bad = copy.deepcopy(state);bad['alpha']=.5
    try: replay.load_state_dict(bad)
    except ValueError: pass
    else: raise AssertionError('strict policy restore bypassed')
    model.activate_stage(2); model.freeze_backbone_bn=True; model.train()
    old = copy.deepcopy(model.state_dict())
    obs = RetentionProbe.__new__(RetentionProbe)
    obs.model=model;obs.backbone_lr=.003
    opt = torch.optim.SGD(obs.parameter_groups(), momentum=.9,weight_decay=1e-4)
    x=torch.randn(2,1,32,32,requires_grad=True)
    # Saliency, clean/global, replay and capture all route through this persistent policy.
    out=model.forward_logits(x,2)
    torch.autograd.grad(out.sum(),x,retain_graph=True)
    with freeze_batchnorm_stats(model.backbone): model.forward_logits(x,1)
    stable_backbone_features(model,x,no_grad=True)
    model.eval();model.train()
    loss=out.square().mean()
    probe = gradient_probe(model,dict(current=loss, feature=loss*.05, T2_replay=loss*.1))
    assert probe['current_T2_replay_cosine'] > .9999
    assert all(p.grad is None for p in model.parameters())
    loss.backward(); opt.step()
    now=model.state_dict()
    for k in old:
        if k.startswith(('heads.0.','heads.1.')) or (k.startswith('backbone.') and k.endswith(('running_mean','running_var','num_batches_tracked'))):
            assert torch.equal(old[k],now[k]),k
    python_state=random.getstate(); numpy_state=np.random.get_state(); torch_state=torch.get_rng_state()
    modes=[m.training for m in model.modules()]
    with observation(model):
        random.random();np.random.rand();torch.rand(3); model.eval(); model.forward_logits(x,0)
    assert python_state==random.getstate()
    assert all(np.array_equal(a,b) for a,b in zip(numpy_state,np.random.get_state()))
    assert torch.equal(torch_state,torch.get_rng_state())
    assert modes==[m.training for m in model.modules()]
    src=inspect.getsource(runner.main)
    scheduler='group["lr"] = group.get("base_lr", args.lr) * max(0.0, 1.0 - iteration / max_iterations) ** 0.9'
    assert scheduler in src
    for iteration in (1,10,178):
        for group in opt.param_groups:
            exec(scheduler,{},dict(group=group,args=SimpleNamespace(lr=.03),iteration=iteration,max_iterations=10680))
        assert abs(opt.param_groups[0]['lr']/opt.param_groups[1]['lr']-.1)<1e-12
        assert opt.param_groups[1]['lr']>.029
    # Verify invalid truncation/test requests rejected before loading any data.
    args=SimpleNamespace(method='zs-derpp',batch_size=4,epochs_per_task=60,max_task=3,seed=42,
        der_buffer_size=128,der_minibatch_size=4,organ_t2_feature_alpha=.05,pce_loss_weight=1.,
        zs_global_weight=1.,zs_spatial_loss_weight=.01,zs_spatial_warmup_epochs=28,lr=.03,
        t3_one_epoch_from='source',organ_t2_supervision_strategy=True,numerical_debug=True,
        test_evaluation=False,max_train_batches=None,organ_task=None,t2_from=None,
        zs_clean_bn_writer=False,zs_gd_loss=False,zs_adversarial_perturbation=False)
    validate_controls(args,'organ')
    for key,value in (('test_evaluation',True),('epochs_per_task',1),('max_train_batches',178)):
        altered=copy.copy(args);setattr(altered,key,value)
        try: validate_controls(altered,'organ')
        except ValueError: pass
        else: raise AssertionError(key)
    assert 'raise SystemExit(0)' in inspect.getsource(RetentionProbe.after_step)
    assert src.index('probe.after_step(iteration, optimizer, numerics, gradient_norm)')<src.index('final_validation = validate_current_task()')
    print('PASS: paired restore, fixed BN/all old heads, nonperturbing eval/gradient probe, group LR/horizon, test/budget guards and terminal exit')


if __name__=='__main__': main()
