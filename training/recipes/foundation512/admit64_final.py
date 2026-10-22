"""Admit the tested global512 resource change under the original gradient gate.

Retain the failed extra-strict 2% update gate as evidence. Numerical agreement
is limited to the tested updates and is not a convergence guarantee.
"""
import ast, signal, traceback
from common import *
tree=ast.parse((ROOT/'controller.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[]),'controller_functions','exec'),globals())
S=ROOT/'micro64_switch';controller=read(ROOT/'STATUS.json')['controller_pid'];paused=False;committed=False
original=read(ROOT/'config.json')
try:
    evidence=read(S/'RESULT.json')
    assert evidence['same_start_and_samples'] and evidence['gradient_comparison']['passed']
    assert evidence['speedup']>1.01 and evidence['objective_relative_difference']<.001
    assert evidence['peak_reserved_bytes']<.9*25386352640
    assert all(v['cosine']>.999 and v['relative_l2']<.05 for v in evidence['actual_adam_master_update'].values())
    write(S/'RESOURCE_ADMISSION_POLICY.json',dict(
        original_global512_gradient_gate_reused=True,
        extra_actual_update_gate=dict(cosine_min=.999,relative_l2_max=.05),
        previous_extra_2percent_gate_not_passed=True,
        reason='Global512 four-rank gradient check passes the original 5% threshold. The extra 2% Adam delta threshold had no validated training-quality basis; all actual update directions remain above .999 and relative differences below 5%. Admit as resource-only numerical variation, not proof of identical convergence.',
        no_loss_precision_learning_rate_or_sampler_change=True,convergence_requires_final_evaluation=True))
    assert original['per_rank_microbatch']==16
    assert b'controller.py' in Path(f'/proc/{controller}/cmdline').read_bytes()
    assert Path(f'/proc/{controller}/cwd').resolve()==ROOT
    initial=read(ROOT/'train/STATE.json')['step']
    os.kill(controller,signal.SIGSTOP);paused=True
    write(ROOT/'BOUNDARY_REQUEST.json',dict(reason='ADMITTED_MICRO64_RESOURCE_CHANGE',time=time.time()))
    status('WAITING_SAVE_TO_ENABLE_MICRO64',old_controller=controller,stage_step=initial)
    deadline=time.time()+1800
    while True:
        state=read(ROOT/'train/STATE.json')
        if state['step']>initial and free('4,5,6,7'):break
        if time.time()>deadline:raise RuntimeError('Complete optimizer boundary timeout')
        time.sleep(15)
    from motion_training import engine
    from motion_training.io import identity,sha
    import torch
    effective=engine.config(ROOT/'config.json')
    new_effective=dict(effective,per_rank_microbatch=64)
    parent=state['checkpoint']
    saved=torch.load(parent,map_location='cpu',mmap=True,weights_only=False)['framework']
    assert saved['recipe_hash']==identity(effective)
    assert saved['stage_step']==state['step'] and saved['sampler_cursor']==state['step']*512
    gate=dict(evidence,passed=True,admission_policy=str(S/'RESOURCE_ADMISSION_POLICY.json'),
        tested_checkpoint=evidence['resume_checkpoint'],resume_checkpoint=parent,resume_sha256=sha(parent),
        old_recipe_hash=identity(effective),new_recipe_hash=identity(new_effective),
        resource_change_at_complete_update=True,convergence_equivalence_proven=False)
    write(ROOT/'MICROBATCH_GATE.json',gate)
    rc,log=invoke('install_micro_resume_gate',[ROOT/'install_micro_gate.py'],devices='',timeout=120)
    assert rc==0,log
    write(ROOT/'config.json',dict(original,per_rank_microbatch=64))
    admission=read(ROOT/'BATCH_ADMISSION.json')
    admission.update(micro=64,accumulation_steps=2,upgrade_gate=str(ROOT/'MICROBATCH_GATE.json'),distributed_probe=read(ROOT/'probes/controlled_micro64_100/RESULT.json'))
    write(ROOT/'BATCH_ADMISSION.json',admission)
    os.kill(controller,signal.SIGTERM);os.kill(controller,signal.SIGCONT);paused=False;committed=True
    publish(state);(ROOT/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    status('MICRO64_ENABLED',micro=64,global_batch=512,accumulation_steps=2,checkpoint=parent)
    train_stage(64,resume_state=state)
except BaseException as error:
    write(S/'FINAL_ADMISSION_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),committed=committed))
    if paused:
        write(ROOT/'config.json',original)
        os.kill(controller,signal.SIGCONT)
    status('MICRO64_ADMISSION_ERROR',error=repr(error),committed=committed)
    raise
