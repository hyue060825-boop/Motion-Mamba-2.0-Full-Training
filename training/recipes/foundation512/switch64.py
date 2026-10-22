import ast, subprocess, signal, fcntl, traceback, math, statistics
from common import *
# Reuse tested launch, ownership, report, and continuation functions without
# executing the original controller's top-level startup.
tree=ast.parse((ROOT/'controller.py').read_text())
parts=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
exec(compile(ast.Module(body=parts,type_ignores=[]),'reused_controller_functions','exec'),globals())
S=ROOT/'micro64_switch';S.mkdir(exist_ok=True)
lock=(S/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old_controller=read(ROOT/'STATUS.json')['controller_pid'];committed=False;paused=False;old_cfg=read(ROOT/'config.json')
assert old_cfg['per_rank_microbatch']==16
def compare_updates(left,right):
    import torch
    torch.set_num_threads(2)
    a=torch.load(ROOT/'probes'/left/'master_update.pt',map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(ROOT/'probes'/right/'master_update.pt',map_location='cpu',mmap=True,weights_only=False)
    stats={}
    for group in ['all','root','body','condition']:
        aa=bb=dot=err=0.
        for n,x in a.items():
            kind='root' if n.startswith('root_model.') else 'body' if n.startswith('body_model.') else 'condition'
            if group!='all' and kind!=group:continue
            x=x.reshape(-1).double();y=b[n].reshape(-1).double()
            aa+=float(x@x);bb+=float(y@y);dot+=float(x@y);err+=float((x-y).square().sum())
        stats[group]=dict(cosine=dot/math.sqrt(max(aa*bb,1e-60)),relative_l2=math.sqrt(err/max(aa,1e-60)),reference_l2=math.sqrt(aa))
    return stats
try:
    assert b'controller.py' in Path(f'/proc/{old_controller}/cmdline').read_bytes()
    assert Path(f'/proc/{old_controller}/cwd').resolve()==ROOT
    initial=read(ROOT/'train/STATE.json')['step']
    write(S/'PROTOCOL.json',dict(parent_stage_at_request=initial,global_batch=512,
        question='same512samples: micro16 versus64 complete inherited Adam update and throughput',
        thresholds=dict(update_cosine_min=.999,update_relative_l2_max=.02,group_update_relative_l2_max=.03,loss_relative_max=.001,max_memory_reserved_fraction=.9,min_speedup=1.01),
        legacy_gradient5percent_threshold_is_not_alone_a_training_quality_failure=True,
        no_changed_loss_data_lr_precision=True,probe_weights_discarded=True))
    os.kill(old_controller,signal.SIGSTOP);paused=True
    write(ROOT/'BOUNDARY_REQUEST.json',dict(reason='USER_REQUEST_MICRO64_CHECK',time=time.time()))
    status('WAITING_COMPLETE_UPDATE_FOR_MICRO64',old_controller=old_controller,prior_stage_step=initial)
    deadline=time.time()+1800
    while True:
        state=read(ROOT/'train/STATE.json')
        if state['step']>initial and free('4,5,6,7'):break
        if time.time()>deadline:raise RuntimeError('did not reach requested owned checkpoint boundary')
        time.sleep(15)
    publish(state);parent=state['checkpoint'];(ROOT/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    from motion_training.io import sha,identity
    results={}
    for micro in [16,64]:
        label=f'controlled_micro{micro}_{state["step"]}'
        status('TESTING_MICRO64',current_micro=micro,parent=parent)
        rc,log=invoke(label,[ROOT/'probe.py','--mode','distributed','--micro',micro,'--label',label,'--parent',parent,'--save-update'],devices='4,5,6,7',ranks=4,timeout=1800)
        assert rc==0,(label,log)
        results[micro]=dict(label=label,result=read(ROOT/'probes'/label/'RESULT.json'))
        assert results[micro]['result']['passed']
    left=results[16]['label'];right=results[64]['label'];grad=compare(left,right);updates=compare_updates(left,right)
    ra=results[16]['result'];rb=results[64]['result']
    speedup=ra['steady_seconds']/rb['steady_seconds']
    restored={micro:read(ROOT/'probes'/results[micro]['label']/'RESTORED.json') for micro in results}
    lossrel=abs(ra['rows'][0]['loss']-rb['rows'][0]['loss'])/max(abs(ra['rows'][0]['loss']),1e-8)
    same=all(restored[16][k]==restored[64][k] for k in ['parent','model_hash','optimizer_hash','scheduler']) and ra['rows'][0]['receipt_hash']==rb['rows'][0]['receipt_hash']
    passed=same and lossrel<=.001 and updates['all']['cosine']>=.999 and updates['all']['relative_l2']<=.02 and all(v['relative_l2']<=.03 for k,v in updates.items() if v['reference_l2']>1e-12) and speedup>=1.01
    new_cfg=dict(old_cfg,per_rank_microbatch=64)
    gate=dict(passed=passed,kind='micro16_to64_same_global512',resume_checkpoint=parent,resume_sha256=sha(parent),
        old_recipe_hash=identity(old_cfg),new_recipe_hash=identity(new_cfg),same_start_and_samples=same,
        objective_relative_difference=lossrel,gradient_comparison=grad,actual_adam_master_update=updates,
        four_micro16_seconds=ra['steady_seconds'],four_micro64_seconds=rb['steady_seconds'],speedup=speedup,
        peak_reserved_bytes=max(v['reserved_peak'] for row in rb['rows'] for v in row['peaks']),
        global_batch=512,old_micro=16,new_micro=64,probe_weights_discarded=True,
        precision_unchanged=True,optimizer_schedule_sampler_inherited=True)
    write(S/'RESULT.json',gate)
    if not passed:
        status('MICRO64_CHECK_NOT_ADMITTED',result=str(S/'RESULT.json'),existing_micro16_will_resume=True)
        os.kill(old_controller,signal.SIGCONT);paused=False;exit()
    write(ROOT/'MICROBATCH_GATE.json',gate)
    rc,log=invoke('install_micro_resume_gate',[ROOT/'install_micro_gate.py'],devices='',timeout=120);assert rc==0,log
    write(ROOT/'config.json',new_cfg)
    admission=read(ROOT/'BATCH_ADMISSION.json');write(S/'OLD_BATCH_ADMISSION.json',admission)
    admission.update(micro=64,accumulation_steps=2,upgrade_gate=str(S/'RESULT.json'),distributed_probe=rb)
    write(ROOT/'BATCH_ADMISSION.json',admission)
    os.kill(old_controller,signal.SIGTERM);os.kill(old_controller,signal.SIGCONT);paused=False;committed=True
    (ROOT/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    status('MICRO64_ADMITTED',micro=64,global_batch=512,ranks=4,accumulation_steps=2,speedup=speedup)
    train_stage(64,resume_state=state)
except BaseException as error:
    write(S/'ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),committed=committed))
    if paused:
        write(ROOT/'config.json',old_cfg)
        try:os.kill(old_controller,signal.SIGCONT)
        except ProcessLookupError:pass
    status('MICRO64_SWITCH_ERROR',error=repr(error),prior_training_resumed=paused and not committed,committed=committed)
    raise
