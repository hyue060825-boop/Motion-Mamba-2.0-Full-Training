"""Diagnostic only: final generic Walk versus explicit Walk forward, fixed noise/goals."""
from pathlib import Path
import os,time,fcntl,traceback,csv
import numpy as np
import audit_process_122k as ap
from execute import R,rd,wr,setup,backend,sha
from controller import free,acquire
O=ap.O
def main():
 lock=(O/'command_clarity.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 selected=rd(O/'SELECTION.json')['rows'];assert rd(O/'REPLAY_COMPLETE.json')['passed']
 cases=[__import__('json').loads(s) for s in (R/'CASES.jsonl').open() if __import__('json').loads(s)['split']=='locked_test']
 assets=dict(np.load(R/'INPUT_ASSETS.npz'));features=dict(zip(rd(R/'TEXT_INDEX.json')['prompts'],np.load(R/'TEXT_FEATURES.npy')))
 model,m,client=setup('native122',O/'clarity_load');backend('native122',m,client,True)
 original=ap.schedule
 def modified(task):
  names,texts=original(task)
  # Only replace the final six "Walk." windows. No timing or earlier command changes.
  assert task=='nominal40' and names[-6:]==['walk']*6
  names=names[:-6]+['forward']*6;texts=texts[:-6]+[rd(R/'PLAN.json')['prompts']['forward']]*6
  return names,texts
 ap.schedule=modified;rows=[]
 for item in selected:
  i=item['case'];si=item['seed_index'];case=cases[i];seed=case['noise_seeds'][si]
  geo,trace,_=ap.independent('native122',model,m,client,case,'nominal40',seed,assets,features)
  before=rd(O/f'{i:03d}_s{si}'/'ACTUAL_CONDITIONING_AND_WINDOW_TRACE.json')
  assert trace['initial_noise_hashes']==before['initial_noise_hashes'] and trace['final_RNG']==before['final_RNG']
  end=float(np.linalg.norm(geo['joints'][-1,case['selected_joint_index']].astype(float)-case['end_goal_xyz_m'])*100)
  rows.append(dict(role=item['role'],case=i,seed_index=si,original_generic_walk_root_CE_cm=before['final_root_CE_cm'],diagnostic_explicit_forward_root_CE_cm=trace['final_root_CE_cm'],diagnostic_end_CE_cm=end,same_all19_initial_noises=True,same_final_RNG=True,only_last_six_text_windows_changed=True))
  wr(O/f'{i:03d}_s{si}'/'DIAGNOSTIC_EXPLICIT_FORWARD_TRACE.json',trace)
  wr(O/'COMMAND_CLARITY_PROGRESS.json',dict(rows=rows,time=time.time()));print(rows[-1],flush=True)
 wr(O/'COMMAND_CLARITY_DIAGNOSTIC.json',dict(passed=True,rows=rows,scope='three preselected diagnostic cases only; not a new formal benchmark or score correction',formal_protocol_unchanged=True,all_formal_scores_preserved=True,new_training=False,source_sha256=sha(__file__),time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:wr(O/'COMMAND_CLARITY_FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
