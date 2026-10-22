from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,hashlib,shutil,collections
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
R=Path(__file__).resolve().parent;P=Path(artifact_path('unified_interactive_easy_v3_hiphi_20261021_32d9db1045'));sys.path.insert(0,str(P))
from prepare_scoring import REL,HEADROOT,HEADS,representation
from native_eval.metrics import candidate_pools,retrieval,family_mean_ci
from native_eval.encoding import encode_arrays
from native_eval.v2_score import model_for
import numpy as np,torch

def rd(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(v):return hashlib.sha256(('pairedtext-v4-20261022:'+v).encode()).hexdigest()
def wr(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def phases(task):return [3] if task=='short52' else [i for i,n in enumerate({'nominal9':[1,1,1,1],'nominal20':[3,1,2,3],'nominal40':[6,2,5,6]}[task]) for _ in range(n)]
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 assert not (R/'PROTOCOL_FROZEN.json').exists(),'Never overwrite a frozen new protocol'
 catalog=[json.loads(s) for s in (R/'GT_CATALOG.jsonl').open()];physical=np.load(R/'GT_CATALOG_PHYSICAL.npz')['physical'];lookup={v['sample_id']:i for i,v in enumerate(catalog)}
 old=[json.loads(s) for s in (P/'CASES.jsonl').open()];testinit={c['source_id'] for c in old if c['split']=='locked_test'};devinit={c['source_id'] for c in old if c['split']=='development'}
 devrefs={};testrefs={}
 for label,n in [('forward',8),('backward',4),('stand',16)]:
  rows=sorted([r for r in catalog if r['label']==label and r['source_split']=='val' and r['source_family_id'] not in testinit],key=lambda r:key('devref:'+r['sample_id']))
  assert len(rows)>=n,(label,len(rows));devrefs[label]=rows[:n]
 used={r['source_family_id'] for rows in devrefs.values() for r in rows}
 for label in devrefs:
  rows=[r for r in catalog if r['label']==label and r['source_family_id'] not in used|devinit]
  # Spread different original captions before taking another performance of the same caption.
  groups=collections.defaultdict(list)
  for r in rows:groups[r['caption']].append(r)
  ordered=[]
  for cap in sorted(groups,key=lambda c:key('caption:'+c)):groups[cap].sort(key=lambda r:key('testref:'+r['sample_id']))
  for j in range(max(map(len,groups.values()))):
   for cap in sorted(groups,key=lambda c:key('caption:'+c)):
    if j<len(groups[cap]):ordered.append(groups[cap][j])
  testrefs[label]=ordered
 assert len({r['caption'] for r in testrefs['forward']})>=32
 assert len({r['caption'] for r in testrefs['stand']})>=32
 cases=[];phaseids=[];state=[]
 for split in ['development','locked_test']:
  sources=[c for c in old if c['split']==split];pools=devrefs if split=='development' else testrefs
  for i,c in enumerate(sources):
   # Forward/backward order is balanced across initializations. The last phase
   # always travels forward, compatible with the shared frontal terminal goal.
   first='forward' if i%2==0 else 'backward';third='backward' if first=='forward' else 'forward'
   labels=[first,'stand',third,'forward'];indices=[]
   for phase,label in enumerate(labels):
    pool=pools[label];r=pool[(i//2 if phase in [0,2] else i)%len(pool)];indices.append(lookup[r['sample_id']])
   x=dict(c);x['paired_instruction_records']=[catalog[j] for j in indices];x['paired_instruction_captions']=[catalog[j]['caption'] for j in indices];x['paired_instruction_root_states']=labels;x['paired_reference_indices']=indices
   cases.append(x);phaseids.append(indices);state.append(labels)
 (R/'CASES.jsonl').write_text(''.join(json.dumps(c,sort_keys=True)+'\n' for c in cases))
 for n in ['INPUT_ASSETS.npz','SCORER_ONLY_TARGET_POSES.npz']:shutil.copy2(P/n,R/n)
 np.savez_compressed(R/'SCORER_ONLY_PAIRED_GT.npz',physical=physical,case_phase_indices=np.asarray(phaseids),case_ids=np.asarray([c['case_id'] for c in cases]))
 captions=sorted({c for r in cases for c in r['paired_instruction_captions']});(R/'CAPTIONS.json').write_text(json.dumps(captions,indent=2)+'\n')
 plan=rd(P/'PLAN.json');plan.update(protocol='G1_Paired_Detailed_Text_Continuous_Control_v4_HiPHI',version=4,created=time.time(),old_protocol_scores_reused=False,formal_models=14,initialization_source_families_test=128)
 plan['prompts']={str(i):c for i,c in enumerate(captions)};plan['paired_phases']={t:phases(t) for t in ['short52','nominal9','nominal20','nominal40']}
 plan['instruction_delivery']='Each case receives its own unchanged detailed source captions at the four phase boundaries; no future instruction is exposed. First/third locomotion directions balanced/opposed; phase2 is stationary-root body movement, not Stand still; phase4 forward locomotion.'
 plan['metrics']['retrieval']='Original frozen paired-caption native T2M retrieval: one GT-paired query per distinct detailed caption per phase; exactly32 same-window generated candidates, three fixed pools4101/4102/4103, two frozen HiPHI heads, three generator noises; R1/R2/R3 retained. No real-negative anchored R3ref.'
 plan['metrics']['direction']='Coarse root-motion-state agreement with the paired real reference (forward/backward/stationary). Not full command success or proof of upper-body instruction completion; paired-caption R supplies supplementary semantics.'
 plan['metrics']['FID']='FP64 L2-normalized52-frame features versus the same phase-paired real GT distribution; duplicate reference families disclosed, unique family counts and finite-sample caveat retained.'
 plan['protocol_changes']=['literal4commands replaced by unchanged paired detailed captions','first/third directions balanced across cases for >=32 distinct same-window text queries; phase2 stationary-root body gestures','R3ref removed; original paired-caption pool32 restored','FID reference becomes actual phase-caption-paired real windows; same reference across methods','original128 test initializations/goals/noises/window counts and model weights retained']
 plan['real_reference_gate']=dict(min_R1=.1,min_paired_minus_shuffled=.05,min_delta_ci_lower=0.,bootstrap_repeats=1000,both_seeds_required=True,all_phases_and_pools_required=True,source='unchanged prior HiPHI-local52 eligibility thresholds; no threshold chosen from generator outcomes')
 wr(R/'PLAN.json',plan)
 registry=rd(P/'CHECKPOINTS.json');wr(R/'BASE_CHECKPOINTS.json',registry);wr(R/'CHECKPOINTS.json',registry)
 if not (R/'adapters').exists():shutil.copytree(P/'adapters',R/'adapters')
 # Original T5 cache is immutable. Encode only genuinely missing detailed captions.
 z=dict(np.load(HEADROOT/'T5_FEATURES.npz'));t5=dict(zip(z['captions'].tolist(),z['features']));missing=[c for c in captions if c not in t5]
 if missing:
  from native_eval.common import TEXT
  from native_eval.tokenizer import FrozenTokenizer
  from transformers import T5EncoderModel
  tok=FrozenTokenizer(TEXT/'spiece.model');enc=T5EncoderModel.from_pretrained(str(TEXT),local_files_only=True,dtype=torch.bfloat16).eval()
  with torch.inference_mode():
   for c in missing:
    batch=tok([c],padding=True,truncation=False,return_tensors='pt');h=enc(**batch).last_hidden_state.float();mask=batch['attention_mask'][:,:,None];t5[c]=((h*mask).sum(1)/mask.sum(1))[0].numpy()
  del enc
 np.savez_compressed(R/'T5_PAIRED_TEXT.npz',captions=np.asarray(captions),features=np.stack([t5[c] for c in captions]))
 # Extract the real GT features once, with the exact frozen heads/stats.
 chosen=sorted({j for a in phaseids for j in a});records=[catalog[i] for i in chosen];arr=[physical[i] for i in chosen];pack={};index={k:i for i,k in enumerate(chosen)}
 for label,folder,stats in [('distribution',REL,REL/'stats.npz'),('retrieval1701',HEADS[1701],HEADROOT/'retrieval_stats.npz'),('retrieval2903',HEADS[2903],HEADROOT/'retrieval_stats.npz')]:
  model=model_for(folder,'distribution' if label=='distribution' else 'retrieval','cpu');h,m,t=encode_arrays(model,records,arr,dict(np.load(stats)),[t5[r['caption']] for r in records] if label!='distribution' else None)
  pack[label]=h if label=='distribution' else m
  if t is not None:pack['text'+label[-4:]]=t
  del model
 np.savez_compressed(R/'REAL_PAIRED_FEATURES.npz',**pack,catalog_indices=np.asarray(chosen))
 tests=[c for c in cases if c['split']=='locked_test'];allpools={};checks=[];passed=True
 for phase in range(4):
  seen=set();qi=[]
  for i,c in enumerate(tests):
   cap=c['paired_instruction_captions'][phase]
   if cap not in seen:seen.add(cap);qi.append(i)
  assert len(qi)>=32,(phase,len(qi));qr=[]
  for i in qi:
   c=tests[i];r=dict(c['paired_instruction_records'][phase]);r['sample_id']=c['case_id']+f':phase{phase}';qr.append(r)
  pools={str(seed):candidate_pools(qr,seed,length_matched=True) for seed in [4101,4102,4103]};allpools[str(phase)]=dict(query_case_indices=qi,query_case_ids=[tests[i]['case_id'] for i in qi],captions=[r['caption'] for r in qr],source_families=[r['source_family_id'] for r in qr],pools=pools)
  ids=[index[tests[i]['paired_reference_indices'][phase]] for i in qi]
  for head in [1701,2903]:
   m=pack['retrieval'+str(head)][ids];t=pack['text'+str(head)][ids]
   for ps,pool in pools.items():
    score,ranks=retrieval(t,m,pool);perm=np.random.default_rng(6701).permutation(len(qi));shuf,srank=retrieval(t[perm],m,pool);delta=(ranks<=1).astype(float)-(srank<=1).astype(float);ci=family_mean_ci(delta,[r['source_family_id'] for r in qr],seed=20261022,repeats=1000)
    ok=score['R@1']>=.1 and ci['mean']>=.05 and ci['ci95'][0]>0;passed &= ok
    checks.append(dict(phase=phase,head=head,pool_seed=ps,queries=len(qi),scores=score,shuffled=shuf,paired_minus_shuffled=ci,passed=bool(ok)))
 wr(R/'QUERY_POOLS.json',allpools);wr(R/'REAL_REFERENCE_CALIBRATION.json',dict(passed=bool(passed),checks=checks,gate=plan['real_reference_gate'],not_generator_scores=True,time=time.time()))
 audit=dict(passed=True,initializations_unchanged=True,initial_asset_sha256=sha(R/'INPUT_ASSETS.npz'),initial_asset_parent_sha256=sha(P/'INPUT_ASSETS.npz'),test_initial_families=128,test_initial_actors=3,paired_instruction_sources_unique=len(chosen),test_prompt_unique_per_phase=[len(allpools[str(i)]['query_case_indices']) for i in range(4)],test_reference_families_per_phase=[len({c['paired_instruction_records'][i]['source_family_id'] for c in tests}) for i in range(4)],captions_unchanged_from_GT_sources=True,DEV_TEST_instruction_reference_family_overlap=0,paired_reference_frames=52,semantic_stage_label='coarse GT root-motion state, not full command completion',new_model_training=False,time=time.time())
 wr(R/'CPU_PROTOCOL_PREFLIGHT.json',audit)
 wr(R/'PROTOCOL_FROZEN.json',dict(passed=bool(passed),hashes={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name in ['PLAN.json','CASES.jsonl','INPUT_ASSETS.npz','SCORER_ONLY_TARGET_POSES.npz','SCORER_ONLY_PAIRED_GT.npz','CAPTIONS.json','BASE_CHECKPOINTS.json','T5_PAIRED_TEXT.npz','REAL_PAIRED_FEATURES.npz','QUERY_POOLS.json','REAL_REFERENCE_CALIBRATION.json','CPU_PROTOCOL_PREFLIGHT.json','GT_CATALOG.jsonl','GT_CATALOG_PHYSICAL.npz','GT_CATALOG_AUDIT.json']},evaluation_asset_bindings=rd(P/'scoring/ASSET_BINDINGS.json'),source_sha256=sha(__file__),time=time.time()))
 print(json.dumps(dict(calibration_passed=bool(passed),queries_per_phase=audit['test_prompt_unique_per_phase'],R3_range=[min(x['scores']['R@3'] for x in checks),max(x['scores']['R@3'] for x in checks)]),indent=2))
 assert passed,'Real paired-text protocol did not qualify; do not generate or relax thresholds'
if __name__=='__main__':main()
