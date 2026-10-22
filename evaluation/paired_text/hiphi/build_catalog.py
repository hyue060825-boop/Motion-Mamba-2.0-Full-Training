"""GT-only detailed-caption eligibility; no generator output or new evaluator training."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,re,time,hashlib,collections
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
R=Path(__file__).resolve().parent;P=Path(artifact_path('unified_interactive_easy_v3_hiphi_20261021_32d9db1045'))
sys.path.insert(0,str(P))
from prepare_scoring import representation,canonical,HEADROOT
import numpy as np,torch
MAN=Path(artifact_path('manifests_8245329bf1'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def wr(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(v):return hashlib.sha256(('pairedtext-v4-20261022:'+v).encode()).hexdigest()
def main():
 torch.set_num_threads(2);rep,sk=representation();R.mkdir(exist_ok=True)
 train={json.loads(s)['source_family_id'] for s in Path(artifact_path('data_index_jsonl_9966c1bc60')).open()}
 head=set()
 for n in ['train.jsonl','development.jsonl']:head|={json.loads(s)['source_family_id'] for s in (HEADROOT/n).open()}
 records=[];seen=set();bad=collections.Counter();arrays=[]
 raw=[]
 for split in ['val','test']:
  raw.extend(json.loads(s) for s in (MAN/f'accepted_{split}.jsonl').open())
 raw=[r for r in raw if r['dataset']=='hiphi' and not r['mirror_flag'] and r['frames']>=76 and r['source_family_id'] not in train|head]
 raw.sort(key=lambda r:key(r['sample_id']))
 for ri,r in enumerate(raw):
  if r['source_family_id'] in seen:continue
  caption=r['captions'][0];c=caption.lower();category=r['source_family_id'].split(':')[-1]
  if any(w in c for w in [' then ',' before ',' after ','turn around','figure-eight','circle','curved','back and forth','sideways']):bad['compound_or_route_text']+=1;continue
  first=' '.join(c.split()[:8]);label=None
  if 'Self_motion' in category:
   if 'backward' in first:label='backward'
   elif 'forward' in first or re.match(r'^(walk|step|stride|shuffle|march|amble|stroll|pad|tiptoe|move) (ahead|forwards?)\b',c):label='forward'
   else:bad['no_explicit_locomotion_direction']+=1;continue
   if re.search(r'\b(run\w*|hop\w*|jump\w*|goose.step\w*|crouch\w*|crawl\w*|kneel\w*|stoop\w*|lunge\w*)\b',c):bad['nonbasic_locomotion']+=1;continue
  elif 'Body_movement' in category:
   if re.search(r'\b(walk\w*|step\w*|run\w*|jump\w*|hop\w*|kneel\w*|crouch\w*|sit\w*|crawl\w*)\b',c):bad['nonstationary_or_low_body_prompt']+=1;continue
   label='stand'
  else:bad['outside_basic_categories']+=1;continue
  a=np.asarray(np.load(r['feature_path']),np.float32)
  starts=list(dict.fromkeys([(len(a)-52)//2,0,len(a)-52,int(key('crop:'+r['sample_id'])[:12],16)%(len(a)-51)]))
  selected=None
  for start in starts:
   x=a[start:start+52].copy();y,g=canonical(rep,[x]);root=g['root_positions'][0].numpy();j=g['posed_joints'][0].numpy();hip=j[:-1,8]-j[:-1,1];f=np.stack((hip[:,2],-hip[:,0]),1);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
   vel=np.diff(root[:,[0,2]],axis=0)*25;speed=np.linalg.norm(vel,axis=1);vf=(vel*f).sum(1)
   if not (.6<=root[:,1].min() and root[:,1].max()<=1.05):continue
   if speed.max()>1.5:continue
   match=(vf>.1) if label=='forward' else (vf<-.1) if label=='backward' else (speed<.1)
   if match.mean()<.8:continue
   selected=(start,y[0],float(vf.mean()),float(speed.mean()),float(match.mean()));break
  if selected is None:bad['no_GT_motion_state_qualified52']+=1;continue
  assert sha(r['feature_path'])==r['features_hash'];start,y,vf,speed,match=selected
  rec=dict(sample_id=r['sample_id'],source_family_id=r['source_family_id'],actor_uid=r['actor_id_namespaced'],source_split=r['project_split'],caption=caption,label=label,feature_path=r['feature_path'],feature_sha256=r['features_hash'],crop_start=start,frames=52,mean_forward_speed=vf,mean_planar_speed=speed,coarse_state_matching_fraction=match,caption_basis='original source caption unchanged; no prompt expansion, paraphrase or generated caption',motion_state_basis='independent FK and root velocity, not a learned evaluator output')
  seen.add(r['source_family_id']);records.append(rec);arrays.append(y)
  if len(records)%32==0:wr(R/'CATALOG_PROGRESS.json',dict(examined=ri+1,qualified=len(records),time=time.time()))
 counts=collections.Counter(r['label'] for r in records);unique={label:len({r['caption'] for r in records if r['label']==label}) for label in counts}
 (R/'GT_CATALOG.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records));np.savez_compressed(R/'GT_CATALOG_PHYSICAL.npz',physical=np.stack(arrays))
 wr(R/'GT_CATALOG_AUDIT.json',dict(passed=True,counts=dict(counts),unique_captions=unique,source_families=len(records),train_family_overlap=0,evaluator_train_dev_family_overlap=0,selection_GT_only=True,generator_outputs_read=False,rejections=dict(bad),inputs=dict(test_manifest=sha(MAN/'accepted_test.jsonl'),val_manifest=sha(MAN/'accepted_val.jsonl')),time=time.time()))
 print(json.dumps(dict(counts=dict(counts),unique_captions=unique,total=len(records)),indent=2))
if __name__=='__main__':main()
