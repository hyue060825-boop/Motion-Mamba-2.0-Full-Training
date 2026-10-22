"""GT-only HumanML3D detailed captions, original source text and independently checked root state."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,re,time,hashlib,collections
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
R=Path(__file__).resolve().parent
from prepare_scoring import representation,canonical,HEADROOT,verify_assets
import numpy as np,torch
MAN=Path(artifact_path('manifests_8245329bf1'))
DATA=Path(artifact_path('data_3f74c56171'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def wr(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(v):return hashlib.sha256(('pairedtext-v4-20261022:'+v).encode()).hexdigest()
def classify(c):
 c=c.lower()
 if re.search(r'\b(then|before|after|while|until)\b|turn\w*|figure.?eight|circl\w*|curved|back and forth|sideways|side to side',c):return None
 if re.search(r'\b(run\w*|hop\w*|jump\w*|goose.step\w*|crouch\w*|crawl\w*|kneel\w*|stoop\w*|lunge\w*|sit\w*)\b',c):return None
 travel=re.search(r'\b(walk\w*|step\w*|stride\w*|shuffle\w*|march\w*|amble\w*|stroll\w*|tiptoe\w*|mov\w*)\b',c)
 if travel:
  if 'backward' in c or 'backwards' in c:return 'backward'
  if 'forward' in c or 'ahead' in c:return 'forward'
  return None
 if re.search(r'\b(stand\w*|clap\w*|wav\w*|arm\w*|hand\w*|elbow\w*|punch\w*|throw\w*|toss\w*|box\w*|sway\w*|shake\w*|shak\w*|point\w*|stretch\w*|reach\w*|bend\w*|gesture\w*)\b',c):return 'stand'
 return None

def main():
 torch.set_num_threads(2);verify_assets();rep,sk=representation()
 train={json.loads(s)['source_family_id'] for s in Path(artifact_path('data_index_jsonl_9966c1bc60')).open()}
 head=set()
 for n in ['humanml3d_train.jsonl','humanml3d_development.jsonl']:head|={json.loads(s)['source_family_id'] for s in (DATA/n).open()}
 records=[];arrays=[];seen=set();bad=collections.Counter()
 raw=[json.loads(s) for s in (MAN/'accepted_test.jsonl').open()]
 raw=[r for r in raw if r['dataset']=='humanml3d' and not r['mirror_flag'] and r['frames']>=76 and r['source_family_id'] not in train|head]
 raw.sort(key=lambda r:key(r['sample_id']))
 for ri,r in enumerate(raw):
  if r['source_family_id'] in seen:continue
  candidates=[(i,c['text'],classify(c['text'])) for i,c in enumerate(r.get('caption_time_spans',[])) if c['whole_motion']]
  candidates=[(i,c,l) for i,c,l in candidates if l]
  if not candidates:bad['no_single_basic_original_caption']+=1;continue
  a=np.asarray(np.load(r['feature_path']),np.float32);starts=list(dict.fromkeys([(len(a)-52)//2,0,len(a)-52,int(key('crop:'+r['sample_id'])[:12],16)%(len(a)-51)]))
  selection=None
  for caption_index,caption,label in candidates:
   for start in starts:
    x=a[start:start+52].copy();y,g=canonical(rep,[x]);root=g['root_positions'][0].numpy();j=g['posed_joints'][0].numpy();hip=j[:-1,8]-j[:-1,1];f=np.stack((hip[:,2],-hip[:,0]),1);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
    vel=np.diff(root[:,[0,2]],axis=0)*25;speed=np.linalg.norm(vel,axis=1);vf=(vel*f).sum(1)
    if not (.6<=root[:,1].min() and root[:,1].max()<=1.05) or speed.max()>1.5:continue
    match=(vf>.1) if label=='forward' else (vf<-.1) if label=='backward' else (speed<.1)
    if match.mean()<.8:continue
    selection=(start,y[0],float(vf.mean()),float(speed.mean()),float(match.mean()),caption_index,caption,label);break
   if selection:break
  if selection is None:bad['no_GT_motion_state_qualified52']+=1;continue
  assert sha(r['feature_path'])==r['features_hash'];start,y,vf,speed,match,caption_index,caption,label=selection
  rec=dict(sample_id=r['sample_id'],source_family_id=r['source_family_id'],actor_uid=r['actor_id_namespaced'],source_split=r['project_split'],caption=caption,original_caption_index=caption_index,label=label,feature_path=r['feature_path'],feature_sha256=r['features_hash'],crop_start=start,frames=52,mean_forward_speed=vf,mean_planar_speed=speed,coarse_state_matching_fraction=match,caption_basis='unchanged whole-motion original HumanML3D caption; no rewrite, paraphrase or generator/evaluator-output selection',motion_state_basis='independent FK and root velocity; stationary-root does not imply still upper body')
  seen.add(r['source_family_id']);records.append(rec);arrays.append(y)
  if len(records)%32==0:wr(R/'CATALOG_PROGRESS.json',dict(examined=ri+1,qualified=len(records),time=time.time()))
 counts=collections.Counter(r['label'] for r in records);unique={l:len({r['caption'] for r in records if r['label']==l}) for l in counts}
 (R/'GT_CATALOG.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records));np.savez_compressed(R/'GT_CATALOG_PHYSICAL.npz',physical=np.stack(arrays))
 wr(R/'GT_CATALOG_AUDIT.json',dict(passed=True,domain='humanml3d',counts=dict(counts),unique_captions=unique,source_families=len(records),train_family_overlap=0,evaluator_train_dev_family_overlap=0,selection_GT_only=True,generator_outputs_read=False,original_official_project_test_only=True,rejections=dict(bad),inputs=dict(test_manifest=sha(MAN/'accepted_test.jsonl'),generator_train=sha(artifact_path('data_index_jsonl_9966c1bc60')),evaluator_train=sha(DATA/'humanml3d_train.jsonl'),evaluator_development=sha(DATA/'humanml3d_development.jsonl')),time=time.time()))
 print(json.dumps(dict(counts=dict(counts),unique_captions=unique,total=len(records)),indent=2),flush=True)
if __name__=='__main__':main()
