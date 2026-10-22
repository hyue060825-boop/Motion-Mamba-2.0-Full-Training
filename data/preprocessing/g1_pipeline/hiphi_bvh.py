from __future__ import annotations
import io,json,tarfile
from pathlib import Path
import numpy as np
import zstandard as zstd
from .common import ROOT,HIPHI,write_json,write_jsonl,sha

def parse_header(lines):
    joints=[];stack=[];channels=[];i=0
    while i<len(lines):
        s=lines[i].strip()
        if s=='MOTION': break
        parts=s.split()
        if parts and parts[0] in ('ROOT','JOINT'):
            parent=stack[-1] if stack else None;joints.append({'name':parts[1],'parent':parent,'offset':None,'channels':[]});stack.append(len(joints)-1)
        elif parts and parts[0]=='End' and len(parts)>1 and parts[1]=='Site':
            stack.append(None)
        elif parts and parts[0]=='OFFSET' and stack and stack[-1] is not None:
            joints[stack[-1]]['offset']=[float(x) for x in parts[1:]]
        elif parts and parts[0]=='CHANNELS' and stack and stack[-1] is not None:
            joints[stack[-1]]['channels']=parts[2:2+int(parts[1])]
            channels.extend((stack[-1],x) for x in joints[stack[-1]]['channels'])
        elif s=='}' and stack:stack.pop()
        i+=1
    if i>=len(lines):raise ValueError('MISSING_MOTION')
    frames=int(lines[i+1].split(':',1)[1]);dt=float(lines[i+2].split(':',1)[1])
    if frames<=0 or not np.isfinite(dt) or dt<=0:raise ValueError('INVALID_BVH_TIMING')
    if len(joints)!=55:raise ValueError(f'EXPECTED_55_ARTICULATED_JOINTS_GOT_{len(joints)}')
    if not all(j['offset'] is not None for j in joints):raise ValueError('MISSING_OFFSET')
    return joints,channels,frames,dt,i+3

def inspect_bytes(data):
    # Header only; body is intentionally not loaded before schema validation.
    text=data[:200000].decode('utf-8',errors='strict')
    lines=text.splitlines()
    joints,channels,frames,dt,start=parse_header(lines)
    root=joints[0]
    if root['name']!='Hips':raise ValueError('ROOT_NOT_HIPS')
    rootch=root['channels']
    if rootch[:3]!=['Xposition','Yposition','Zposition'] or rootch[3:]!=['Zrotation','Xrotation','Yrotation']:
        raise ValueError('ROOT_CHANNEL_ORDER_MISMATCH')
    if any(j['channels']!=['Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation'] for j in joints):
        raise ValueError('JOINT_CHANNELS_MISMATCH')
    return {'joints':joints,'channels':len(channels),'frames':frames,'frame_time':dt,'fps':1/dt,
            'unit':'cm','basis':'right-handed Y-up','rotation_order':'Z-X-Y','header_lines':start,
            'schema':'hiphi_body_bvh_55j_v1'}

def extract_selected():
    manifest=[json.loads(x) for x in (ROOT/'manifests/hiphi_smoke100.jsonl').read_text().splitlines() if x.strip()]
    wanted={r['source_motion_id']:r for r in manifest};by_archive={}
    for r in manifest:by_archive.setdefault(r['archive'],set()).add(r['source_motion_id'])
    out=ROOT/'work/hiphi_packages';out.mkdir(parents=True,exist_ok=True);results=[]
    for archive,ids in by_archive.items():
        path=HIPHI/'data'/archive
        with open(path,'rb') as fh:
            reader=zstd.ZstdDecompressor().stream_reader(fh);tar=tarfile.open(fileobj=reader,mode='r|')
            for member in tar:
                for mid in list(ids):
                    suffix=f'/{mid}/motion_actor.bvh'
                    if member.name.endswith(suffix):
                        b=tar.extractfile(member).read();
                        try:info=inspect_bytes(b);status='VALIDATED';reason=None
                        except Exception as exc:info={};status='REJECTED';reason=str(exc)
                        p=out/(mid+'.bvh');p.write_bytes(b)
                        results.append({'sample_id':wanted[mid]['sample_id'],'source_motion_id':mid,'archive':archive,
                            'path':str(p.relative_to(ROOT)),'sha256':sha(p),'status':status,'reason':reason,'bvh':info})
                        ids.remove(mid);break
    missing=[{'source_motion_id':x,'status':'MISSING'} for ids in by_archive.values() for x in ids]
    results.extend(missing);write_jsonl(ROOT/'manifests/hiphi_bvh_validation.jsonl',results)
    write_json(ROOT/'qa/aggregate/hiphi_bvh_validation.json',{'attempted':len(manifest),'results':len(results),
        'validated':sum(x['status']=='VALIDATED' for x in results),'rejected':sum(x['status']=='REJECTED' for x in results),
        'missing':len(missing),'status':'PASS' if len(missing)==0 and all(x['status']=='VALIDATED' for x in results) else 'FAIL'})
    return results

if __name__=='__main__': print(extract_selected()[:2])
