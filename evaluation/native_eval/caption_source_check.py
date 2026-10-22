from motion_mamba2.paths import artifact_path
from pathlib import Path
from .common import *

def check():
    out=ROOT/'CAPTION_SOURCE_CHECK.json'
    if out.exists():return read(out)
    base=Path(artifact_path('texts_d59bd5bfcc'));failures=[];checked=0;files={};examples=[]
    for split in ['train','development','confirmation']:
        for r in rows(ROOT/'data'/f'humanml3d_{split}.jsonl'):
            p=base/(r['source_motion_id']+'.txt')
            if not p.exists():failures.append(dict(sample_id=r['sample_id'],reason='missing_source_text'));continue
            source=[]
            for line in p.read_text().splitlines():
                parts=line.split('#')
                if len(parts)!=4:continue
                start,end=float(parts[-2]),float(parts[-1])
                if start==0 and end==0:source.append(parts[0].strip())
            for caption in r['captions']:
                checked+=1
                if caption not in source:failures.append(dict(sample_id=r['sample_id'],caption=caption,reason='not_exact_official_whole_motion_caption'))
            files[str(p)]=sha(p)
    result=dict(checked_captions=checked,source_files=len(files),failures=failures,passed=not failures,
                official_whole_motion_tags=[0.,0.],caption_frame_interval_policy='complete G1 retargeted clip, no arbitrary short-window relabeling')
    write(out,result);write(ROOT/'data/HUMANML_TEXT_SOURCE_HASHES.json',files)
    if failures:raise ValueError('Official source caption mismatch; quarantine required')
    return result

if __name__=='__main__':print(check())
