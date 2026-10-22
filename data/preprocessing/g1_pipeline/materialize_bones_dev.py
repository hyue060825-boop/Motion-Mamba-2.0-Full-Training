"""Materialize the predetermined 200 development pairs, never the acceptance set."""
import tarfile
from .common import ROOT,BONES,read_jsonl,write_json,sha,atomic_bytes,now


def run():
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';rows=read_jsonl(manifest)
    if len(rows)!=200 or any(r['project_split']!='train' for r in rows):raise ValueError('INVALID_DEV_PANEL')
    output=ROOT/'work/bones_pairs_dev';output.mkdir(exist_ok=True)
    wanted={r['source_paths']['soma_uniform']:r for r in rows};results=[]
    for member,row in list(wanted.items()):
        path=output/(row['source_motion_id']+'.bvh');side=path.with_suffix('.source.json')
        if path.exists() and side.exists():
            meta=__import__('json').loads(side.read_text())
            if meta.get('member')==member and meta.get('sha256')==sha(path):
                results.append(meta);wanted.pop(member)
    if wanted:
        with tarfile.open(BONES/'soma_uniform.tar.gz','r|gz') as archive:
            for member in archive:
                row=wanted.get(member.name)
                if row is None:continue
                if not member.isfile():raise ValueError('NOT_REGULAR_MEMBER')
                data=archive.extractfile(member).read();path=output/(row['source_motion_id']+'.bvh')
                atomic_bytes(path,data)
                meta={'sample_id':row['sample_id'],'member':member.name,'archive':str(BONES/'soma_uniform.tar.gz'),
                      'bytes':len(data),'sha256':sha(path),'path':str(path),'time':now()}
                write_json(path.with_suffix('.source.json'),meta);results.append(meta);wanted.pop(member.name)
                if len(results)%10==0:print('materialized',len(results),'/ 200',flush=True)
                if not wanted:break
    report={'status':'PASS' if len(results)==200 and not wanted else 'FAIL','manifest_sha256':sha(manifest),
            'expected':200,'materialized':len(results),'missing':list(wanted),'results':results}
    write_json(ROOT/'audit/bones_dev_materialization.json',report)
    print({k:v for k,v in report.items() if k!='results'},flush=True)
    return report
if __name__=='__main__':run()
