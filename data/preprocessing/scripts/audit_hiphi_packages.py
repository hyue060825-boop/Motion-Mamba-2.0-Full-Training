"""Cross-check the fixed HiPHI100 package metadata without retargeting motion."""
import csv
import hashlib
import json
import tarfile
from pathlib import PurePosixPath
import zstandard as zstd
from g1_pipeline.common import ROOT,HIPHI,read_jsonl,write_json,sha,atomic_bytes,now


def run():
    manifest=ROOT/'manifests/hiphi_smoke100.jsonl';rows=read_jsonl(manifest)
    output=ROOT/'work/hiphi_package_evidence_v1';output.mkdir(exist_ok=True)
    global_path=HIPHI/'metadata/hiphi_metadata.csv';actor_path=HIPHI/'metadata/hiphi_actor_metadata.csv'
    global_rows={r['motion_id']:r for r in csv.DictReader(global_path.open())}
    actors={r['actor_id']:r for r in csv.DictReader(actor_path.open())}
    by_archive={}
    for row in rows:by_archive.setdefault(row['archive'],{})[row['source_motion_id']]=row
    design={'time':now(),'mode':'INPUT_METADATA_AUDIT_ONLY_NO_RETARGET','clips':len(rows),'archives':len(by_archive),
            'compressed_bytes_upper_bound':sum((HIPHI/'data'/a).stat().st_size for a in by_archive),
            'workers':1,'gpu':False,'new_output_bytes_estimate':100*10000,
            'inputs':{str(p):sha(p) for p in [manifest,global_path,actor_path]},'code_sha256':sha(__file__)}
    write_json(output/'design.json',design);results=[]
    for archive,selected in by_archive.items():
        archive_report=output/(archive+'.json')
        if archive_report.exists():
            cached=json.loads(archive_report.read_text())
            if cached['design_inputs']==design['inputs'] and cached['code_sha256']==design['code_sha256']:
                results.extend(cached['results']);continue
            raise ValueError('PACKAGE_AUDIT_NAMESPACE_CHANGED')
        found={mid:{'sample_id':row['sample_id'],'source_motion_id':mid,'archive':archive,'members':[],'status':'ERROR'} for mid,row in selected.items()}
        path=HIPHI/'data'/archive
        with path.open('rb') as raw,zstd.ZstdDecompressor().stream_reader(raw) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
            for member in tar:
                parts=PurePosixPath(member.name).parts
                mids=set(parts)&set(selected)
                if not mids:continue
                if len(mids)!=1:raise ValueError('AMBIGUOUS_PACKAGE_MEMBER')
                mid=next(iter(mids));record=found[mid];record['members'].append(member.name)
                if member.isfile() and parts[-1]=='metadata.json':
                    data=tar.extractfile(member).read();metadata=json.loads(data)
                    record['metadata']=metadata;record['metadata_sha256']=hashlib.sha256(data).hexdigest()
                    atomic_bytes(output/(mid+'.metadata.json'),data)
                elif member.isfile() and parts[-1]=='motion_actor.bvh':
                    h=hashlib.sha256();fh=tar.extractfile(member)
                    for data in iter(lambda:fh.read(1<<20),b''):h.update(data)
                    record['bvh_member_sha256']=h.hexdigest()
        for mid,record in found.items():
            failures=[];meta=record.get('metadata',{});global_row=global_rows[mid];actor=actors[global_row['actor_id']]
            if meta.get('motion_id')!=mid or meta.get('dataset')!='HiPHI':failures.append('PACKAGE_IDENTITY_MISMATCH')
            if meta.get('is_hoi') is not False or meta.get('objects')!=[] or global_row['is_hoi'].lower()!='false' or global_row['object_categories'].strip():failures.append('BODY_ONLY_NOT_PROVEN')
            if any('object_tracks' in PurePosixPath(name).parts for name in record['members']):failures.append('OBJECT_TRACK_MEMBER_PRESENT')
            if meta.get('mirrored') is not False or mid.endswith('__mirror') or global_row['mirrored'].lower()!='false':failures.append('UNEXPECTED_MIRROR')
            if meta.get('actor_id')!=global_row['actor_id']:failures.append('ACTOR_MISMATCH')
            if meta.get('frame_count')!=int(global_row['frame_count']):failures.append('FRAME_COUNT_MISMATCH')
            if meta.get('frame_lu')!=global_row['frame_lu'] or meta.get('text_annotation')!=global_row['text_annotation']:failures.append('SEMANTIC_METADATA_MISMATCH')
            if float(meta.get('actor_metadata',{}).get('height_cm',-1))!=float(actor['height_cm']):failures.append('HEIGHT_MISMATCH')
            cached_bvh=ROOT/'work/hiphi_packages'/(mid+'.bvh')
            if not cached_bvh.is_file() or record.get('bvh_member_sha256')!=sha(cached_bvh):failures.append('MATERIALIZED_BVH_NOT_IDENTICAL')
            record.update(status='INPUT_METADATA_PASS' if not failures else 'REJECTED',failures=failures,
                          archive_stat={'size_bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns},
                          stage_acceptance=False)
            results.append(record)
        write_json(archive_report,{'design_inputs':design['inputs'],'code_sha256':design['code_sha256'],'results':list(found.values())})
        write_json(ROOT/'audit/hiphi_package_evidence_v1.json',{'design':design,'expected':len(rows),'measured':len(results),
                   'input_metadata_pass':sum(r['status']=='INPUT_METADATA_PASS' for r in results),
                   'status':'INPUT_METADATA_AUDIT_ONLY','results':results,'robot_retarget_run':False})
        print(archive,len(results),'/',len(rows),'pass',sum(r['status']=='INPUT_METADATA_PASS' for r in results),flush=True)
    return results


if __name__=='__main__':run()
