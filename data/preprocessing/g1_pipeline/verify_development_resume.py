"""Controlled partial-output recovery on one existing development smoke clip."""
import hashlib
from .common import ROOT,write_json,sha
from .bones_locked_smoke import run


def verify():
    namespace='bones_resume_verification_v3'
    kwargs={'max_clips':1,'config_path':ROOT/'configs/bones_soma_rest_v3.json','namespace':namespace}
    first=run(**kwargs);r=first['results'][0]
    if r['status']!='DIAGNOSTIC_COMPLETE':raise RuntimeError('BASE_RUN_FAILED')
    mid=r['sample_id'].split(':',1)[1];path=ROOT/'work'/namespace/(mid+'.npy')
    before=sha(path);mtime=path.stat().st_mtime_ns
    second=run(**kwargs,resume=True)
    verified_skip=path.stat().st_mtime_ns==mtime and sha(path)==before
    # Emulate a process dying mid-output: truncate the dedicated test output only.
    path.write_bytes(b'\x93NUMPY interrupted writer\n')
    third=run(**kwargs,resume=True)
    recovered=sha(path)==before and third['results'][0]['status']=='DIAGNOSTIC_COMPLETE'
    report={'status':'PASS' if verified_skip and recovered else 'FAIL',
            'scope':'single-worker development runner only; production multiworker/interruption acceptance remains pending',
            'sample_id':r['sample_id'],'verified_unchanged_output_skipped':verified_skip,
            'truncated_output_recomputed_identically':recovered,'canonical_output_sha256':before}
    write_json(ROOT/'qa/aggregate/development_resume_verification.json',report)
    print(report,flush=True)
    return report
if __name__=='__main__':verify()
