"""Trajectory-local correction and relative anomaly evidence, without retuning."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,BONES,read_jsonl,write_json,sha
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample
from .bones_relative_motion_v2 import labels,series
from .bones_scene_evidence import load
from .calibration_protocol import require_frozen


def run(ids,namespace='bones_frozen_accept_v25'):
    policy=require_frozen();manifest=ROOT/'manifests/bones_pairs_accept.jsonl';rows={r['sample_id']:r for r in read_jsonl(manifest)}
    sc={r['sample_id']:r for r in load(manifest)};baseline=json.loads((ROOT/policy['relative_baseline']).read_text())['baseline'];m,_=compile_target();result=[]
    for sid in ids:
        row=rows[sid];mid=row['source_motion_id'];base=ROOT/'work'/namespace;qp=base/(mid+'.npy');rawp=base/(mid+'.unprocessed.npy');tp=base/(mid+'.time.json');sp=base/(mid+'.solver.json')
        q=np.load(qp);raw=np.load(rawp);t=np.array(json.loads(tp.read_text())['source']['target_timestamps']);solver=json.loads(sp.read_text())
        official,ot,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps']);official,_,_=resample(official,ot,m.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
        values=series(m,q,t);orig=series(m,raw,t);reference=series(m,official,t)
        action=labels({**row,'source_description':sc[sid]['technical_description']});supported=[k for k in action if baseline[k]['supported']]
        fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True);x=t-t[0]
        delta=abs(q[:,7:]-raw[:,7:]);indices=np.argsort(delta.max(axis=0))[-3:]
        for j in indices:axes[0].plot(x,delta[:,j],label=m.joint(j+1).name)
        axes[0].axhline(.35,color='red',ls='--',label='frozen maximum correction');axes[0].set_ylabel('hinge correction rad');axes[0].legend(fontsize=7)
        axes[1].plot(x,q[:,2],label='candidate height');axes[1].plot(x,raw[:,2],label='before postprocessing',ls=':');axes[1].plot(x,official[:,2],label='official height',alpha=.7)
        axes[1].set_ylabel('root height m');axes[1].legend(fontsize=7)
        events=[]
        for key,axis,offset in [('joint_speed_rad_s',axes[2],1),('joint_acceleration_rad_s2',axes[3],2)]:
            if supported:
                threshold=np.min([baseline[label]['thresholds'][key] for label in supported],axis=0)
                ratio=values[key]/threshold;j=int(np.unravel_index(np.argmax(ratio),ratio.shape)[1]);where=np.flatnonzero(ratio[:,j]>1)
                for name,array in [('candidate',values[key]),('before postprocessing',orig[key]),('official',reference[key])]:axis.plot(x[offset:],array[:,j]/threshold[j],label=name)
                axis.axhline(1,color='red',ls='--',label='frozen relative flag');axis.set_ylabel(key+' / threshold');axis.legend(fontsize=7)
                axis.set_title(m.joint(j+1).name,fontsize=9);events.append({'metric':key,'joint':m.joint(j+1).name,'maximum_ratio':float(ratio[:,j].max()),'flagged_times_sec':x[offset:][where].tolist()})
        axes[-1].set_xlabel('time from first source sample, seconds');fig.suptitle(mid);fig.tight_layout()
        path=ROOT/'qa/plots'/(mid+'_acceptance_v25.png');fig.savefig(path,dpi=130);plt.close(fig)
        result.append({'sample_id':sid,'plot_path':str(path.relative_to(ROOT)),'plot_sha256':sha(path),'local_relative_events':events,
            'correction_max_rad':float(delta.max()),'correction_max_frame_joint':list(map(int,np.unravel_index(np.argmax(delta),delta.shape))),
            'human_target_residual_change':solver['human_target_residual_change'],'source_labels':supported,
            'input_bindings':{str(p.relative_to(ROOT)):sha(p) for p in (qp,rawp,tp,sp)}})
    write_json(ROOT/'audit'/f'{namespace}_local_review_plots.json',{'status':'REVIEW_EVIDENCE_NOT_AUTOMATIC_ACCEPTANCE','results':result,'plot_module_sha256':sha(__file__)})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',required=True);a=p.parse_args();run(a.ids)
