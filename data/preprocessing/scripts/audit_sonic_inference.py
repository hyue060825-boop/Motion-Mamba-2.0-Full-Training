"""Synthetic stationary input through the real SONIC ONNX pair, CPU only.

An interface check, not a motion tracking experiment or dynamic QA gate.
"""
import io
import sys
import time
import numpy as np
from g1_pipeline.common import ROOT,sha,write_json,atomic_bytes
from g1_pipeline.sonic_observations import SonicObservations


def run():
    sys.path.insert(0,str(ROOT/'deps/onnxruntime_1_22_1'))
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    obs=SonicObservations();models=obs.audit['models'];sessions=[]
    for model in models:
        if sha(model['path'])!=model['sha256']:raise ValueError('SONIC_MODEL_HASH_CHANGED')
        sessions.append(ort.InferenceSession(model['path'],sess_options=options,providers=['CPUExecutionProvider']))
    n=60;times=np.arange(n)*.02;q=np.tile(obs.constants['default_angles'],(n,1));v=np.zeros_like(q)
    quaternion=np.tile([1.,0,0,0],(n,1));encoder_input=obs.encoder(q,v,quaternion,0,quaternion[0],timestamps=times)
    obs.push_state(q[0],v[0],quaternion[0],np.zeros(3),np.zeros(29),time_s=0.)
    start=time.monotonic();token=sessions[0].run(None,{'obs_dict':encoder_input})[0]
    decoder_input=obs.decoder(token);action=sessions[1].run(None,{'obs_dict':decoder_input})[0]
    elapsed=time.monotonic()-start
    token2=sessions[0].run(None,{'obs_dict':encoder_input})[0]
    action2=sessions[1].run(None,{'obs_dict':obs.decoder(token2)})[0]
    if token.shape!=(1,64) or action.shape!=(1,29) or not np.isfinite(token).all() or not np.isfinite(action).all():raise ValueError('SONIC_ONNX_INTERFACE_FAILURE')
    path=ROOT/'audit/sonic_synthetic_inference_arrays.npz';buf=io.BytesIO()
    np.savez(buf,encoder_input=encoder_input,token=token,decoder_input=decoder_input,action=action,joint_target_mujoco=obs.joint_target(action))
    atomic_bytes(path,buf.getvalue())
    report={'status':'SYNTHETIC_INTERFACE_PASS','input':'analytic stationary default joint angles; identity root; zero joint/base velocity; zero prior action',
            'model_sha256':[m['sha256'] for m in models],'onnxruntime_version':ort.__version__,'provider':'CPUExecutionProvider',
            'encoder_input_shape':list(encoder_input.shape),'token_shape':list(token.shape),'decoder_input_shape':list(decoder_input.shape),'action_shape':list(action.shape),
            'repeat_max_abs':float(max(np.max(abs(token-token2)),np.max(abs(action-action2)))),
            'first_inference_pair_elapsed_s':elapsed,'action_max_abs':float(np.max(abs(action))),
            'arrays_sha256':sha(path),'code_hashes':{str(p):sha(p) for p in [ROOT/'g1_pipeline/sonic_observations.py',ROOT/'scripts/audit_sonic_inference.py']},
            'dynamic_qa_run':False,'robot_connection_created':False,'dataset_motion_used':False,
            'limits':'does not establish stability, tracking accuracy, actuator-model compatibility or motion release readiness'}
    write_json(ROOT/'audit/sonic_synthetic_inference.json',report);print(report)


if __name__=='__main__':run()
