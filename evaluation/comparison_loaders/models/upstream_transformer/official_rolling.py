"""Thin input/output loop around official MotionMamba2.autoregressive_step.

Uses the official CLI's longest history fitting its trained ten-second window.
Retains the benchmark's initial pose history and all supplied future goals.
No denoiser, tokenizer, temporal mask or diffusion kernel is changed.
"""
import types
import torch
HISTORY_FRAMES=196

def rolling_forward(self,texts,num_frames,num_denoising_steps,pad_mask,first_heading_angle,
                    motion_mask,observed_motion,target_motion=None,cfg_weight=2.,text_feat=None,
                    text_pad_mask=None,cfg_type=None,progress_bar=None,
                    return_text_embeddings_x_dict=False,crop_history_length=None,init_history_sequence=None):
    assert target_motion is None and init_history_sequence is not None
    assert init_history_sequence.shape[1]==24 and crop_history_length is None
    assert cfg_type is None and not return_text_embeddings_x_dict and bool(pad_mask.all())
    horizon=self.gen_horizon_len;assert (num_frames-24)%horizon==0
    sequence=init_history_sequence.clone();trace=[]
    while sequence.shape[1]<num_frames:
        end=sequence.shape[1];start=max(0,end-HISTORY_FRAMES)
        history=sequence[:,start:end]
        mm=motion_mask[:,start:num_frames].clone();obs=observed_motion[:,start:num_frames].clone()
        mm[:,:history.shape[1]]=0.;obs[:,:history.shape[1]]=0.
        # Full remaining observation timeline preserves the common benchmark
        # information: the target exists from the beginning, not only when near.
        prediction=self.autoregressive_step(num_frames=num_frames-start,num_denoising_steps=num_denoising_steps,
            motion_mask=mm,observed_motion=obs,cfg_weight=cfg_weight,texts=texts,
            text_feat=text_feat,text_pad_mask=text_pad_mask,init_history_sequence=history)
        assert prediction.shape[1]==history.shape[1]+horizon
        sequence=torch.cat([sequence,prediction[:,-horizon:]],dim=1)
        trace.append(dict(history_start_frame=start,history_end_frame=end,history_frames=history.shape[1],
            generated_frames=horizon,remaining_input_frames=num_frames-start,
            observed_scalar_count=int(mm.count_nonzero()),all_future_goals_retained=True))
    self._benchmark_rolling_trace=trace
    return sequence

class OfficialRollingModel:
    """A real callable proxy: official MotionMamba2.__call__ bypasses assigned forward."""
    def __init__(self, model, trace_path):
        self.model=model;self.trace_path=trace_path;self.calls=[]
    def __getattr__(self, name):
        return getattr(self.model,name)
    def __call__(self,*args,**kwargs):
        out=rolling_forward(self.model,*args,**kwargs)
        trace=self.model._benchmark_rolling_trace
        assert all(x['history_frames']<=HISTORY_FRAMES and x['generated_frames']==52 for x in trace)
        self.calls.append(dict(batch_size=int(out.shape[0]),windows=trace))
        from util import write
        write(self.trace_path,dict(status='ACTUAL_OFFICIAL_AUTOREGRESSIVE_STEP_EXECUTED',calls=self.calls,
            all_histories_at_most_196=True,full_remaining_goal_timeline_preserved=True))
        return out

def attach(model, trace_path):
    return OfficialRollingModel(model,trace_path)
