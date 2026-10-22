"""Metadata-only compatibility for attributed upstream assets and old experiment saves.

Legacy names here identify upstream formats; they are not the public project name.
Tensor names and values are deliberately never rewritten by this module.
"""
from .paths import resolve_tree

def target_namespace(value):
 replacements=[('ardy.model.ardy_model','motion_mamba2.models.motion_model'),('ardy.motion_rep.reps.ardy_motionrep','motion_mamba2.representation.reps.motion_representation'),('ardy.model','motion_mamba2.models'),('ardy.motion_rep','motion_mamba2.representation'),('ardy.skeleton','motion_mamba2.skeleton'),('ardy_adapter','motion_mamba2.adapters'),('ardy.','motion_mamba2.')]
 if isinstance(value,dict):return {k.replace('ardy_','motion_mamba2_'):target_namespace(v) for k,v in value.items()}
 if isinstance(value,list):return [target_namespace(v) for v in value]
 if isinstance(value,str) and '://' not in value and 'nvidia/' not in value:
  for old,new in replacements:value=value.replace(old,new)
  value=value.replace('.ARDY','.MotionMamba2').replace('.Ardy','.MotionMamba2').replace('ardy_informed_v1','motion_mamba2_informed_v1')
 return value

def checkpoint_metadata(checkpoint):
 result=dict(checkpoint)
 for key in ['config','initialization','motion_contract','stats_sha256']:
  if key in result:result[key]=resolve_tree(target_namespace(result[key]))
 return result
