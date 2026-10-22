from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'original_graph.py').read_text()
s=s.replace('def __init__(self,limit=8):','def __init__(self,limit=96):')
s=s.replace("or kw.get('current_mask') is not None or kw.get('attention_mask') is not None", "or kw.get('current_mask') is not None")
s=s.replace('att._validated_graph_all_valid=True',"att._validated_graph_all_valid=skw.get('attention_mask') is None")
# Capture masks as inputs, not as assumed constants. Current values are copied
# on every replay by the existing extras loop; changing validity is supported.
needle="        state=kw['state']"
s=s.replace(needle,"""        if getattr(stack,'repair_crop_suffix',False) and mask is not None:
            groups=kw.get('_serial_cfg_groups',1); batch=x.shape[0]//groups
            indices=torch.arange(x.shape[1],device=x.device)+1
            lengths=(mask.reshape(groups,batch,-1).any(1)*indices).amax(1).tolist()
            kw=dict(kw,_repair_prefix_lengths=tuple(max(1,int(n)) for n in lengths))
        state=kw['state']""")
s=s.replace("if k!='_serial_cfg_groups'", "if k not in ('_serial_cfg_groups','_repair_prefix_lengths')")
old="outputs.append(stack._forward_eager(x[start:start+batch],condition[start:start+batch],**part)[0])"
new="""length=kw.get('_repair_prefix_lengths',(x.shape[1],)*groups)[start//batch]
            if length<x.shape[1]:
                for key in ('current_mask','attention_mask'):
                    if part.get(key) is not None:part[key]=part[key][:,:length]
            y=stack._forward_eager(x[start:start+batch,:length],condition[start:start+batch],**part)[0]
            if length<x.shape[1]:y=torch.nn.functional.pad(y,(0,0,0,x.shape[1]-length))
            outputs.append(y)"""
assert old in s;s=s.replace(old,new)
s+='''

def captured_mask(*,key_valid,query_valid,prefix_tokens,causal_within_chunk):
    # Same boolean tensor as the reference, including the safe key-0 row for
    # invalid queries, without scalar reads or dynamic nonzero in capture.
    B,Q=query_valid.shape;K=key_valid.shape[1]
    allowed=key_valid[:,None,:].expand(B,Q,K)
    if causal_within_chunk:
        q=torch.arange(Q,device=key_valid.device)[:,None]
        k=torch.arange(K,device=key_valid.device)[None,:]
        allowed=allowed & (k <= prefix_tokens+q)[None]
    allowed=allowed & query_valid[:,:,None]
    if K>0:
        allowed=torch.cat((allowed[:,:,:1] | (~query_valid)[:,:,None],allowed[:,:,1:]),-1)
    return allowed
'''
(p/'masked_graph_candidate.py').write_text(s)
compile(s,str(p/'masked_graph_candidate.py'),'exec')
print(p/'masked_graph_candidate.py')
