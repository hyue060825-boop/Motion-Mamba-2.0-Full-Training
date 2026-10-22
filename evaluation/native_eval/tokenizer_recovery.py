"""One-time evaluator-only failure quarantine. Never touches upstream assets."""
from motion_mamba2.paths import artifact_path
import shutil
import time
from .common import *
from .tokenizer import tokenizer_checks

def recover():
    dest=ROOT/'rejected_tokenizer_v0'
    if dest.exists():raise RuntimeError('Recovery already recorded; do not repeat')
    dest.mkdir()
    for name in ['runs','text','TRAIN_PLAN.json','SMALL_SUBSET_LEARNING_CHECK.json','TRAIN_TEXT_MASK_DIAGNOSTIC.json','IMPLEMENTATION_TESTS.json']:
        source=ROOT/name
        if source.exists():shutil.move(str(source),dest/name)
    pilot=dest/'runs/bones/pilot_seed1701'
    shutil.copytree(pilot,ROOT/'runs/bones/pilot_seed1701')
    report=read(ROOT/'runs/bones/pilot_seed1701/DONE.json')
    report['reuse_scope']='throughput only; v0 tokenizer invalid, no semantic or model selection reuse; subsequent training starts randomly'
    write(ROOT/'runs/bones/pilot_seed1701/DONE.json',report)
    p=ROOT/'GPU_BUDGET.json';budget=read(p)
    if budget.get('active_since'):
        elapsed=time.time()-budget['active_since'];budget['seconds']+=elapsed
        budget['stages'].append(dict(stage=budget['active_stage'],seconds=elapsed,status='interrupted_invalid_tokenizer'))
        budget.update(active_since=None,active_stage=None);write(p,budget)
    known=tokenizer_checks(TEXT/'spiece.model');write(ROOT/'TOKENIZER_CHECKS.json',known)
    issue=dict(cause='Transformers5 T5Tokenizer constructor uses vocab, not legacy vocab_file. Legacy kwargs silently yielded vocabulary104 and UNK-only sentences.',
       evidence=dict(wrong_walk=[3,2,3,2,3,2,3,2,1],wrong_sit=[3,2,3,2,3,2,3,2,1],correct=known),
       intervention='one targeted correction: use original SentencePiece model directly, verify exact token IDs, rebuild all text caches, discard evaluator weights trained with invalid tokens',
       untouched='main generator processes, code, assets and checkpoints',pre_confirmation=True,
       pilot='120 GPU pilot updates retained only for throughput. No repeated pilot or additional architecture search.',
       invalidated_checks='v0 small-subset loss descent is an implementation arithmetic check only, not text semantics proof',
       correction_configurations_used=1,training_must_restart=True)
    write(ROOT/'TOKENIZER_CORRECTION.json',issue)
    ledger('targeted_tokenizer_correction',**issue);status('TOKENIZER_FIXED_REBUILDING_TEXT')
    # Capture the generic model origin metadata without inspecting old embedding caches.
    source=Path(artifact_path('t5_metadata_json_affff26ef5'))
    shutil.copyfile(source,ROOT/'evidence/T5_ORIGIN_METADATA.json')

if __name__=='__main__':recover()
