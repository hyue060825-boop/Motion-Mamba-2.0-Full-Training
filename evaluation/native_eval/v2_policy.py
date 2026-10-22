from .common import *

def freeze_policy():
    path=ROOT/'VALIDATION_POLICY_V2.json'
    value=dict(version='native-v2-corrected-20261004',threshold_origin='basic effect/mirror/non-collapse thresholds retained from v1 development; sample-size floor based on 256D covariance; capability scopes separated before any v2 confirmation',
        retrieval=dict(direction='text_to_motion',candidates=32,pool_seeds=[4101,4102,4103],stable_ties='sample_id',real_R1_min=.1,paired_minus_shuffled_min=.05,paired_delta_ci_lower_min=0,max_seed_gap=.15,length_matched_min_queries=32,length_matched_R1_min=.1,ambiguity='unique chosen query captions, family exclusion; other semantically similar motions remain a stated single-positive limitation'),
        direction=dict(source_backed_mirror_pairs_min=32,margin_ci_lower_min=0,position_reflection_rmse_max=.025),
        order=dict(independent_temporal_alignment_required=True,reversal_is_order_test=False,source_caption_only='INSUFFICIENT_EVIDENCE',no_fixed_tiny_N_shortcut=True),
        distribution=dict(feature='pre-projection unnormalized 256D h',dtype='FP64 standard ddof1 FID, no shrinkage/PCA',window='center52',min_families=1024,min_per_side_N=256,effective_rank_min=8,mean_variance_min=.001,bootstrap_repeats=1000,split_seeds=[1004,2004,3004,4004,5004],strong_edits=['static','repeat4'],pass_rule='all five fixed splits positive; percentile and centered basic paired descriptive intervals positive; both training seeds; BONES actor-cluster sensitivity also passes',interval_scope='finite-panel resampling stability, NOT promised population nominal coverage; raw FID bootstrap bias disclosed'),
        local=dict(edits=['shuffle4','shuffle52'],pass_rule='same stability checks, independently from basic FID'),
        selection=dict(retrieval='maximum development R@1, earliest/name stable tie',distribution='among development R within .02 of best and effective rank>=8, lowest raw-center52 bottleneck reconstruction; dynamics tie-break; explicit checkpoint identities',seeds=[1701,2903],all_models_from_random=True,old_confirmation='exposed development only',new_holdout='one confirmation after selection and this policy hashes frozen'),
        provenance=dict(evaluator_holdout='original train family/actor identity hash holdout',generator_unseen_claim=False,normalizer='v2 train family balanced only'),full_sequence='NOT_TESTED without independent continuous validation')
    frozen=ROOT/'POLICY_FROZEN.json'
    if frozen.exists():
        if read(path)!=value or read(frozen)['sha256']!=sha(path):raise ValueError('Policy changed')
        return value
    if (ROOT/'CONFIRMATION_OPENED.json').exists():raise ValueError('Cannot replace policy after holdout exposure')
    write(path,value);write(frozen,dict(sha256=sha(path)));ledger('v2_policy_frozen',sha256=sha(path));return value
