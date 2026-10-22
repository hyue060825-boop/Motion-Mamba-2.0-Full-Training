"""Text/metadata-only action and support evidence for HML/HiPHI candidates.

No robot residual, output geometry, or retention result enters these labels.
Multiple descriptions may support multiple classes; all are retained.
"""
import re

PATTERNS={
 'locomotion':r'\b(?:walk\w*|run\w*|ran|jog\w*|stroll\w*|step\w*|skip\w*|pac(?:e|es|ing)|march\w*|sprint\w*|scoot\w*|footwork)\b',
 'turning':r'\b(?:turn\w*|spin\w*|circl\w*|pivot\w*)\b',
 'crouching':r'\b(?:crouch\w*|squat\w*|lung\w*|bow\w*|bend\w*|bent|huddl\w*)\b',
 'balance_evasion':r'\b(?:dodg\w*|evad\w*|avoid\w*|duck\w*)\b',
 'jumping':r'\b(?:jump\w*|hop\w*|leap\w*|vault\w*|flip\w*)\b',
 'upper_body':r'\b(?:arm\w*|hand\w*|gestur\w*|throw\w*|punch\w*|wav\w*|reach\w*|grasp\w*|wip\w*|scratch\w*|clap\w*|grab\w*|lift\w*|pick\w*|push\w*|pull\w*|carr\w*|shoulder\w*|shrug\w*|point\w*|touch\w*|catch\w*|salut\w*|swing\w*|paw\w*|palm\w*|manipulation)\b',
 'standing_idle':r'\b(?:stand\w*|stood|idle|look\w*|wait\w*)\b',
 'dancing_exercising':r'\b(?:danc\w*|stretch\w*|kick\w*|exercis\w*|sport\w*|aerobic\w*|tennis|basketball|boxing|swim\w*|sanda|combat|muay thai|martial arts)\b',
 'ground_body_support':r'\b(?:sit|sits|sitting|sat|kneel\w*|lying|lays? down|laying|crawl\w*|somersault\w*|push[- ]?ups?|sit[- ]?ups?)\b|\broll\w* (?:over|on (?:the )?(?:floor|ground))\b',
 'climbing_external':r'\b(?:climb\w*|stairs?|ladder)\b',
}


def classify(row):
    captions=row.get('captions',[])
    text=' '.join([*captions,str(row.get('category','')),str(row.get('lu',''))]).lower().replace('_',' ')
    labels=[label for label,pattern in PATTERNS.items() if re.search(pattern,text)]
    external=re.findall(r'\b(?:chairs?|stairs?|ladders?|couch|sofa|bed|waist high object)\b',text)
    external += re.findall(r'\b(?:climb\w* on(?:to)? (?:something|an object)|sit\w* on (?:a |the )?(?:box|bench)|lean\w* (?:against|on) (?:a |the )?wall)\b',text)
    ground=re.findall(r'\b(?:lying|lays? down|laying|kneel\w*|crawl\w*|somersault\w*|cross[- ]legged|legs crossed|push[- ]?ups?|sit[- ]?ups?)\b',text)
    ground += re.findall(r'\bsit\w* (?:down )?on (?:the )?(?:ground|floor)\b',text)
    sit=bool(re.search(r'\b(?:sit|sits|sitting|sat)\b',text))
    if external:scene='external_support_or_nonflat'
    elif ground:scene='ground_multisupport'
    elif sit and not re.search(r'\b(?:squat\w*|crouch\w*)\b',text):scene='support_ambiguous_review'
    else:scene='flat_ground_core_candidate'
    return {'action_labels':labels,'scene_class':scene,'support_evidence':sorted(set(external+ground)),
            'classification_source':'native captions plus source category/LU; no robot quality',
            'external_scene_required':bool(external),'support_ambiguous':scene=='support_ambiguous_review',
            'action_classification_missing':not labels,'original_captions':captions}
