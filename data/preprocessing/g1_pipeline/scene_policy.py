"""Conservative source-metadata classification; no motion quality values used."""
import re


def classify(row):
    props=str(row.get('props','')).strip().lower()
    posture=str(row.get('body_position','')).strip().lower()
    take=str(row.get('take_id') or row.get('source_motion_id','')).lower()
    category=str(row.get('category','')).lower()
    description=str(row.get('source_description','')).strip().lower()
    # Action categories are independent source metadata too. A take labelled
    # "standing" can still contain a cartwheel or intentional fall; inspecting
    # only the initial posture would select the wrong floor/support protocol.
    tokens=set(re.split(r'[^a-z]+',take+' '+posture+' '+props+' '+category))
    # Curated long descriptions can describe transitions omitted from the
    # coarse posture tag, e.g. "standing ... sits down on the floor".
    described_external=set(re.findall(r'\b(?:ladder|stairs|stair|chair|couch|sofa|bed|underwater|swimming)\b',description))
    metadata_external=tokens & {'ladder','stairs','stair','box','platform','wall','chair','couch','sofa','bed','swimming','swim','underwater'}
    if described_external or metadata_external:
        return {'scene_class':'external_support_or_nonflat','core_eligible':False,
                'reason':'source metadata or technical description names an external scene or support',
                'matched_tokens':sorted(described_external|metadata_external),
                'evidence_field':'source_description' if described_external and not metadata_external else 'metadata_and_description'}
    ground_patterns=[r'\b(?:lying|laying)\b',r'\b(?:lies|lays|lay) down\b',
                     r'\b(?:sit|sits|sitting|sat) (?:down )?on (?:the )?(?:floor|ground)\b',
                     r'\blegs crossed\b',r'\bcross[- ]legged\b',r'\bsit[- ]ups?\b',
                     r'\bkneel(?:s|ing)?\b',r'\bon (?:all fours|hands and knees)\b',
                     r'\b(?:handstand|headstand|cartwheel|somersault)\b',
                     r'\broll(?:s|ing)? on (?:the )?(?:floor|ground)\b']
    described_ground=[match.group(0) for pattern in ground_patterns for match in re.finditer(pattern,description)]
    if described_ground:
        return {'scene_class':'ground_multisupport','core_eligible':False,
                'reason':'source technical description specifies ground/hand/knee support or a transition to it',
                'matched_phrases':sorted(set(described_ground)),'evidence_field':'source_description'}
    ground_tokens={'lying','crawl','crawling','cartwheel','cartwheelin','somersault',
                   'roll','rolling','kneeling','sit','sitting','flipping','flip','falling','fainting'}
    if tokens & ground_tokens or 'all fours' in posture or 'hands and knees' in category:
        return {'scene_class':'ground_multisupport','core_eligible':False,
                'reason':'ground body support requires a separate contact policy',
                'matched_tokens':sorted(tokens & ground_tokens)}
    if props not in ('','0','none','nan','null'):
        return {'scene_class':'object_interaction_review','core_eligible':False,
                'reason':'prop metadata is nonempty; support role must be reviewed','props':props}
    if posture in ('standing','standing, crouching','crouching','standing,crouching'):
        return {'scene_class':'flat_ground_core_candidate','core_eligible':True,
                'reason':'standing/crouching with no prop/support metadata; kinematic QA still required'}
    return {'scene_class':'unclassified','core_eligible':False,'reason':'metadata insufficient for core scene assignment'}
