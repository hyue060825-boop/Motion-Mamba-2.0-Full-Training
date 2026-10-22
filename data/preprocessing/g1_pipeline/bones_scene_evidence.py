"""Join sealed BONES pair IDs to unchanged source technical descriptions."""
import csv
from .common import ROOT,BONES,sha,read_jsonl
from .scene_policy import classify


def load(manifest):
    rows=read_jsonl(manifest);metadata=BONES/'metadata/seed_metadata_v004.csv';metadata_hash=sha(metadata)
    if any(r['provenance_evidence']['metadata_sha256']!=metadata_hash for r in rows):
        raise ValueError('SEALED_BONES_METADATA_CHANGED')
    wanted={r['source_motion_id'] for r in rows};matches={}
    with metadata.open() as stream:
        for source in csv.DictReader(stream):
            if source['filename'] in wanted:
                if source['filename'] in matches:raise ValueError('DUPLICATE_BONES_METADATA_ID')
                matches[source['filename']]=source
    if set(matches)!=wanted:raise ValueError('MISSING_BONES_METADATA_ID')
    result=[]
    for row in rows:
        source=matches[row['source_motion_id']]
        enriched={**row,'source_description':source['content_technical_description']}
        result.append({'sample_id':row['sample_id'],'metadata_scene_without_description':classify(row),
                       'scene':classify(enriched),'technical_description':enriched['source_description'],
                       'source_body_position':row.get('body_position'),'source_category':row.get('category'),
                       'metadata_row_id':source['filename'],'metadata_sha256':metadata_hash})
    return result
