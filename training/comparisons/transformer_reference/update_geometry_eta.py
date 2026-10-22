from motion_mamba2.paths import artifact_path
import json,pathlib,shutil
for root in [pathlib.Path(artifact_path('g1_motion_mamba2_four_gpu_matched100k_20261009_a4a8fc8513')),pathlib.Path(artifact_path('g1_minimal_four_gpu_matched100k_20261009_9abe8ef732'))]:
 p=root/'gates/FULL_UPDATE.json';backup=p.with_name('FULL_UPDATE_BEFORE_NATIVE_GEOMETRY.json');assert not backup.exists();shutil.copy2(p,backup)
 old=json.loads(p.read_text());new=json.loads((root/'gates/GEOMETRY_FULL_UPDATE.json').read_text());assert old['passed'] and new['passed']
 old['measurements'][1].update(median=new['median_seconds'],warm_seconds=[x['seconds'] for x in new['measurements']],source='GEOMETRY_FULL_UPDATE.json',optimizer_updates_discarded=8)
 old.update(timing_updated_for_admitted_geometry=True,previous_proof=str(backup),new_geometry_proof=str(root/'gates/GEOMETRY_FULL_UPDATE.json'))
 p.write_text(json.dumps(old,indent=2)+'\n')
 print(root.name,'future batch512 seconds',new['median_seconds'])
