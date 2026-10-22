from pathlib import Path
import json
def write(p,x):
 Path(p).write_text(json.dumps(x,default=str))
