"""Motion Mamba 2.0. Importing does not train, load weights or access the network."""
from pathlib import Path
import sys
__version__ = '2.0.0'
_runtime = str(Path(__file__).resolve().parent/'runtime')
if _runtime not in sys.path:sys.path.insert(0,_runtime)

import os
for _variable in ['MM2_TRITON_PATH','MM2_MAMBA_SOURCE']:
 _directory=os.environ.get(_variable)
 if _directory and _directory not in sys.path:sys.path.insert(0,_directory)
