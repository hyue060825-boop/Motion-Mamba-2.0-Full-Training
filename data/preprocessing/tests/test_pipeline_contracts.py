import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from g1_pipeline.resample import time_grid
from g1_pipeline.robot import header,FORMAT

def test_time_grid_identity():
 t=[i/25 for i in range(5)]; out,m=time_grid(t,fps=25); assert (out==t).all()

def test_header_36_columns(): assert len(header())==36 and FORMAT.startswith('bones_g1')

def test_time_grid_does_not_extrapolate():
 out,m=time_grid([0,.01,.02],fps=25); assert out[-1]<=.02
