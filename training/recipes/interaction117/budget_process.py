"""Account isolated collection/evaluation once per allocated process group."""
from common import *
import atexit
from motion_training.io import Budget

def start_budget(name,devices,closing=False):
    cfg=read(ROOT/'config.json')
    context=Budget(cfg['budget_root']).timed(name,devices=devices,closing=closing)
    context.__enter__()
    atexit.register(context.__exit__,None,None,None)
