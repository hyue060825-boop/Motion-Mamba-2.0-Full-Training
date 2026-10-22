"""Reuse the existing live observation for an adjacent 32/48-worker comparison."""
import ast
from pathlib import Path


def run():
    path=Path(__file__).with_name('kinematic_worker_observe.py')
    tree=ast.parse(path.read_text())
    substitutions={'workers32_trial':'workers48_trial','baseline16.json':'baseline32.json',
                   'observation32.json':'observation48.json','baseline16_frames_per_sec':'baseline32_frames_per_sec'}
    class Window(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in substitutions:node.value=substitutions[node.value]
            elif type(node.value) is int and node.value==32:node.value=48
            elif type(node.value) is int and node.value==240:node.value=300
            return node
    ns={'__name__':'authorized_48_worker_observation'}
    exec(compile(ast.fix_missing_locations(Window().visit(tree)),str(path),'exec'),ns)
    ns['run']()


if __name__=='__main__':run()
