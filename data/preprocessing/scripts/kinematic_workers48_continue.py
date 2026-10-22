"""Authorized 48-worker trial; reuse the existing 16/32-worker continuation."""
import ast,argparse
from pathlib import Path


def compiled():
    source=Path(__file__).with_name('kinematic_worker_count_continue.py')
    tree=ast.parse(source.read_text())
    class Allow48(ast.NodeTransformer):
        changes=0
        def visit_Tuple(self,node):
            if all(isinstance(n,ast.Constant) for n in node.elts) and tuple(n.value for n in node.elts)==(16,32):
                node.elts.append(ast.Constant(value=48));self.changes+=1
            return node
        def visit_List(self,node):
            if all(isinstance(n,ast.Constant) for n in node.elts) and [n.value for n in node.elts]==[16,32]:
                node.elts.append(ast.Constant(value=48));self.changes+=1
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value=='scripts/kinematic_worker_count_continue.py':node.value='scripts/kinematic_workers48_continue.py'
            if node.value=='ONLY_AUTHORIZED_16_OR_32_WORKERS':node.value='ONLY_AUTHORIZED_16_32_OR_48_WORKERS'
            return node
    edit=Allow48();tree=edit.visit(tree)
    if edit.changes!=2:raise ValueError('WORKER_COUNT_AUTHORIZATION_PATCH_MISMATCH')
    ns={'__name__':'authorized_48_worker_continuation'}
    exec(compile(ast.fix_missing_locations(tree),str(source), 'exec'),ns)
    return ns


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=[16,32,48],required=True)
    compiled()['run'](p.parse_args().workers)
