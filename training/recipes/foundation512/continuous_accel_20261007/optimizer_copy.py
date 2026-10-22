"""Batch only independent FP32-master gradient/compute casts, retain AdamW."""
import types, torch


def install_optimizer_copy(opt):
    assert opt.masters and hasattr(opt, 'optimizer')
    buffers = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures unsupported')
        sources, targets = [], []
        for i, master in self.masters.items():
            grad = self.compute_parameters[i].grad
            if grad is None:
                master.grad = None
                continue
            if grad.is_sparse:
                raise ValueError('Sparse gradients unsupported')
            if i not in buffers:
                buffers[i] = torch.empty_like(master)
            master.grad = buffers[i]
            sources.append(grad.detach())
            targets.append(buffers[i])
        if targets:
            torch._foreach_copy_(targets, sources)
        result = self.optimizer.step()
        torch._foreach_copy_([self.compute_parameters[i] for i in self.masters], list(self.masters.values()))
        return result

    opt.step = types.MethodType(step, opt)

