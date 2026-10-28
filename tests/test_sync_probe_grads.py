"""The probe heads are averaged across ranks before every step.

`training_step` forwards through `model.module`, which bypasses DDP's own
forward, so its Reducer never arms and `loss.backward()` leaves each rank with
the gradient of its own shard. Without the all-reduce, a 4-GPU run trains four
different copies of every head on a quarter of the data each, and every
gathered number is a mixture of the four. This runs the real all-reduce on a
two-process gloo group.
"""

import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _worker(rank, world, init_file, out_dir):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=world)
    from miol.model.offline_trainer import all_reduce_grads
    lin = torch.nn.Linear(3, 1)
    with torch.no_grad():
        lin.weight.grad = torch.full((1, 3), float(rank + 1))     # rank 0: 1, rank 1: 2
        lin.bias.grad = torch.tensor([10.0 * (rank + 1)])
    n = all_reduce_grads(lin.parameters())
    torch.save({"n": n, "w": lin.weight.grad.clone(), "b": lin.bias.grad.clone()},
               os.path.join(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="no torch.distributed")
def test_two_ranks_end_with_the_same_averaged_gradient(tmp_path):
    import torch.multiprocessing as mp
    init_file = tmp_path / "init"
    mp.spawn(_worker, args=(2, str(init_file), str(tmp_path)), nprocs=2, join=True)
    a = torch.load(tmp_path / "rank0.pt"); b = torch.load(tmp_path / "rank1.pt")
    assert a["n"] == b["n"] == 2
    assert torch.allclose(a["w"], torch.full((1, 3), 1.5)) and torch.allclose(b["w"], a["w"])
    assert torch.allclose(a["b"], torch.tensor([15.0])) and torch.allclose(b["b"], a["b"])


def test_outside_a_process_group_it_is_a_no_op():
    from miol.model.offline_trainer import all_reduce_grads
    lin = torch.nn.Linear(2, 1)
    lin.weight.grad = torch.ones(1, 2)
    assert all_reduce_grads(lin.parameters()) == 1
    assert torch.equal(lin.weight.grad, torch.ones(1, 2))


def test_the_sync_happens_before_the_clip_and_step():
    src = (ROOT / "src/miol/model/offline_trainer.py").read_text()
    i = src.index("    def training_step(")
    body = src[i:src.index("\n    def ", i + 10)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert code.index("all_reduce_grads(head_params)") < code.index("clip_grad_norm_(")
    assert code.index("all_reduce_grads(head_params)") < code.index("optimizer.step()")
