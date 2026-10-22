"""Regression tests for the spawn+fake-__main__ family (docs/BUG_spawn_main_pickling.md).

The 2026-07-28 interpreter rewrite moved the code-runner child to a spawn
context; the child then inherited spawn as its own default start method, so
DataLoader workers had to pickle Dataset classes out of a __main__ that was
only an exec dict — every ``num_workers>0`` solution died with PicklingError.
A sibling gap: the exec namespace lacked ``__file__``, so code touching it
raised a NameError that plain ``python runfile.py`` never raises.
"""

from pathlib import Path

from aide.interpreter import Interpreter

DATALOADER_CODE = """
import torch
from torch.utils.data import Dataset, DataLoader

class CactusDataset(Dataset):
    def __len__(self):
        return 8
    def __getitem__(self, i):
        return torch.zeros(3), i

dl = DataLoader(CactusDataset(), batch_size=2, num_workers=2)
print("OK batches:", sum(1 for _ in dl))
"""


def test_dataloader_workers_survive(tmp_path: Path) -> None:
    """A __main__-defined Dataset must survive num_workers>0."""
    interp = Interpreter(tmp_path, timeout=180)
    try:
        result = interp.run(DATALOADER_CODE)
    finally:
        interp.cleanup_session()
    out = "".join(result.term_out)
    assert result.exc_type is None, out
    assert "OK batches: 4" in out


def test_child_defaults_to_fork_start_method(tmp_path: Path) -> None:
    """Generated code sees fork as the default, like plain python on Linux."""
    interp = Interpreter(tmp_path, timeout=30)
    try:
        result = interp.run(
            "import multiprocessing as mp\nprint('start-method:', mp.get_start_method())"
        )
    finally:
        interp.cleanup_session()
    out = "".join(result.term_out)
    assert result.exc_type is None, out
    assert "start-method: fork" in out


def test_exec_namespace_matches_real_cli(tmp_path: Path) -> None:
    """__file__ is defined and readable during exec; the __name__ guard fires."""
    code = (
        "print('file:', __file__)\n"
        "print('lines:', len(open(__file__).read().splitlines()))\n"
        "if __name__ == '__main__':\n"
        "    print('guard-ran')\n"
    )
    interp = Interpreter(tmp_path, timeout=30)
    try:
        result = interp.run(code)
    finally:
        interp.cleanup_session()
    out = "".join(result.term_out)
    assert result.exc_type is None, out
    assert "runfile.py" in out   # __file__ is the absolute path to runfile.py
    assert "guard-ran" in out


def test_main_module_is_real(tmp_path: Path) -> None:
    """Exec-defined classes live in a registered sys.modules['__main__'], so
    pickle-by-reference (direct pickling, torch.save of whole models) works."""
    code = (
        "import pickle, sys\n"
        "class CactusDataset:\n"
        "    pass\n"
        "assert sys.modules['__main__'].CactusDataset is CactusDataset\n"
        "print('pickled-bytes:', len(pickle.dumps(CactusDataset())) > 0)\n"
    )
    interp = Interpreter(tmp_path, timeout=30)
    try:
        result = interp.run(code)
    finally:
        interp.cleanup_session()
    out = "".join(result.term_out)
    assert result.exc_type is None, out
    assert "pickled-bytes: True" in out


EXPLICIT_SPAWN_CODE = """
import torch
from torch.utils.data import Dataset, DataLoader

class SpawnDataset(Dataset):
    def __len__(self):
        return 4
    def __getitem__(self, i):
        return torch.zeros(2), i

if __name__ == "__main__":
    dl = DataLoader(SpawnDataset(), batch_size=2, num_workers=2,
                    multiprocessing_context="spawn")
    print("SPAWN batches:", sum(1 for _ in dl))
"""


def test_explicit_spawn_dataloader_guarded(tmp_path: Path) -> None:
    """A main-guarded script may explicitly request spawn workers: they
    re-import runfile.py and unpickle the Dataset through the registered
    __main__, matching real `python runfile.py` behavior."""
    interp = Interpreter(tmp_path, timeout=180)
    try:
        result = interp.run(EXPLICIT_SPAWN_CODE)
    finally:
        interp.cleanup_session()
    out = "".join(result.term_out)
    assert result.exc_type is None, out
    assert "SPAWN batches: 2" in out


def test_unguarded_explicit_spawn_matches_cli(tmp_path: Path) -> None:
    """UNGUARDED code that explicitly requests spawn fails exactly like plain
    `python runfile.py`: the standard worker-died RuntimeError — not a
    PicklingError, not a hang. The worker's bootstrapping-phase advisory also
    reaches term_out through its inherited stderr descriptor."""
    code = EXPLICIT_SPAWN_CODE.replace(
        'if __name__ == "__main__":\n    dl', "dl"
    ).replace("    print(\"SPAWN", "print(\"SPAWN")
    assert "__main__" not in code   # truly unguarded
    interp = Interpreter(tmp_path, timeout=180)
    try:
        result = interp.run(code)
    finally:
        interp.cleanup_session()
    out = "".join(result.term_out)
    assert result.exc_type == "RuntimeError", (result.exc_type, out[-300:])
    assert "exited unexpectedly" in out
    assert "bootstrapping phase" in out
