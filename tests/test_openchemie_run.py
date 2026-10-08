import importlib.util
import multiprocessing
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_run():
    path = REPO / "benchmarks" / "tools" / "complete_systems" / "openchemie" / "run.py"
    spec = importlib.util.spec_from_file_location("openchemie_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inline_pool_maps_in_order_in_this_process():
    run = load_run()
    with run.InlinePool(16) as pool:
        assert pool.map(lambda x: x * 2, [3, 1, 2], chunksize=128) == [6, 2, 4]
        assert pool.starmap(lambda a, b: a - b, zip([5, 9], [1, 2]), chunksize=128) == [4, 7]
        assert pool.map(lambda _: multiprocessing.current_process().name, [0]) == ["MainProcess"]


def test_pools_in_process_swaps_the_module_pool():
    run = load_run()
    chemistry = types.SimpleNamespace(multiprocessing=multiprocessing)
    run.pools_in_process(chemistry)
    with chemistry.multiprocessing.Pool(16) as pool:
        assert isinstance(pool, run.InlinePool)
    assert multiprocessing.Pool is not run.InlinePool  # the real module is left alone
