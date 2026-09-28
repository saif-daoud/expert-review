from pathlib import Path
from types import SimpleNamespace

import pytest

from server import inference_manager


class FakeGPUAllocator:
    def __init__(self, candidates=(0, 1, 2, 3)):
        self.available = list(candidates)

    def candidates(self, runtime, excluded=None):
        del runtime
        excluded = excluded or set()
        return [gpu for gpu in self.available if gpu not in excluded]


def test_workers_are_scoped_to_a_panel_and_released(monkeypatch, tmp_path: Path):
    created = []

    class FakeWorker:
        def __init__(self, runtime, method, panel_id, server_dir, gpu):
            self.runtime = runtime
            self.method = method
            self.panel_id = panel_id
            self.server_dir = server_dir
            self.gpu = gpu
            self.running = True
            self.stopped = False
            created.append(self)

        def generate(self, method, history):
            assert method == self.method
            return f"{method}:{len(history)}"

        def stop(self):
            self.running = False
            self.stopped = True

    monkeypatch.setattr(inference_manager, "ModelWorker", FakeWorker)
    manager = inference_manager.InferenceManager(
        tmp_path / "unused.sqlite3",
        tmp_path,
        mode="real",
        gpu_allocator=FakeGPUAllocator(),
    )

    assert manager._generate("panel-one", "topas", []) == "topas:0"
    assert manager._generate("panel-one", "topas", [{"role": "patient", "content": "Hi"}]) == "topas:1"
    assert len(created) == 1
    assert created[0].runtime == "base"
    assert created[0].gpu == 0

    assert manager._generate("panel-two", "aria", []) == "aria:0"
    assert len(created) == 2

    manager.release_panel("panel-one")
    assert created[0].stopped is True
    assert "panel-one" not in manager.workers
    assert "panel-two" in manager.workers

    manager.release_panel("panel-one")  # cleanup is idempotent


def test_pausing_unloads_worker_and_requires_reactivation(monkeypatch, tmp_path: Path):
    created = []

    class FakeWorker:
        def __init__(self, runtime, method, panel_id, server_dir, gpu):
            self.runtime = runtime
            self.method = method
            self.panel_id = panel_id
            self.server_dir = server_dir
            self.gpu = gpu
            self.running = True
            self.stopped = False
            created.append(self)

        def generate(self, method, history):
            return f"{method}:{len(history)}"

        def stop(self):
            self.running = False
            self.stopped = True

    monkeypatch.setattr(inference_manager, "ModelWorker", FakeWorker)
    manager = inference_manager.InferenceManager(
        tmp_path / "unused.sqlite3",
        tmp_path,
        mode="real",
        gpu_allocator=FakeGPUAllocator(),
    )

    assert manager._generate("panel-one", "topas", []) == "topas:0"
    first_worker = created[0]

    manager.pause_panel("panel-one")
    assert first_worker.stopped is True
    assert "panel-one" not in manager.workers
    with pytest.raises(RuntimeError, match="session was left"):
        manager._generate("panel-one", "topas", [])
    assert len(created) == 1

    manager.activate_panel("panel-one")
    assert manager._generate("panel-one", "topas", []) == "topas:0"
    assert len(created) == 2
    assert created[1] is not first_worker


def test_idle_worker_is_unloaded_after_one_minute(monkeypatch, tmp_path: Path):
    created = []

    class FakeWorker:
        def __init__(self, runtime, method, panel_id, server_dir, gpu):
            self.runtime = runtime
            self.method = method
            self.panel_id = panel_id
            self.server_dir = server_dir
            self.gpu = gpu
            self.running = True
            self.stopped = False
            created.append(self)

        def generate(self, method, history):
            return f"{method}:{len(history)}"

        def stop(self):
            self.running = False
            self.stopped = True

    monkeypatch.setattr(inference_manager, "ModelWorker", FakeWorker)
    manager = inference_manager.InferenceManager(
        tmp_path / "unused.sqlite3",
        tmp_path,
        mode="real",
        gpu_allocator=FakeGPUAllocator(),
        idle_timeout_seconds=60,
    )
    assert manager._generate("panel-one", "topas", []) == "topas:0"
    last_activity = manager._last_activity["panel-one"]
    assert manager.expire_inactive_workers(now=last_activity + 59.9) == []
    assert manager.expire_inactive_workers(now=last_activity + 60) == ["panel-one"]
    assert created[0].stopped is True
    assert "panel-one" not in manager.workers


def test_cuda_oom_tries_the_next_gpu(monkeypatch, tmp_path: Path):
    created = []

    class FakeWorker:
        def __init__(self, runtime, method, panel_id, server_dir, gpu):
            self.runtime = runtime
            self.method = method
            self.panel_id = panel_id
            self.server_dir = server_dir
            self.gpu = gpu
            self.running = True
            self.stopped = False
            created.append(self)

        def generate(self, method, history):
            if self.gpu == 0:
                raise RuntimeError("CUDA out of memory")
            return f"{method}:{len(history)}:gpu-{self.gpu}"

        def diagnostic_tail(self):
            return ""

        def stop(self):
            self.running = False
            self.stopped = True

    monkeypatch.setattr(inference_manager, "ModelWorker", FakeWorker)
    manager = inference_manager.InferenceManager(
        tmp_path / "unused.sqlite3",
        tmp_path,
        mode="real",
        gpu_allocator=FakeGPUAllocator((0, 1, 2, 3)),
    )
    assert manager._generate("panel-one", "aria", []) == "aria:0:gpu-1"
    assert [worker.gpu for worker in created] == [0, 1]
    assert created[0].stopped is True
    assert manager.workers["panel-one"].gpu == 1


def test_all_full_returns_capacity_error(tmp_path: Path):
    manager = inference_manager.InferenceManager(
        tmp_path / "unused.sqlite3",
        tmp_path,
        mode="real",
        gpu_allocator=FakeGPUAllocator(()),
    )
    with pytest.raises(inference_manager.GPUCapacityError, match="retry again in 5 minutes"):
        manager._generate("panel-one", "archer", [])


def test_gpu_allocator_uses_first_gpu_with_enough_free_memory(monkeypatch):
    monkeypatch.setenv("STUDY_GPU_ORDER", "0,1,2,3")
    monkeypatch.setenv("STUDY_GPU_MIN_FREE_MB_BASE", "18000")
    monkeypatch.setattr(inference_manager.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        inference_manager.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="0, 12000\n1, 17500\n2, 22000\n3, 30000\n"
        ),
    )
    allocator = inference_manager.GPUAllocator()
    assert allocator.candidates("base") == [2, 3]
    assert allocator.candidates("base", {2}) == [3]
