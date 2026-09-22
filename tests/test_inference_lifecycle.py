from pathlib import Path

import pytest

from server import inference_manager


def test_workers_are_scoped_to_a_panel_and_released(monkeypatch, tmp_path: Path):
    created = []

    class FakeWorker:
        def __init__(self, runtime, method, panel_id, server_dir):
            self.runtime = runtime
            self.method = method
            self.panel_id = panel_id
            self.server_dir = server_dir
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
    )

    assert manager._generate("panel-one", "topas", []) == "topas:0"
    assert manager._generate("panel-one", "topas", [{"role": "patient", "content": "Hi"}]) == "topas:1"
    assert len(created) == 1
    assert created[0].runtime == "base"

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
        def __init__(self, runtime, method, panel_id, server_dir):
            self.runtime = runtime
            self.method = method
            self.panel_id = panel_id
            self.server_dir = server_dir
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
