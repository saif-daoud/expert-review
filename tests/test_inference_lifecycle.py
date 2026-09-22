from pathlib import Path

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
