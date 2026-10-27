"""The worker loop dispatches operations to a fake driver; no process is spawned."""
import multiprocessing as mp
import threading

import pytest

from sdl_camera_server import worker as worker_module


class FakeRealSense:
    def __init__(self, settings, camera_id=''):
        self.settings, self.camera_id = settings, camera_id
        self.calls = []
        self.stopped = False

    def calibration(self, *, refresh=False):
        self.calls.append(('calibration', refresh))
        return {'camera_id': self.camera_id, 'refreshed': refresh}

    def describe(self):
        return {'state': 'off'}

    def stop(self):
        self.stopped = True


@pytest.fixture
def realsense_pipe(monkeypatch):
    from sdl_camera_server import realsense_camera
    created = []

    def factory(settings, camera_id=''):
        camera = FakeRealSense(settings, camera_id)
        created.append(camera)
        return camera

    monkeypatch.setattr(realsense_camera, 'RealSenseCamera', factory)
    parent, child = mp.Pipe()
    thread = threading.Thread(target=worker_module._run, args=(child, {'id': 'rs', 'kind': 'realsense'}), daemon=True)
    thread.start()
    yield parent, created
    parent.send(('shutdown', {}))
    assert parent.poll(2) and parent.recv() == (True, None)
    thread.join(2)
    assert created[0].stopped


def test_calibration_operation_passes_refresh(realsense_pipe):
    parent, created = realsense_pipe
    parent.send(('calibration', {}))
    assert parent.recv() == (True, {'camera_id': 'rs', 'refreshed': False})
    parent.send(('calibration', {'refresh': True}))
    assert parent.recv() == (True, {'camera_id': 'rs', 'refreshed': True})
    assert created[0].calls == [('calibration', False), ('calibration', True)]
    assert created[0].settings['enabled'] is True


def test_calibration_on_a_usb_camera_is_409(monkeypatch):
    from sdl_camera_server import usb_camera

    class Manager:
        def discover(self):
            return [{'id': 0, 'device': {'name': 'cam'}}]

        def get(self, device_id):
            return type('Cam', (), {'idle_timeout': 0, 'resolution': None, 'stop': lambda self: None})()

    monkeypatch.setattr(usb_camera, 'CameraManager', Manager)
    parent, child = mp.Pipe()
    thread = threading.Thread(target=worker_module._run,
                              args=(child, {'id': 'web', 'kind': 'usb', 'match': {'name': 'cam'}}), daemon=True)
    thread.start()
    parent.send(('calibration', {}))
    assert parent.recv() == (False, (409, 'This camera has no stored calibration'))
    parent.send(('shutdown', {}))
    parent.recv()
    thread.join(2)
