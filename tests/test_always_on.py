"""Always-on supervision uses fake workers; no physical cameras are opened."""
import time

from sdl_camera_server.service import CameraService
from sdl_camera_server.worker import CameraError


class FakeWorker:
    def __init__(self, spec):
        self.spec = spec
        self.state = 'off'
        self.starts = 0
        self.stops = 0
        self.fail_start = False
        self.closed = False
        self.frame_age_s = 0

    def call(self, operation, **kwargs):
        if operation == 'status':
            return {'state': self.state, 'frame_age_s': self.frame_age_s}
        if operation == 'start':
            self.starts += 1
            if self.fail_start:
                raise CameraError('Camera unavailable')
            self.state = 'streaming'
            return {'state': self.state}
        if operation == 'stop':
            self.stops += 1
            self.state = 'off'
            return {'state': self.state}
        raise AssertionError(operation)

    def close(self):
        self.closed = True


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('Supervisor did not run')


def test_always_on_starts_recovers_and_admin_can_pause(monkeypatch):
    monkeypatch.setattr('sdl_camera_server.service.CameraWorker', FakeWorker)
    service = CameraService({'cameras': [
        {'id': 'rgb', 'kind': 'usb', 'match': {'identity': 'synthetic'}, 'always_on': True},
        {'id': 'manual', 'kind': 'usb', 'match': {'identity': 'other'}},
    ]})
    try:
        service.start_always_on()
        rgb = service.get('rgb')
        wait_until(lambda: rgb.starts == 1)
        assert service.get('manual').starts == 0
        assert rgb.state == 'streaming'

        service.stop_camera('rgb')
        assert rgb.state == 'off'
        service._supervise_once('rgb')
        assert rgb.starts == 1
        service.start_camera('rgb')
        assert rgb.starts == 2

        # A failed device retry is isolated from API callers and other cameras.
        rgb.state = 'off'
        rgb.fail_start = True
        service._supervise_once('rgb')
        assert rgb.starts == 3
        rgb.fail_start = False
        service._supervise_once('rgb')
        assert rgb.state == 'streaming'

        rgb.frame_age_s = 9
        service._supervise_once('rgb')
        assert rgb.stops == 2
        assert rgb.starts == 5
    finally:
        service.close()
    assert rgb.closed
