"""Configuration, process ownership, and capability discovery."""
import json
import multiprocessing as mp
import re
import threading
import time
from pathlib import Path
from .worker import CameraWorker, CameraError


def _inventory(pipe):
    result = {'cameras': [], 'errors': {}}
    try:
        try:
            from .usb_camera import OpenCVBackend
            for device in OpenCVBackend().devices():
                result['cameras'].append({'id': device['id'], 'kind': 'usb', 'device': device})
        except Exception as exc:
            result['errors']['usb'] = str(exc)
        try:
            from .realsense_camera import RealSenseCamera
            probe = RealSenseCamera({'enabled': True})
            if not probe.installed:
                raise CameraError(probe.install_error)
            for device in probe.list_devices(force=True):
                result['cameras'].append({'id': 'rs-' + device['serial'], 'kind': 'realsense', 'device': device})
        except Exception as exc:
            result['errors']['realsense'] = str(exc)
        pipe.send(result)
    finally:
        pipe.close()


def discover():
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_inventory, args=(child,), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(20):
            raise CameraError('Device enumeration timed out')
        return parent.recv()
    finally:
        process.join(1)
        if process.is_alive():
            process.terminate()
            process.join(3)
        parent.close()


class CameraService:
    def __init__(self, config):
        self.config = config
        self.workers = {}
        self.lock = threading.RLock()
        self.discovery = None
        self._always_on_threads = []
        self._stop_supervisors = threading.Event()
        self._paused = set()
        self._camera_locks = {}
        self._starting_since = {}
        identities = set()
        for spec in config.get('cameras', []):
            if not spec.get('enabled', True):
                continue
            camera_id = spec.get('id', '')
            if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,31}', camera_id) or camera_id in self.workers:
                raise ValueError('Invalid or duplicate camera id')
            if spec.get('kind') not in ('usb', 'realsense'):
                raise ValueError('Camera kind must be usb or realsense')
            if spec['kind'] == 'realsense' and not spec.get('serial'):
                raise ValueError('RealSense aliases require a serial number')
            if spec['kind'] == 'usb':
                match = spec.get('match', {})
                if not match or set(match) - {'identity','path','vid','pid','name'}:
                    raise ValueError('USB selector must use hardware identity, not capture index')
                try:
                    width, height = int(spec.get('width', 1280)), int(spec.get('height', 720))
                except (TypeError, ValueError) as exc:
                    raise ValueError('USB width and height must be positive integers') from exc
                if width <= 0 or height <= 0:
                    raise ValueError('USB width and height must be positive integers')
            identity = (spec['kind'], spec.get('serial') or json.dumps(spec.get('match'), sort_keys=True))
            if identity in identities:
                raise ValueError('Two aliases cannot own the same camera')
            identities.add(identity)
            self.workers[camera_id] = CameraWorker(spec)
            self._camera_locks[camera_id] = threading.Lock()

    def refresh_discovery(self):
        with self.lock:
            self.discovery = discover()
            if self.config.get('auto_discover', False):
                for entry in self.discovery['cameras']:
                    if entry['id'] in self.workers:
                        continue
                    device = entry['device']
                    if entry['kind'] == 'realsense':
                        if any(w.spec.get('serial') == device['serial'] for w in self.workers.values()):
                            continue
                        spec = dict(id=entry['id'], kind='realsense', serial=device['serial'],
                                    idle_timeout_seconds=30)
                    else:
                        if any(w.spec['kind'] == 'usb' and all(device.get(k) == v for k,v in w.spec['match'].items())
                               for w in self.workers.values()):
                            continue
                        spec = dict(id=entry['id'], kind='usb', match={'identity': device['identity']})
                    self.workers[entry['id']] = CameraWorker(spec)
                    self._camera_locks[entry['id']] = threading.Lock()
            return self.discovery

    def start_always_on(self):
        """Keep configured cameras capturing even when nobody is viewing."""
        with self.lock:
            for camera_id, worker in self.workers.items():
                if not worker.spec.get('always_on'):
                    continue
                if any(thread.name == f'camera-supervisor-{camera_id}' for thread in self._always_on_threads):
                    continue
                thread = threading.Thread(target=self._supervise, args=(camera_id,),
                                          name=f'camera-supervisor-{camera_id}', daemon=True)
                self._always_on_threads.append(thread)
                thread.start()

    def _supervise(self, camera_id):
        while not self._stop_supervisors.is_set():
            self._supervise_once(camera_id)
            self._stop_supervisors.wait(5)

    def _supervise_once(self, camera_id):
        with self._camera_locks[camera_id]:
            if camera_id in self._paused or self._stop_supervisors.is_set():
                return
            worker = self.workers[camera_id]
            try:
                status = worker.call('status')
                if self._stop_supervisors.is_set():
                    return
                state = status.get('state')
                if state == 'starting':
                    since = self._starting_since.setdefault(camera_id, time.monotonic())
                    if time.monotonic() - since <= 10:
                        return
                else:
                    self._starting_since.pop(camera_id, None)
                age = status.get('frame_age_s', status.get('last_frame_age_s'))
                stale = state == 'streaming' and (
                    (age is not None and age > 5) or
                    (age is None and (status.get('uptime_s') or 0) > 10))
                if stale or state == 'starting':
                    worker.call('stop')
                if stale or state != 'streaming':
                    worker.call('start')
            except CameraError:
                # Missing/busy cameras are retried without taking down the API.
                pass

    def start_camera(self, camera_id):
        worker = self.get(camera_id)
        with self._camera_locks[camera_id]:
            result = worker.call('start')
            self._paused.discard(camera_id)
            return result

    def stop_camera(self, camera_id):
        worker = self.get(camera_id)
        with self._camera_locks[camera_id]:
            self._paused.add(camera_id)
            return worker.call('stop')

    def get(self, camera_id):
        with self.lock:
            if camera_id not in self.workers:
                raise CameraError('Unknown camera', 404)
            return self.workers[camera_id]

    def close(self):
        self._stop_supervisors.set()
        for thread in self._always_on_threads:
            thread.join(timeout=30)
        for worker in self.workers.values():
            worker.close()
