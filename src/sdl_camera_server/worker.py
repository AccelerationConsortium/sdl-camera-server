"""One spawned process per camera; native calls never run in the API process."""
import multiprocessing as mp
import threading
import time
from datetime import datetime, timezone


class CameraError(RuntimeError):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


def _run(pipe, spec):
    camera = None
    try:
        kind = spec['kind']
        if kind == 'realsense':
            from .realsense_camera import RealSenseCamera
            camera = RealSenseCamera(dict(spec, enabled=True), camera_id=spec['id'])
        else:
            from .usb_camera import CameraManager
            manager = CameraManager()
            matches = [d for d in manager.discover()
                       if all(d['device'].get(k) == v for k, v in spec['match'].items())]
            if len(matches) != 1:
                raise CameraError(f'USB selector matched {len(matches)} cameras; expected exactly one')
            camera = manager.get(matches[0]['id'])
        while True:
            operation, args = pipe.recv()
            try:
                if operation == 'shutdown':
                    camera.stop()
                    pipe.send((True, None))
                    break
                if operation == 'status':
                    result = camera.describe()
                    result.update(id=spec['id'], camera_id=spec['id'], kind=kind,
                                  label=spec.get('label', spec['id']))
                elif operation == 'stop':
                    camera.stop()
                    result = camera.describe()
                elif operation == 'start':
                    camera.start(**args) if kind == 'realsense' else camera.start()
                    result = camera.describe()
                elif operation == 'diagnostic':
                    if kind != 'realsense':
                        raise CameraError('Depth diagnostics require RealSense', 409)
                    if args.get('start_if_idle'):
                        camera.start(preserve_sensor_settings=True)
                    result = camera.diagnostic_export()
                elif operation == 'intrinsics':
                    if kind != 'realsense':
                        raise CameraError('This camera has no depth calibration', 409)
                    result = camera.intrinsics()
                elif operation == 'depth_at':
                    if kind != 'realsense':
                        raise CameraError('This camera has no depth stream', 409)
                    result = camera.depth_at(**args)
                elif operation in ('snapshot', 'depth_png', 'frameset'):
                    if kind == 'realsense':
                        camera.ensure_started()
                        bundle = camera.latest()
                        if time.monotonic() - bundle.captured_at > 2:
                            raise CameraError('Latest frame is stale')
                        metadata = dict(frame_number=bundle.frame_number, timestamp_ms=bundle.timestamp_ms,
                                        depth_scale_m=bundle.depth_scale, intrinsics=bundle.intrinsics,
                                        aligned_depth_to_color=camera.align_depth_to_color)
                        if operation == 'snapshot':
                            result = (camera.encode_jpeg(bundle, args.get('stream', 'color')), metadata)
                        elif operation == 'depth_png':
                            result = (camera.encode_depth_png(bundle), metadata)
                        else:
                            result = {'color': camera.encode_jpeg(bundle) if bundle.color is not None else None,
                                      'depth': camera.encode_depth_png(bundle) if bundle.depth is not None else None,
                                      'frame': metadata, 'camera': camera.describe()}
                    else:
                        if operation == 'depth_png' or args.get('stream', 'color') != 'color':
                            raise CameraError('This camera supports color only', 409)
                        camera.start()
                        jpeg, number = camera.jpeg()
                        metadata = {'frame_number': number, 'timestamp_ms': None,
                                    'host_utc': datetime.now(timezone.utc).isoformat()}
                        result = ((jpeg, metadata) if operation == 'snapshot' else
                                  {'color': jpeg, 'depth': None, 'frame': metadata, 'camera': camera.describe()})
                else:
                    raise CameraError('Unknown camera operation', 400)
                pipe.send((True, result))
            except Exception as exc:
                status = getattr(exc, 'status', 400 if isinstance(exc, ValueError) else 503)
                if type(exc).__name__ == 'RealSenseNotStreaming':
                    status = 409
                pipe.send((False, (status, str(exc))))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            pipe.send((False, (503, str(exc))))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if camera:
            camera.stop()
        pipe.close()


class CameraWorker:
    def __init__(self, spec):
        self.spec = spec
        self.lock = threading.Lock()
        self.process = None
        self.pipe = None
        self.context = mp.get_context('spawn')

    def _terminate(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(3)
            if self.process.is_alive():
                raise CameraError('Camera worker could not be stopped; refusing a second owner')
        if self.pipe:
            self.pipe.close()
        self.process = self.pipe = None

    def call(self, operation, **args):
        with self.lock:
            if self.process is None or not self.process.is_alive():
                self._terminate()
                self.pipe, child = self.context.Pipe()
                self.process = self.context.Process(target=_run, args=(child, self.spec), daemon=True)
                self.process.start()
                child.close()
            try:
                self.pipe.send((operation, args))
                if not self.pipe.poll(60 if operation == 'diagnostic' else 25):
                    raise TimeoutError('Camera driver timed out; worker stopped')
                ok, data = self.pipe.recv()
            except (EOFError, OSError, TimeoutError) as exc:
                self._terminate()
                raise CameraError(str(exc)) from exc
            if not ok:
                status, message = data
                # Re-enumerate on next access after missing/busy hardware errors.
                if status == 503:
                    self._terminate()
                raise CameraError(message, status)
            return data

    def close(self):
        with self.lock:
            if self.process and self.process.is_alive():
                try:
                    self.pipe.send(('shutdown', {}))
                    if self.pipe.poll(3):
                        self.pipe.recv()
                except (EOFError, OSError):
                    pass
            self._terminate()
