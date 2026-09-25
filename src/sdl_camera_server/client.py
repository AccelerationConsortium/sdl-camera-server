"""Synchronous Python API. Only the HTTP dependencies are needed by clients."""
from pathlib import Path
import httpx


class CameraClient:
    def __init__(self, base_url, token, timeout=40):
        self.http = httpx.Client(base_url=base_url.rstrip('/'),
                                 headers={'Authorization': f'Bearer {token}'},
                                 timeout=timeout, trust_env=False,
                                 limits=httpx.Limits(keepalive_expiry=4))

    def request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response

    def cameras(self):
        return self.request('GET', '/v1/cameras').json()

    def status(self, camera_id):
        return self.request('GET', f'/v1/cameras/{camera_id}/status').json()

    def snapshot(self, camera_id, stream='color'):
        return self.request('GET', f'/v1/cameras/{camera_id}/snapshot.jpg',
                            params={'stream': stream}).content

    def depth_png(self, camera_id):
        return self.request('GET', f'/v1/cameras/{camera_id}/depth.png').content

    def capture(self, camera_id, **metadata):
        return self.request('POST', f'/v1/cameras/{camera_id}/captures', json=metadata).json()

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
