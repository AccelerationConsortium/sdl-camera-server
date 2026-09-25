"""Authenticated camera API; no robot dependencies or robot actuation."""
import asyncio
import hmac
import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from filelock import FileLock

from . import __version__
from .service import CameraService
from .worker import CameraError
from .realsense_captures import CaptureStore, CaptureNotFound, CaptureStoreError


class CaptureRequest(BaseModel):
    label: str | None = Field(None, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=100)
    protected: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


def urls(camera_id):
    base = f'/v1/cameras/{camera_id}'
    return {key: base + '/' + value for key, value in dict(status='status', snapshot='snapshot.jpg',
            stream='stream.mjpg', depth_png='depth.png', depth='depth', intrinsics='intrinsics',
            captures='captures', start='start', stop='stop').items()}


def create_app(config, service=None):
    service = service or CameraService(config)
    store = CaptureStore(config.get('captures', {'enabled': False}))
    store_lock = threading.RLock()
    auth = config.get('tokens', [])
    if not auth or any(len(a.get('token','')) < 24 for a in auth):
        raise ValueError('Configure service tokens of at least 24 characters')
    if len({a['token'] for a in auth}) != len(auth):
        raise ValueError('Duplicate service tokens')

    @asynccontextmanager
    async def lifespan(app):
        # A second API worker/instance with this config must not duplicate owners.
        lock = FileLock(config.get('_lock_path', str(Path(store.root).parent / 'camera-service.lockfile')), thread_local=False)
        await asyncio.to_thread(lock.acquire, timeout=0)
        try:
            if config.get('auto_discover'):
                await asyncio.to_thread(service.refresh_discovery)
            yield
        finally:
            await asyncio.to_thread(service.close)
            lock.release()

    app = FastAPI(title='SDL Camera Server', version=__version__, lifespan=lifespan)
    app.state.service = service

    def principal(authorization: str = Header(default='')):
        token = authorization.removeprefix('Bearer ')
        for entry in auth:
            if hmac.compare_digest(token, entry['token']):
                return entry
        raise HTTPException(401, detail={'error': 'authentication_required'})

    def permitted(p, camera_id):
        if '*' not in p.get('cameras', []) and camera_id not in p.get('cameras', []):
            raise HTTPException(403, detail={'error': 'camera_not_permitted'})

    def worker(camera_id, p):
        permitted(p, camera_id)
        return service.get(camera_id)

    async def run(fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except CameraError as exc:
            raise HTTPException(exc.status, detail={'error': 'camera_error', 'reason': str(exc)})
        except CaptureNotFound:
            raise HTTPException(404, detail={'error': 'capture_not_found'})
        except (ValueError, CaptureStoreError) as exc:
            raise HTTPException(400, detail={'error': 'capture_error', 'reason': str(exc)})

    @app.exception_handler(CameraError)
    async def camera_error(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status, content={'detail': {'error': 'camera_error', 'reason': str(exc)}})

    @app.get('/health')
    async def health():
        return {'status': 'healthy', 'version': __version__}

    @app.get('/v1/discovery')
    async def discovery(p=Depends(principal)):
        if not p.get('admin'):
            raise HTTPException(403, detail='admin_required')
        return await run(service.refresh_discovery)

    @app.get('/v1/cameras')
    async def cameras(p=Depends(principal)):
        out = []
        for camera_id, w in list(service.workers.items()):
            if '*' not in p.get('cameras', []) and camera_id not in p.get('cameras', []):
                continue
            try:
                item = await run(w.call, 'status')
            except HTTPException as exc:
                item = {'id': camera_id, 'camera_id': camera_id, 'kind': w.spec['kind'],
                        'state': 'unavailable', 'present': False, 'streaming': False,
                        'reason': exc.detail}
            caps = ['color', 'snapshot', 'mjpeg', 'captures']
            if w.spec['kind'] == 'realsense':
                caps += ['depth', 'intrinsics', 'diagnostic']
            item.update(id=camera_id, kind=w.spec['kind'], capabilities=caps, urls=urls(camera_id))
            if w.spec['kind'] == 'usb':
                for key in ('depth','depth_png','intrinsics'):
                    item['urls'].pop(key, None)
            out.append(item)
        return {'cameras': out, 'default': out[0]['id'] if len(out)==1 else None}

    @app.get('/v1/cameras/{camera_id}/status')
    async def status(camera_id: str, p=Depends(principal)):
        return await run(worker(camera_id,p).call, 'status')

    @app.post('/v1/cameras/{camera_id}/start')
    async def start(camera_id: str, p=Depends(principal)):
        return await run(worker(camera_id,p).call, 'start')

    @app.post('/v1/cameras/{camera_id}/stop')
    async def stop(camera_id: str, p=Depends(principal)):
        if not p.get('admin'):
            raise HTTPException(403, detail='admin_required')
        return await run(worker(camera_id,p).call, 'stop')

    @app.get('/v1/cameras/{camera_id}/snapshot.jpg')
    async def snapshot(camera_id: str, stream: str = Query('color', pattern='^(color|depth)$'), p=Depends(principal)):
        data, meta = await run(worker(camera_id,p).call, 'snapshot', stream=stream)
        return Response(data, media_type='image/jpeg', headers=frame_headers(meta))

    @app.get('/v1/cameras/{camera_id}/depth.png')
    async def depth_png(camera_id: str, p=Depends(principal)):
        data, meta = await run(worker(camera_id,p).call, 'depth_png')
        return Response(data, media_type='image/png', headers=frame_headers(meta))

    @app.get('/v1/cameras/{camera_id}/intrinsics')
    async def intrinsics(camera_id: str, p=Depends(principal)):
        return await run(worker(camera_id,p).call, 'intrinsics')

    @app.get('/v1/cameras/{camera_id}/depth')
    async def depth(camera_id: str, x: int, y: int, window: int = Query(5,ge=1,le=101), p=Depends(principal)):
        return await run(worker(camera_id,p).call, 'depth_at', x=x,y=y,window=window)

    @app.post('/v1/cameras/{camera_id}/diagnostic')
    async def diagnostic(camera_id: str, start_if_idle: bool=False, p=Depends(principal)):
        if not p.get('admin'):
            raise HTTPException(403, detail='admin_required')
        cid, data = await run(worker(camera_id,p).call, 'diagnostic', start_if_idle=start_if_idle)
        return Response(data, media_type='application/zip', headers={'X-Capture-ID': cid, 'Cache-Control':'no-store'})

    @app.get('/v1/cameras/{camera_id}/stream.mjpg')
    async def stream(camera_id: str, stream: str=Query('color',pattern='^(color|depth)$'),
                     fps: float=Query(10, ge=0.5,le=30,allow_inf_nan=False), p=Depends(principal)):
        w = worker(camera_id,p)
        first = await run(w.call, 'snapshot', stream=stream)
        async def body():
            data, meta = first
            while True:
                yield b'--camera-frame\r\nContent-Type: image/jpeg\r\n\r\n'+data+b'\r\n'
                await asyncio.sleep(1/fps)
                try:
                    data,meta = await asyncio.to_thread(w.call, 'snapshot', stream=stream)
                except CameraError:
                    return
        return StreamingResponse(body(), media_type='multipart/x-mixed-replace; boundary=camera-frame',
                                 headers={'Cache-Control':'no-store'})

    @app.post('/v1/cameras/{camera_id}/captures')
    async def capture(camera_id: str, body: CaptureRequest, p=Depends(principal)):
        w = worker(camera_id,p)
        frames = await run(w.call, 'frameset')
        meta = dict(body.context, label=body.label, tags=body.tags, protected=body.protected,
                    frame=frames['frame'], camera=frames['camera'],
                    intrinsics=frames['frame'].get('intrinsics'), camera_id=camera_id)
        def write():
            with store_lock:
                return store.write(camera_id=camera_id, color_jpeg=frames['color'],depth_png=frames['depth'],meta=meta)
        record = await run(write)
        return capture_result(camera_id, record['capture_id'], record)

    @app.get('/v1/cameras/{camera_id}/captures')
    async def captures(camera_id: str, limit: int=Query(50,ge=1,le=500),node_id: str|None=None,
                       label: str|None=None,since: str|None=None,p=Depends(principal)):
        worker(camera_id,p)
        items = await run(store.list_captures,camera_id=camera_id,limit=limit,node_id=node_id,label=label,since=since)
        return {'camera_id':camera_id,'captures':items,'count':len(items)}

    @app.get('/v1/cameras/{camera_id}/captures/{capture_id}')
    async def capture_meta(camera_id: str,capture_id: str,p=Depends(principal)):
        worker(camera_id,p)
        meta = await run(store.get,camera_id,capture_id)
        return capture_result(camera_id,capture_id,meta)

    @app.get('/v1/cameras/{camera_id}/captures/{capture_id}/{filename}')
    async def capture_file(camera_id: str,capture_id: str,filename: str,p=Depends(principal)):
        worker(camera_id,p)
        path = await run(store.file_path,camera_id,capture_id,filename)
        from fastapi.responses import FileResponse
        return FileResponse(path,media_type='image/jpeg' if filename.endswith('.jpg') else 'image/png')

    @app.delete('/v1/cameras/{camera_id}/captures/{capture_id}')
    async def capture_delete(camera_id: str,capture_id: str,p=Depends(principal)):
        worker(camera_id,p)
        if not p.get('admin'):
            raise HTTPException(403,detail='admin_required')
        def remove():
            with store_lock:
                return store.delete(camera_id,capture_id)
        if not await run(remove):
            raise HTTPException(404,detail='capture_not_found')
        return {'ok':True,'camera_id':camera_id,'capture_id':capture_id}

    @app.get('/v1/store')
    async def store_status(p=Depends(principal)):
        d = await run(store.describe)
        allowed = p.get('cameras',[])
        per_camera = {k:v for k,v in d['cameras'].items() if '*' in allowed or k in allowed}
        d['cameras'] = per_camera
        d['count'] = sum(v['count'] for v in per_camera.values())
        d['bytes'] = sum(v['bytes'] for v in per_camera.values())
        newest = max(per_camera.values(),key=lambda v:v['last_at'] or '',default={})
        d['last_id'],d['last_at'] = newest.get('last_id'),newest.get('last_at')
        d.pop('root',None)
        return d

    return app


def frame_headers(meta):
    headers={'Cache-Control':'no-store','X-Frame-Number':str(meta['frame_number'])}
    for key,header in [('timestamp_ms','X-Frame-Timestamp-Ms'),('depth_scale_m','X-Depth-Scale-M')]:
        if meta.get(key) is not None:
            headers[header] = str(meta[key])
    return headers


def capture_result(camera_id,capture_id,meta):
    base=f'/v1/cameras/{camera_id}/captures/{capture_id}'
    result={'meta':base}
    for filename,key in [('color.jpg','color'),('depth.png','depth')]:
        if filename in meta.get('files',{}):
            result[key]=f'{base}/{filename}'
    return {'camera_id':camera_id,'capture_id':capture_id,'meta':meta,'urls':result}
