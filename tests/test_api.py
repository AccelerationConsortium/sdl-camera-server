import io
from types import SimpleNamespace
from fastapi.testclient import TestClient
from sdl_camera_server.api import create_app
from sdl_camera_server.worker import CameraError
from sdl_camera_server.service import CameraService
import pytest

TOKEN='test-token-with-at-least-32-characters'

class Worker:
    spec={'kind':'usb'}
    def call(self,op,**kwargs):
        if op=='status':return {'id':'overhead','present':True,'state':'off'}
        if op=='snapshot':return b'jpeg',{'frame_number':3}
        if op=='frameset':return {'color':b'jpeg','depth':None,'frame':{'frame_number':3},'camera':{}}
        if op in ('start','stop'):return {'state':'off'}
        raise CameraError('Unsupported',409)

class Service:
    workers={'overhead':Worker()}
    def get(self,cid):
        if cid not in self.workers:raise CameraError('Unknown camera',404)
        return self.workers[cid]
    def close(self):pass
    def start_always_on(self):pass
    def start_camera(self,cid):return self.get(cid).call('start')
    def stop_camera(self,cid):return self.get(cid).call('stop')

@pytest.fixture
def client(tmp_path):
    config={'tokens':[{'token':TOKEN,'cameras':['overhead'],'admin':True}],
            'captures':{'enabled':True,'root':str(tmp_path/'captures')},'_lock_path':str(tmp_path/'test.lock')}
    with TestClient(create_app(config,Service())) as client:
        yield client

def test_auth_scope_and_capabilities(client):
    assert client.get('/health').status_code==200
    assert client.get('/v1/cameras').status_code==401
    client.headers['Authorization']='Bearer '+TOKEN
    camera=client.get('/v1/cameras').json()['cameras'][0]
    assert 'depth' not in camera['capabilities']
    assert 'depth' not in camera['urls']
    assert client.get('/v1/cameras/other/status').status_code==403
    assert client.get('/v1/cameras/overhead/depth.png').status_code==409

def test_snapshot_and_capture_archive(client):
    client.headers['Authorization']='Bearer '+TOKEN
    r=client.get('/v1/cameras/overhead/snapshot.jpg')
    assert r.content==b'jpeg' and r.headers['X-Frame-Number']=='3'
    r=client.post('/v1/cameras/overhead/captures',json={'label':'test','context':{'arm':{'node_id':'home'}}})
    assert r.status_code==200,r.text
    result=r.json()
    assert client.get(result['urls']['color']).content==b'jpeg'
    assert client.get('/v1/cameras/overhead/captures?node_id=home').json()['count']==1
    assert client.get('/v1/store').json()['count']==1
    assert client.delete(result['urls']['meta']).status_code==200
    assert client.get(result['urls']['meta']).status_code==404

def test_mjpeg_has_frame_boundaries_and_skips_duplicate_frames(tmp_path):
    class StreamingWorker(Worker):
        def __init__(self):self.calls=0
        def call(self,op,**kwargs):
            if op=='snapshot':
                self.calls+=1
                if self.calls==1:return b'first',{'frame_number':1}
                if self.calls==2:return b'first',{'frame_number':1}
                if self.calls==3:return b'second',{'frame_number':2}
                raise CameraError('Stopped')
            return super().call(op,**kwargs)
    service=Service()
    service.workers={'overhead':StreamingWorker()}
    config={'tokens':[{'token':TOKEN,'cameras':['overhead'],'admin':True}],
            '_lock_path':str(tmp_path/'test.lock')}
    with TestClient(create_app(config,service)) as stream_client:
        response=stream_client.get('/v1/cameras/overhead/stream.mjpg?fps=30',
                                   headers={'Authorization':'Bearer '+TOKEN})
    assert response.status_code==200
    assert response.content.count(b'--camera-frame\r\n')==2
    assert b'Content-Length: 5\r\nX-Frame-Number: 1\r\n' in response.content
    assert b'Content-Length: 6\r\nX-Frame-Number: 2\r\n' in response.content
    assert service.workers['overhead'].calls==4

def test_duplicate_camera_owner_rejected():
    with pytest.raises(ValueError,match='same camera'):
        CameraService({'cameras':[{'id':'a','kind':'realsense','serial':'123'},
                                   {'id':'b','kind':'realsense','serial':'123'}]})

def test_index_selector_rejected():
    with pytest.raises(ValueError,match='hardware identity'):
        CameraService({'cameras':[{'id':'a','kind':'usb','match':{'index':0}}]})
