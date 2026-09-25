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

def test_duplicate_camera_owner_rejected():
    with pytest.raises(ValueError,match='same camera'):
        CameraService({'cameras':[{'id':'a','kind':'realsense','serial':'123'},
                                   {'id':'b','kind':'realsense','serial':'123'}]})

def test_index_selector_rejected():
    with pytest.raises(ValueError,match='hardware identity'):
        CameraService({'cameras':[{'id':'a','kind':'usb','match':{'index':0}}]})
