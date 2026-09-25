import pytest

@pytest.fixture(autouse=True)
def synthetic_realsense_config(tmp_path,monkeypatch):
    from sdl_camera_server import realsense_camera
    path=tmp_path/'settings'/'realsense.yaml'
    path.parent.mkdir()
    path.write_text('''enabled: true
cameras:
  - {id: rs435i, serial: TEST-D435I, short_label: RS 435i}
  - {id: rs405, serial: TEST-D405, short_label: RS D405, mount: {facing: down}}
''')
    monkeypatch.setattr(realsense_camera,'default_config_path',lambda:str(path))
