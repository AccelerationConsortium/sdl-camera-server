# SDL Camera Server

An independent REST service and Python client for USB color webcams and Intel
RealSense cameras. One service runs on the computer physically connected to the
cameras. Robot applications use its API without installing camera drivers.

## Install and run

Python 3.10–3.13 (exclusive upper bound: 3.14). Tested deployment target: Python 3.12.

```sh
uv sync --extra usb --extra realsense
# Copy examples/cameras.example.json to cameras.local.json and set real
# selectors plus a random token (at least 24 characters).
uv run --no-sync sdl-camera-server --config cameras.local.json --port 8070
```

Use a dedicated virtual environment. Omit an extra only on machines that do not
need that driver. Discovery reports missing backends independently. The CLI runs
one API process; a configuration lock rejects duplicate service instances.
Each camera runs in its own spawned worker process. Driver timeouts stop that
worker; the next explicit request can try again. Cameras configured with
`"always_on": true` are started when the service starts and retried every five
seconds if their worker or device fails. They keep capturing with no viewers,
and every MJPEG viewer reads the same camera owner. An administrator's `/stop`
pauses automatic restart until `/start` is called or the service restarts.
Always-on overrides the camera's idle timeout. No robot SDK is required.
USB webcams default to 1280×720. Set the camera's `width` and `height` to request
any supported resolution, including higher modes such as 1920×1080 or
3840×2160. USB cameras reject a driver fallback to another size; select a mode
the webcam supports. RealSense color/depth streams remain at 1280×720 by default,
with native profiles configurable through `color` and `depth`. The example uses
15 fps for both RealSense streams; the code default is 30 fps.
After a RealSense pipeline starts, no frame is served until auto exposure and
white balance have settled: at least `warmup_frames` (default 30) and
`warmup_seconds` (default 2) have passed. The first snapshot after idle is
therefore about two seconds slower. Set both to 0 to serve the first frame.

## Python

Install the Python client from this repository (there is no PyPI release yet):

```sh
pip install "sdl-camera-server @ git+https://github.com/AccelerationConsortium/sdl-camera-server.git"
```

Driver extras are needed by the service, not by HTTP clients.

```python
import os
from pathlib import Path
from sdl_camera_server import CameraClient

with CameraClient("http://127.0.0.1:8070", os.environ["CAMERA_API_TOKEN"]) as camera:
    print(camera.cameras())
    Path("snapshot.jpg").write_bytes(camera.snapshot("overhead"))
    # RealSense only:
    # Path("depth.png").write_bytes(camera.depth_png("depth_camera"))
```

The Python `CameraService` class also provides local worker management. Prefer the
HTTP client when the service is running: another driver instance must not open
the same camera.

For computer vision, decode a snapshot into an array and pass it to your model:

```python
import io
import os
import numpy as np
from PIL import Image
from sdl_camera_server import CameraClient

with CameraClient("http://<camera-host>:<port>", os.environ["CAMERA_API_TOKEN"]) as camera:
    rgb = np.asarray(Image.open(io.BytesIO(camera.snapshot("<camera-id>"))).convert("RGB"))
    # predictions = model(rgb)
    # For a RealSense camera with depth enabled:
    raw_depth = np.asarray(Image.open(io.BytesIO(camera.depth_png("<camera-id>"))))
    depth_info = camera.request("GET", "/v1/cameras/<camera-id>/intrinsics").json()
    depth_metres = raw_depth.astype(np.float32) * depth_info["depth_scale_m"]
```

The snapshot and depth requests can read different frames. For a matching RGB
and depth pair, use `POST /v1/cameras/{id}/captures` and fetch its two image
artifacts. Only use corresponding pixel coordinates when `aligned_to` from
`/intrinsics` is `color`.

## API

`/health` and OpenAPI documentation are public. Every `/v1/*` route requires
`Authorization: Bearer <token>`. Credentials restrict the camera IDs a caller
can access. Only administrators can discover hardware, globally stop capture,
export diagnostics, or delete captures. Lab gateways keep their existing user
identity and robot claim rules; service credentials are not end-user credentials.

- `GET /v1/discovery`: OS hardware inventory; administrator only.
- `GET /v1/cameras`: assigned cameras, states, capabilities, and URLs.
- `GET /v1/cameras/{id}/status`: camera state, without starting video.
- `POST /v1/cameras/{id}/start`, `/stop`: camera lifecycle.
- `GET /v1/cameras/{id}/snapshot.jpg?stream=color`: JPEG; starts on demand.
- `GET /v1/cameras/{id}/stream.mjpg?fps=10`: MJPEG; 0.5–30 output fps.
- RealSense: `GET .../depth.png`, `.../intrinsics`, `.../depth?x=100&y=100`.
- RealSense: `POST .../diagnostic?start_if_idle=true`: bounded raw diagnostic ZIP.
- `POST .../captures`: persist a paired capture with label/tags/context.
- `GET .../captures`, `.../captures/{capture_id}`: capture metadata.
- `GET .../captures/{capture_id}/{filename}`: image artifact.
- `DELETE .../captures/{capture_id}`: administrator deletion.

For an RGB preview, give each viewer a credential scoped to that camera and
connect to `/v1/cameras/{id}/stream.mjpg?stream=color&fps=10` with a Bearer
authorization header. Multiple viewers can connect simultaneously; the server
captures once per camera and sends each viewer new frames at up to its requested
rate. A plain browser `<img>` cannot attach the Bearer header, so a browser UI needs an
authenticated gateway or a client that fetches with that header. Reconnect the
viewer if a camera outage ends its MJPEG response.

For a webcam-style preview, `always_on` on the RGB camera avoids startup delay
for each viewer. On a RealSense camera, `always_on` also runs depth continuously
when `depth.enabled` is true. Set `depth.enabled` to false if the PC only needs
RGB; depth endpoints and paired depth captures then have no depth data. Depth
on demand would require a pipeline mode switch and a short RGB interruption.

Frames and archives are data, not repository content. Keep capture roots outside
checkouts, or in the git-ignored `local/` directory for a self-contained installation. Existing RealSense capture IDs and directory formats are supported.
Context supplied by a robot is explicitly separate from camera timestamps; it is
not a hardware synchronization guarantee. Do not auto-retry capture writes.

## Camera assignment and deployment

Use `/v1/discovery` to find RealSense serials and USB device identities, then bind
meaningful local aliases. USB selectors can use identity/path/name/VID/PID;
ambiguous matches are refused. Numeric camera indices are resolved automatically.
USB port moves can change an identity on cameras without a serial. RealSense
interfaces are excluded from generic webcam discovery.

`auto_discover: true` registers discovered cameras using generated IDs when no
alias already matches. It is useful for initial setup; explicitly bind aliases
before using images as workflow evidence. No camera opens during discovery.

Default binding is loopback. For remote clients, bind a specific private network
address and restrict Windows Firewall/Tailscale ACLs to approved callers. Never
publish a camera service to the public Internet. Use HTTPS when the network does
not provide encrypted transport. Tokens and deployment files stay outside git.

On Windows, run the environment's Python directly through NSSM, using
`-m sdl_camera_server --config <absolute-config-path> --host <address> --port <port>`.
Use a separate service account/environment and preserve all existing services.
Configure restart recovery and log rotation. Stop the old camera owner before
handover; rollback stops this service before restoring the old owner.

Validate the wheel/driver combination on the target OS; USB 3 bandwidth is shared
across a hub. Test all cameras concurrently, including frame freshness, bounded
stream cancellation, and robot API responsiveness. A camera outage is not a robot
fault unless the workflow explicitly requires its evidence.

## Development

```sh
uv sync --extra test
uv run pytest
```

Unit tests use synthetic frames and fake drivers; they do not move lab equipment.
The package includes camera code extracted from xarm-translocation; see LICENSE
for the original attribution. No private AC dependencies are needed.
