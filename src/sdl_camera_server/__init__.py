"""Camera drivers and service, usable independently of any robot."""
__version__ = "0.2.0"
from .client import CameraClient
__all__ = ["CameraClient"]
