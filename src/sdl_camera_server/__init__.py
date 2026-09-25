"""Camera drivers and service, usable independently of any robot."""
__version__ = "0.1.0"
from .client import CameraClient
__all__ = ["CameraClient"]
