from typing import Optional

class FullImage:
    __slots__ = ["width", "height", "channels", "total_bytes", "data"]
    def __init__(self):
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.channels: Optional[int] = None
        self.total_bytes: Optional[int] = None
        self.data: Optional[int] = None
