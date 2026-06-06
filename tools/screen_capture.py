# tools/screen_capture.py
"""
屏幕捕获模块 —— BetterCam（快）→ mss（兜底），输出 numpy BGR 帧。
"""

import time
from typing import Optional, Tuple

import numpy as np

# 优先 BetterCam（2-3x 快），失败则 mss
try:
    import bettercam
    HAS_BETTERCAM = True
except ImportError:
    HAS_BETTERCAM = False

try:
    import mss
    import mss.tools
except ImportError:
    raise ImportError("请安装屏幕捕获库: pip install bettercam mss")


class ScreenCapture:
    """实时屏幕捕获器（BetterCam优先，mss兜底）。

    Usage:
        cap = ScreenCapture(monitor=1, fps=60)
        for frame in cap:
            cv2.imshow("screen", frame)
    """

    def __init__(
        self,
        monitor: int = 1,
        fps: int = 30,
        region: Optional[Tuple[int, int, int, int]] = None,
        downsample: float = 1.0,
    ):
        self._fps = fps
        self._downsample = downsample
        self._monitor_idx = monitor
        self._backend = "bettercam" if HAS_BETTERCAM else "mss"

        if self._backend == "bettercam":
            self._init_bettercam(monitor, region, fps)
        else:
            self._init_mss(monitor, region)

        self._width = int(self._raw_width * downsample)
        self._height = int(self._raw_height * downsample)
        self._frame_time = 1.0 / fps if fps > 0 else 0

        self._last_time = 0.0
        self._frame_count = 0
        self._start_time = time.time()
        self._fallback = None

    def _init_bettercam(self, monitor, region, fps):
        self._cam = bettercam.create(
            device_idx=monitor,
            output_idx=monitor,
            max_buffer_len=64,
        )
        self._cam.start(target_fps=fps, video_mode=True)
        if region:
            self._raw_width, self._raw_height = region[2], region[3]
        else:
            self._raw_width, self._raw_height = 1920, 1080  # 默认

    def _init_mss(self, monitor, region):
        self._sct = mss.mss()
        if monitor < 0 or monitor >= len(self._sct.monitors):
            raise ValueError(f"显示器 {monitor} 无效，可用: 0~{len(self._sct.monitors)-1}")
        base = self._sct.monitors[monitor]
        if region:
            x, y, w, h = region
            self._monitor = {"left": base["left"]+x, "top": base["top"]+y, "width": w, "height": h}
        else:
            self._monitor = base
        self._raw_width = self._monitor["width"]
        self._raw_height = self._monitor["height"]

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        """实际帧率。"""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed

    def grab(self) -> np.ndarray:
        """捕获一帧 BGR。失败时返回黑色帧。"""
        now = time.time()
        if now - self._last_time < self._frame_time:
            time.sleep(self._frame_time - (now - self._last_time))

        try:
            if self._backend == "bettercam":
                img = self._cam.get_latest_frame()
                if img is None:
                    return self._get_fallback()
                img = img[:, :, :3]  # BGRA → BGR
            else:
                img = np.array(self._sct.grab(self._monitor))[:, :, :3]
        except Exception:
            return self._get_fallback()

        if self._downsample != 1.0:
            import cv2
            img = cv2.resize(img, (self._width, self._height))

        self._last_time = time.time()
        self._frame_count += 1
        return img

    def _get_fallback(self):
        if self._fallback is None:
            self._fallback = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return self._fallback.copy()

    def __iter__(self):
        """迭代器：持续返回屏幕帧。"""
        return self

    def __next__(self) -> np.ndarray:
        return self.grab()

    def close(self):
        if self._backend == "bettercam":
            self._cam.stop()
        else:
            self._sct.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def list_monitors():
    """列出所有可用显示器。"""
    if HAS_BETTERCAM:
        print("  后端: BetterCam (高速)")
        for i in range(4):
            try:
                cam = bettercam.create(device_idx=i, output_idx=i, max_buffer_len=1)
                cam.start(target_fps=1, video_mode=True)
                cam.stop()
                print(f"  显示器 {i}: 可用")
            except Exception:
                break
    else:
        print("  后端: mss")
        with mss.mss() as sct:
            for i, m in enumerate(sct.monitors):
                print(f"  显示器 {i}: {m['width']}x{m['height']} @ ({m['left']}, {m['top']})")


if __name__ == "__main__":
    print("可用显示器：")
    list_monitors()

    print("\n按 'q' 退出预览...")
    import cv2

    with ScreenCapture(monitor=1, fps=30) as cap:
        for frame in cap:
            cv2.putText(
                frame,
                f"FPS: {cap.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Screen Capture Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
