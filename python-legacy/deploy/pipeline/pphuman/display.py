# -*- coding: utf-8 -*-
"""显示/写盘工作线程

工业实时预览做法：推理按自身速度跑，画面按采集帧率刷新。
主线程只管渲染出图，投递到只有 1 帧的队列（新帧覆盖旧帧），
显示线程按 target_fps 稳定刷新窗口并写视频，主线程不再被 imshow/编码阻塞。
"""
import queue
import threading
import time

import cv2


class DisplayWorker(threading.Thread):
    def __init__(self, writer=None, window_name='Paddle-Pipeline',
                 target_fps=30.0, show=True, writer_fps=None):
        super(DisplayWorker, self).__init__(daemon=True)
        self._q = queue.Queue(maxsize=1)
        self._writer = writer
        self._window = window_name
        self._target_fps = target_fps
        self._show = show
        self._writer_fps = writer_fps or target_fps
        self._writer_period = 1.0 / self._writer_fps
        self._last_write = 0.0
        self.stop_requested = False
        self._closed = False

    def submit(self, image):
        """投递待显示图像（非阻塞，新帧覆盖旧帧）。"""
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(image)
        except queue.Full:
            pass

    def run(self):
        period = 1.0 / self._target_fps
        while not self.stop_requested:
            try:
                img = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._writer is not None:
                now = time.perf_counter()
                if now - self._last_write >= self._writer_period:
                    self._writer.write(img)
                    self._last_write = now
            if self._show:
                cv2.imshow(self._window, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_requested = True
                    break
            time.sleep(period)

    def close(self):
        self.stop_requested = True
        if not self._closed:
            self._closed = True
            if self._writer is not None:
                self._writer.release()
            if self._show:
                cv2.destroyWindow(self._window)
