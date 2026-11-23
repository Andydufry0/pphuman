# -*- coding: utf-8 -*-
"""语义分频调度器

工业做法：各检测模块按自身时间常数触发（检测 25Hz / 关键点 10Hz / 行为 2Hz），
而不是整体跳帧。跳帧会同时削弱所有模块；分频只降语义上允许降频的模块。
"""
import time


class TimeGate(object):
    """到点才放行：interval_ms 内只放行一次。"""

    __slots__ = ("interval", "_last")

    def __init__(self, interval_ms):
        self.interval = max(0.0, float(interval_ms) / 1000.0)
        self._last = -1e9

    def due(self, now=None):
        if self.interval <= 0:
            return True
        now = time.perf_counter() if now is None else now
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False

    def reset(self):
        self._last = -1e9
