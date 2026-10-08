# -*- coding: utf-8 -*-
"""进程级实时日志缓冲（T52，P2-1 实时日志）。

提供线程安全的环形日志缓冲 + ``sys.stdout`` 捕获器：

- ``LogBuffer``：``list[str]`` 环形缓冲（默认最近 2000 行），带位置游标
  ``position``（单调递增），供前端 ``GET /logs?since=N`` 增量拉取。
- ``capture_stdout()``：contextmanager，把 ``sys.stdout`` 替换为
  ``tee``（写入缓冲 + 保留终端输出），配合 ``api.py`` 里
  VC/pipeline/batch 的执行块使用；
  支持嵌套（同线程递归/多线程交错都能正确还原，多线程共享同一缓冲，
  行间交错由调用方自然产生，日志查看器按行显示）。
- ``write_line(text)``：直接写一行到缓冲（API 层用它写带前缀的
  进度/状态行，如 ``"进度: 3/12"``、``"批量: 文件 2/5"``），
  不经过 stdout，也不在终端回显。

设计说明：
- 行是缓冲的最小单位；``write`` 收到的任意长度片段按 ``\\n`` 切分，
  未结束的片段暂存于 ``pending``，遇到换行才落盘（完整行）。
- 位置语义：``recent(since)`` 返回 ``(新行列表, 新位置)``；位置只按
  完整行推进，``pending`` 片段会随每次 ``recent`` 附带返回但不推进
  位置（前端展示为"正在输出"的实时行，重复拉取不影响进度条）。
- 线程安全：所有公开方法持锁；``capture_stdout`` 的换入/换出在锁内
  完成，捕获区间本身不持锁（避免长任务期间 /logs 拉取被阻塞）。
"""

from __future__ import annotations

import contextlib
import sys
import threading

__all__ = ["LogBuffer", "log_buffer", "capture_stdout", "write_line",
           "recent", "position", "clear"]

_DEFAULT_MAX_LINES = 2000


class LogBuffer:
    """线程安全的环形日志缓冲（行级）。"""

    def __init__(self, max_lines: int = _DEFAULT_MAX_LINES):
        self._max = max(max_lines, 1)
        self._lines: list[str] = []
        self._pos = 0            # 已丢弃行数（= 缓冲首行的全局序号）
        self._pending = ""       # 未换行结束的片段
        self._lock = threading.RLock()
        self._capture_depth = 0  # 当前活动的 capture_stdout 层数

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def write(self, text) -> int:
        """sys.stdout 兼容写入：按换行切分，完整行入缓冲。"""
        text = str(text)
        with self._lock:
            chunks = text.split("\n")
            chunks[0] = self._pending + chunks[0]
            for chunk in chunks[:-1]:
                self._append_line(chunk.rstrip("\r"))
            self._pending = chunks[-1]
        return len(text)

    def write_line(self, line: str) -> None:
        """直接写一行（自动补换行，不经 stdout）。"""
        with self._lock:
            self._append_line(str(line).rstrip("\r\n"))
            self._pending = ""

    def flush(self) -> None:
        """把未换行片段落为一行（capture 结束时调用）。"""
        with self._lock:
            if self._pending:
                self._append_line(self._pending)
                self._pending = ""

    def _append_line(self, line: str) -> None:
        self._lines.append(line)
        if len(self._lines) > self._max:
            drop = len(self._lines) - self._max
            del self._lines[:drop]
            self._pos += drop

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def recent(self, since: int = 0):
        """返回 ``(新行列表, 新位置)``；``since`` 之前的行已丢弃时返回全部。

        未完成的 pending 片段会附带在末尾（不推进位置，供"实时行"展示）。
        """
        with self._lock:
            if since < self._pos:
                since = self._pos
            start = since - self._pos
            lines = list(self._lines[start:])
            if self._pending:
                lines = lines + [self._pending]
            return lines, self._pos + len(self._lines)

    def all_lines(self) -> list:
        with self._lock:
            lines = list(self._lines)
            if self._pending:
                lines = lines + [self._pending]
            return lines

    @property
    def position(self) -> int:
        with self._lock:
            return self._pos + len(self._lines)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._pos = 0
            self._pending = ""

    @property
    def is_capturing(self) -> bool:
        with self._lock:
            return self._capture_depth > 0


class _Tee:
    """同时写入 LogBuffer 与真实 stdout 的文件对象替身。"""

    def __init__(self, real, buffer: LogBuffer):
        self._real = real
        self._buffer = buffer

    def write(self, text):
        self._buffer.write(text)
        try:
            return self._real.write(text)
        except Exception:  # noqa: BLE001  终端已关闭等不影响缓冲
            return len(text)

    def flush(self):
        self._buffer.flush()
        try:
            self._real.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self):
        try:
            return self._real.isatty()
        except Exception:  # noqa: BLE001
            return False

    def fileno(self):
        return self._real.fileno()

    def writable(self):
        return True

    def __getattr__(self, name):
        # 其它属性（encoding 等）透传真实 stdout
        return getattr(self._real, name)


# ----------------------------------------------------------------------
# 模块级单例与便捷 API
# ----------------------------------------------------------------------
log_buffer = LogBuffer()

_capture_lock = threading.RLock()
_current_tee = None        # 当前活动的共享 tee（多线程捕获共用同一个）
_original_stdout = None    # 首次捕获前的真实 stdout，供最终还原


@contextlib.contextmanager
def capture_stdout():
    """把 ``sys.stdout`` 重定向到进程级缓冲（保留终端回显）。

    用法::

        with capture_stdout():
            vc.vc_single(...)   # 内部 print 全部进入缓冲

    可嵌套（同一线程递归 / 多线程并发均安全）：进程内共享**同一个**
    tee（引用计数式深度），任何线程进入捕获时 print 都会进缓冲；只有
    最后一个捕获退出时才还原真实 stdout。捕获区间不持全局锁，长任务
    期间 ``/logs`` 拉取不受阻塞。
    """
    global _current_tee, _original_stdout
    with _capture_lock:
        if _current_tee is None:
            _original_stdout = sys.stdout
            _current_tee = _Tee(_original_stdout, log_buffer)
            sys.stdout = _current_tee
        log_buffer._capture_depth += 1
    try:
        yield
    finally:
        with _capture_lock:
            log_buffer._capture_depth -= 1
            if log_buffer._capture_depth <= 0:
                log_buffer._capture_depth = 0
                sys.stdout = _original_stdout
                _current_tee = None
                _original_stdout = None
        log_buffer.flush()


def write_line(text: str) -> None:
    """API 层写一行结构化状态（进度/批量提示等），不经 stdout。"""
    log_buffer.write_line(text)


def recent(since: int = 0):
    """便捷读取：``(lines, pos)``。"""
    return log_buffer.recent(since)


def position() -> int:
    return log_buffer.position


def clear() -> None:
    log_buffer.clear()


def _self_test():
    """logstream 自测：写入/环形/位置/捕获。"""
    print("=== logstream._self_test ===")
    ok = True
    clear()

    # 1) write_line + recent 位置
    write_line("a")
    write_line("b")
    lines, pos = recent(0)
    ok &= lines == ["a", "b"] and pos == 2
    lines, pos = recent(2)
    ok &= lines == [] and pos == 2
    print("  write_line/recent PASS:", ok)

    # 2) 捕获：print 进缓冲且终端保留
    with capture_stdout():
        print("c1")
        print("c2")
    lines, pos = recent(pos)
    ok &= lines == ["c1", "c2"] and pos == 4
    print("  capture_stdout PASS:", ok)

    # 3) 环形裁剪：max_lines=3 时最老的行被丢弃，位置正确
    buf = LogBuffer(max_lines=3)
    for i in range(5):
        buf.write_line("L%d" % i)
    lines, pos = buf.recent(0)
    ok &= lines == ["L2", "L3", "L4"] and pos == 5
    ok &= buf.position == 5
    print("  ring-buffer PASS:", ok)

    # 4) 片段写入（模拟 print 分多次 write）
    buf2 = LogBuffer()
    buf2.write("hel")
    buf2.write("lo\nwor")
    lines, pos = buf2.recent(0)
    ok &= lines == ["hello", "wor"] and pos == 1
    buf2.flush()
    lines, pos = buf2.recent(pos)
    ok &= lines == ["wor"] and pos == 2
    print("  fragment PASS:", ok)

    # 5) 嵌套捕获还原
    with capture_stdout():
        print("outer")
        with capture_stdout():
            print("inner")
        print("outer2")
    import io
    sink = io.StringIO()
    old = sys.stdout
    sys.stdout = sink
    try:
        print("after-capture")
        ok &= sink.getvalue() == "after-capture\n"
    finally:
        sys.stdout = old
    print("  nesting/restore PASS:", ok)
    print("  logstream._self_test %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
