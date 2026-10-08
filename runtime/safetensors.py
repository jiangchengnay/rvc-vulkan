# -*- coding: utf-8 -*-
"""safetensors 读取器（纯 Python + numpy，零依赖）。

格式（https://huggingface.co/docs/safetensors）：
    文件 = [8B header_len(uint64 LE)] [header_json(UTF-8)] [原始数据区]
    header_json = {"<name>": {"dtype": str, "shape": [..], "data_offsets": [start, end]},
                   "__metadata__": {...}}
    data_offsets 相对数据区起点。

用法：
    with SafetensorsReader(path) as r:
        names = r.keys()
        arr = r.get("model.embed_tokens.weight")   # numpy.ndarray
    r = load_safetensors(path)                      # 全部加载为 dict
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional

import numpy as np

__all__ = ["SafetensorsReader", "load_safetensors", "save_safetensors"]

_DTYPE_MAP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float32,      # bf16 读入后以 float32 承载（不丢位模式，见 _decode）
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafetensorsReader:
    def __init__(self, path: str):
        self.path = str(path)
        self._f = open(self.path, "rb")
        try:
            head_len = int(np.frombuffer(self._f.read(8), dtype="<u8")[0])
            header = json.loads(self._f.read(head_len).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._f.close()
            raise ValueError("无法解析 safetensors 文件 %s: %s" % (self.path, exc))
        if not isinstance(header, dict):
            self._f.close()
            raise ValueError("无法解析 safetensors 文件 %s: header 不是 JSON 对象" % self.path)
        # P1-008 完整性校验：header 声明的数据区必须在文件范围内，
        # 且各张量 data_offsets 合法、字节数与 dtype/shape 自洽——
        # 截断/损坏文件在打开时即报错，不进入"静默加载垃圾张量"路径。
        self._file_size = os.path.getsize(self.path)
        self._data_start = 8 + head_len
        if self._data_start > self._file_size:
            self._f.close()
            raise ValueError(
                "safetensors 文件截断: header 声明数据区起点 %d > 文件大小 %d (%s)"
                % (self._data_start, self._file_size, self.path))
        self.header: dict = header  # _validate_header 依赖 self.header
        try:
            self._validate_header()
        except Exception as exc:  # noqa: BLE001
            self._f.close()
            raise ValueError("safetensors 完整性校验失败 %s: %s" % (self.path, exc)) from None

    def _validate_header(self) -> None:
        """逐张量校验 data_offsets/字节数自洽（P1-008）。"""
        avail = self._file_size - self._data_start
        for name, meta in self.header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                raise ValueError("张量 %s 缺少 data_offsets" % name)
            start, end = meta["data_offsets"]
            if not (isinstance(start, int) and isinstance(end, int)) or start < 0 or end < start:
                raise ValueError("张量 %s data_offsets 非法: %r" % (name, meta["data_offsets"]))
            if end > avail:
                raise ValueError(
                    "张量 %s 数据越界: end=%d > 数据区可用 %d（文件截断）" % (name, end, avail))
            dtype_str = meta["dtype"]
            if dtype_str not in _DTYPE_MAP:
                raise ValueError("张量 %s 不支持的 dtype: %s" % (name, dtype_str))
            shape = tuple(meta.get("shape", []))
            count = int(np.prod(shape)) if shape else 1
            itemsize = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
                        "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}[dtype_str]
            if (end - start) != count * itemsize:
                raise ValueError(
                    "张量 %s 字节数不自洽: offsets=%d 但 shape%s×%s=%d"
                    % (name, end - start, shape, dtype_str, count * itemsize))

    def keys(self) -> List[str]:
        return [k for k in self.header if k != "__metadata__"]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __enter__(self) -> "SafetensorsReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def _read_tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        nbytes = end - start
        np_dtype = _DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            raise ValueError("不支持的 safetensors dtype: %s" % dtype_str)
        itemsize = nbytes // max(1, int(np.prod(shape))) if shape else nbytes
        self._f.seek(self._data_start + start)
        raw = self._f.read(nbytes)
        if dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            arr = ((u16 & 0x7FFF) << 16).astype(np.uint32)
            arr = (arr | ((u16 >> 15) << 31)).view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.dtype(np_dtype).newbyteorder("<"))
        arr = arr.reshape(shape)
        if dtype_str == "BOOL":
            arr = arr.astype(np.bool_)
        return arr.copy()  # 拷贝，避免持有文件 mmap

    def get(self, name: str) -> np.ndarray:
        if name not in self.header:
            raise KeyError("safetensors 中没有张量: %s（可用: %s）" % (name, list(self.keys())[:8]))
        return self._read_tensor(name)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get(name)

    def nbytes(self, name: Optional[str] = None) -> int:
        if name is None:
            return sum(self.header[k]["data_offsets"][1] - self.header[k]["data_offsets"][0]
                       for k in self.keys())
        s, e = self.header[name]["data_offsets"]
        return e - s


def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    with SafetensorsReader(path) as r:
        return {k: r.get(k) for k in r.keys()}


def save_safetensors(path: str, tensors: Dict[str, np.ndarray],
                     metadata: Optional[dict] = None) -> None:
    """把 dict[str, ndarray] 写为 safetensors 文件（便于权重分发/校验）。"""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    blob = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.bool_:
            dtype = "BOOL"
            raw = arr.astype(np.uint8).tobytes()
        elif arr.dtype in (np.float16,):
            dtype = "F16"
            raw = arr.tobytes()
        elif arr.dtype == np.float32:
            dtype = "F32"
            raw = arr.tobytes()
        elif arr.dtype == np.float64:
            dtype = "F64"
            raw = arr.tobytes()
        elif arr.dtype == np.int64:
            dtype = "I64"
            raw = arr.tobytes()
        elif arr.dtype == np.int32:
            dtype = "I32"
            raw = arr.tobytes()
        elif arr.dtype == np.uint8:
            dtype = "U8"
            raw = arr.tobytes()
        else:
            raise ValueError("暂不支持写出 dtype: %s" % arr.dtype)
        start = len(blob)
        blob += raw
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [start, len(blob)],
        }
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.uint64(len(head)).tobytes())
        f.write(head)
        f.write(bytes(blob))


if __name__ == "__main__":
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "_st_test.safetensors")
    save_safetensors(path, {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.array([1, 0, 1], dtype=np.bool_),
        "c": np.float16(3.5),
    }, {"note": "roundtrip"})
    with SafetensorsReader(path) as r:
        assert r.keys() == ["a", "b", "c"], r.keys()
        assert np.allclose(r["a"], np.arange(6, dtype=np.float32).reshape(2, 3))
        assert np.array_equal(r["b"], np.array([1, 0, 1], dtype=np.bool_))
        assert float(r["c"]) == 3.5
    print("safetensors roundtrip PASS")
