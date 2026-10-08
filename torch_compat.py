# -*- coding: utf-8 -*-
"""
torch_compat.py —— 纯 Python 读取 PyTorch .pth 权重文件（无需安装 PyTorch）。

背景
----
PyTorch 的 .pth / .pt 文件是 zip 容器。不同 torch 版本的 zip 内部布局略有差异，
本模块对以下两种都兼容：

  旧布局（torch 1.6 ~ 2.5，RVC 模型的主流格式）:
      archive/data.pkl         pickle 序列化的对象图
      archive/data/<key>       storage 二进制数据（张量底层缓冲，裸字节，little-endian）
      archive/version          协议版本信息（本读取器不依赖）

  新布局（torch 2.6+，以 <保存文件名>/ 为前缀）:
      <basename>/data.pkl
      <basename>/data/<key>
      <basename>/.format_version / .storage_alignment / byteorder / ...

对象图中的 torch 张量以 ``GLOBAL torch._utils._rebuild_tensor_v2`` 的 REDUCE 形式出现，
其 storage 通过 pickle 的 ``persistent_load`` 惰性引用。persistent id 形如：
    ('storage', <storage类型类如 torch.FloatStorage>, <key>, <location>, <numel>)
storage 的 dtype 由 <storage类型类> 决定（torch 内部即如此）。本模块用自定义
``pickle.Unpickler`` 把张量节点还原为 numpy 兼容对象。

对外 API
--------
- ``load_pth(path) -> dict``       等价 torch.load(map_location='cpu')；张量为 TensorArray
                                   （numpy.ndarray 子类，额外带 .numpy()/.item()/.tolist() 等）。
- ``load_pth_lazy(path) -> dict``  同样结构，但张量为 LazyTensor 惰性占位（不读 storage
                                   字节，避免大模型全量读盘）；处理完调用 close_lazy()。
- ``LazyTensor`` / ``TensorView``  .dtype / .shape / .numpy() / .item() / .tolist() ...
- ``materialize(obj)``             把对象图里的惰性张量全部物化为 ndarray。
- ``close_lazy(obj)``              关闭惰性加载持有的 zip 句柄。

说明：load_pth 默认 eager —— 与 torch.load 语义一致（文件读完即可丢弃）。
如果只想读元数据或内存紧张，用 load_pth_lazy。
"""

from __future__ import annotations

import io
import os
import pickle
import struct
import sys
import warnings
import zipfile
from collections import OrderedDict

import numpy as np

__all__ = [
    "load_pth",
    "load_pth_lazy",
    "LazyTensor",
    "TensorView",
    "TensorArray",
    "materialize",
    "close_lazy",
    "TorchCompatError",
    "UnsupportedTorchFeature",
]

# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class TorchCompatError(Exception):
    """读取 .pth 失败（文件不存在、zip 损坏、缺 storage 等）。"""


class UnsupportedTorchFeature(TorchCompatError):
    """遇到本读取器不支持的 torch 序列化特性（稀疏/量化/float8 等）。"""


# ---------------------------------------------------------------------------
# dtype 映射
# ---------------------------------------------------------------------------

# torch storage 类名 -> numpy dtype（persistent_id 的 storage_type 用类名出现）
_STORAGE_CLASS_TO_DTYPE = {
    "torch.ByteStorage": np.dtype("uint8"),
    "torch.CharStorage": np.dtype("int8"),
    "torch.ShortStorage": np.dtype("int16"),
    "torch.IntStorage": np.dtype("int32"),
    "torch.LongStorage": np.dtype("int64"),
    "torch.HalfStorage": np.dtype("float16"),
    "torch.FloatStorage": np.dtype("float32"),
    "torch.DoubleStorage": np.dtype("float64"),
    "torch.BoolStorage": np.dtype("bool"),
    "torch.ComplexFloatStorage": np.dtype("complex64"),
    "torch.ComplexDoubleStorage": np.dtype("complex128"),
    "torch.QUInt8Storage": np.dtype("uint8"),
    "torch.QInt8Storage": np.dtype("int8"),
    "torch.QInt32Storage": np.dtype("int32"),
}
# bfloat16 依赖 numpy 是否提供（部分 numpy 构建没有）
if hasattr(np, "bfloat16"):
    _STORAGE_CLASS_TO_DTYPE["torch.BFloat16Storage"] = np.dtype("bfloat16")

# 使用 untyped storage 序列化的新 dtype（_rebuild_tensor_v3 路径），numpy 不支持
_UNSUPPORTED_NEW_DTYPES = {
    "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
    "float8_e8m0fnu", "float4_e2m1fn_x2", "bits8", "bits16", "bits1x8",
    "bits2x4", "bits4x2", "complex32", "bcomplex32", "uint16", "uint32",
    "uint64",
}
if not hasattr(np, "bfloat16"):
    _UNSUPPORTED_NEW_DTYPES.add("bfloat16")

_DTYPE_NAME_TO_NUMPY = {
    "uint8": np.dtype("uint8"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "bool": np.dtype("bool"),
    "complex64": np.dtype("complex64"),
    "complex128": np.dtype("complex128"),
}
if hasattr(np, "bfloat16"):
    _DTYPE_NAME_TO_NUMPY["bfloat16"] = np.dtype("bfloat16")


def _dtype_from_torch_name(name):
    """把 torch dtype 名（'float32'）或 np.dtype 归一化为 numpy dtype。"""
    if name is None:
        return None
    if isinstance(name, np.dtype):
        return name
    if isinstance(name, str):
        key = name.split(".")[-1]
        dt = _DTYPE_NAME_TO_NUMPY.get(key)
        if dt is not None:
            return dt
        if key in _UNSUPPORTED_NEW_DTYPES:
            raise UnsupportedTorchFeature(
                f"torch dtype '{name}' 是 float8/bits 等新 dtype，"
                f"当前 numpy({np.__version__}) 无法表示"
            )
        raise TorchCompatError(f"未知 torch dtype: {name!r}")
    if hasattr(name, "name"):
        return _dtype_from_torch_name(getattr(name, "name"))
    raise TorchCompatError(f"无法识别 dtype 参数: {name!r}")


def _storage_type_factory(name):
    """构造一个表示 torch storage 类的对象（带 .dtype），供 find_class 返回。

    pickle 流中 persistent_id 元组的第二个元素是 storage 类型＊类＊（GLOBAL
    torch.FloatStorage），此处返回同名同 dtype 的类对象即可。
    """
    dt = _STORAGE_CLASS_TO_DTYPE.get(name)
    if dt is None:
        raise TorchCompatError(f"未知 storage 类型: {name!r}")

    def __init__(self, *a, **k):
        pass

    cls = type(name, (), {
        "dtype": dt,
        "__init__": __init__,
        "__repr__": lambda self: f"<torch_compat {name} dtype={dt}>",
    })
    return cls


def _make_dtype(name):
    """torch.dtype 的 pickle 重建（REDUCE 参数为 dtype 名字符串）。"""
    if name is None:
        return None
    if isinstance(name, np.dtype):
        return name
    try:
        return _dtype_from_torch_name(name)
    except TorchCompatError:
        return str(name)  # 容错：保留名字字符串


# ---------------------------------------------------------------------------
# 惰性 storage：持有 zip 引用与 key，真正读字节时才触达 zip
# ---------------------------------------------------------------------------


class _StorageRef:
    """zip 内一个 storage 的惰性引用（persistent_load 的返回值）。"""

    __slots__ = ("zip", "prefix", "key", "dtype", "location", "numel",
                 "_loaded", "_data")

    def __init__(self, zip_file, prefix, key, dtype, location=None, numel=None):
        self.zip = zip_file
        self.prefix = prefix        # 'archive/' / '<basename>/' / ''
        self.key = key
        self.dtype = dtype          # np.dtype 或 None（未确定/不支持）
        self.location = location    # 'cpu' / 'cuda:0' ...
        self.numel = numel          # persistent_id 中记录的元素个数（可能为 None）
        self._loaded = False
        self._data = None

    def get_bytes(self) -> bytes:
        if self._loaded:
            return self._data
        for name in (f"{self.prefix}data/{self.key}",
                     f"{self.prefix}data/{self.key}/0"):
            try:
                self._data = self.zip.read(name)
                self._loaded = True
                return self._data
            except KeyError:
                continue
        raise TorchCompatError(
            f"zip 内缺少 storage 数据 '{self.prefix}data/{self.key}'"
            f"（及 mmap 变体 '{self.prefix}data/{self.key}/0'）；"
            "文件可能损坏或格式不受支持"
        )

    def __repr__(self):
        return f"<StorageRef key={self.key!r} dtype={self.dtype} zipped={self._loaded}>"


def _close_storage_zip(refs):
    """幂等关闭一组 _StorageRef 共用的 zip 文件句柄。"""
    seen = set()
    for r in refs:
        z = getattr(r, "zip", None)
        if z is not None and id(z) not in seen:
            seen.add(id(z))
            try:
                z.close()
            except Exception:
                pass
            r.zip = None


# ---------------------------------------------------------------------------
# 张量视图（惰性）与 eager 数组（numpy ndarray 子类）
# ---------------------------------------------------------------------------


class TensorView:
    """
    惰性张量视图：元数据（dtype/shape/stride/offset）来自 pickle 对象图，
    数据在调用 ``numpy()`` 时才从 zip 读出。

    提供与 torch.Tensor / numpy 兼容的常用接口：
    numpy()/shape/dtype/item()/tolist()/size()/numel()/dim()/ndim/nbytes/
    __array__/__getitem__/__repr__/cpu()/detach()/to() 等。
    """

    __slots__ = ("_storage", "_offset", "_size", "_stride", "_dtype",
                 "_requires_grad", "_name", "_device")

    def __init__(self, storage, storage_offset, size, stride, dtype,
                 requires_grad=False, name=None, device=None):
        self._storage = storage
        self._offset = int(storage_offset or 0)
        self._size = tuple(int(s) for s in (size or ()))
        self._stride = None if stride is None else tuple(int(s) for s in stride)
        self._dtype = _dtype_from_torch_name(dtype)
        self._requires_grad = bool(requires_grad)
        self._name = name
        self._device = device

    # -- 元数据 -------------------------------------------------------------
    @property
    def shape(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def device(self):
        return self._device

    @property
    def ndim(self):
        return len(self._size)

    @property
    def nbytes(self):
        if self._dtype is None:
            raise UnsupportedTorchFeature("该张量 dtype 不受支持，无法计算 nbytes")
        return self.numel() * self._dtype.itemsize

    def size(self, dim=None):
        """torch 风格：size() -> shape tuple；size(dim) -> 该维长度。"""
        return self._size if dim is None else self._size[dim]

    def numel(self):
        n = 1
        for s in self._size:
            n *= s
        return n

    def dim(self):
        return len(self._size)

    # -- 物化 ---------------------------------------------------------------
    def numpy(self) -> np.ndarray:
        """真正从 zip 读取 storage 字节并返回 numpy 数组（只读视图）。"""
        if self._dtype is None:
            raise UnsupportedTorchFeature(
                "该张量 dtype 不受 numpy 支持，无法物化"
            )
        if self._storage is None:
            raise TorchCompatError("该张量没有绑定 storage 源，无法物化")
        raw = self._storage.get_bytes()
        dtype_le = self._dtype.newbyteorder("<")
        base = np.frombuffer(raw, dtype=dtype_le, offset=0)

        offset = self._offset
        end = offset + self.numel()
        if end > len(base):
            raise TorchCompatError(
                f"storage '{self._storage.key}' 数据不足：需要元素 {offset}..{end}，"
                f"实际仅 {len(base)} 个（dtype={self._dtype}）"
            )
        base = base[offset:end]

        if not self._size:
            return base.reshape(())

        if self._stride is None:
            return base.reshape(self._size)

        strides = tuple(s * dtype_le.itemsize for s in self._stride)
        try:
            return np.lib.stride_tricks.as_strided(
                base, shape=self._size, strides=strides
            )
        except ValueError as e:
            raise TorchCompatError(
                f"无法按 stride={self._stride} 构建张量视图: {e}"
            ) from e

    # -- numpy 兼容 ---------------------------------------------------------
    def __array__(self, dtype=None, copy=None):
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def item(self):
        arr = self.numpy()
        if arr.size != 1:
            raise ValueError(
                f"can only convert an array of size 1 to a Python scalar, "
                f"got size {arr.size}"
            )
        return arr.item()

    def tolist(self):
        return self.numpy().tolist()

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def __getitem__(self, key):
        return self.numpy()[key]

    def __len__(self):
        if self._size:
            return self._size[0]
        raise TypeError("len() of a 0-d tensor")

    def __iter__(self):
        return iter(self.numpy())

    def __float__(self):
        return float(self.item())

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    def __repr__(self):
        return (
            f"TensorView(dtype={self._dtype}, shape={self._size}, "
            f"storage_key={getattr(self._storage, 'key', None)!r}, "
            f"offset={self._offset}, requires_grad={self._requires_grad})"
        )


class LazyTensor(TensorView):
    """load_pth_lazy 返回的惰性张量占位。

    常规构造沿用 TensorView；也支持占位式三参构造 ``LazyTensor(key, dtype, shape)``
    （此时没有 zip 源，仅供元数据场景）。
    """

    __slots__ = ("_lazy_key",)

    def __init__(self, *args, **kwargs):
        if len(args) >= 3 and not isinstance(args[0], _StorageRef) \
                and not isinstance(args[0], _InlineStorage):
            key, dtype, shape = args[0], args[1], args[2]
            self._storage = None
            self._offset = 0
            self._size = tuple(int(s) for s in shape)
            self._stride = None
            self._dtype = _dtype_from_torch_name(dtype)
            self._requires_grad = False
            self._name = None
            self._device = None
            self._lazy_key = key
        else:
            super().__init__(*args, **kwargs)
            self._lazy_key = getattr(self._storage, "key", None)

    @property
    def key(self):
        return self._lazy_key

    def numpy(self):
        if self._storage is None:
            raise TorchCompatError(
                f"LazyTensor(key={self._lazy_key}) 未绑定 zip 源，仅能读取元数据"
            )
        return super().numpy()

    def close(self):
        """释放该张量引用的 zip 文件句柄。"""
        _close_storage_zip([self._storage])


class TensorArray(np.ndarray):
    """load_pth(eager) 返回的张量类型：numpy.ndarray 子类，兼容 torch 常用接口。

    注意：torch 风格的 ``t.size()`` 与 numpy 的 ``t.size``（属性）二选一，
    这里保留 numpy 语义（.size 属性 = 元素个数）；可以用 ``numel()`` 代替。
    """

    requires_grad = False

    def __array_finalize__(self, obj):
        if obj is not None:
            self.requires_grad = getattr(obj, "requires_grad", False)

    def numpy(self):
        return self

    def numel(self):
        return self.size

    def dim(self):
        return self.ndim

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def __repr__(self):
        return f"TensorArray(dtype={self.dtype}, shape={self.shape})\n" + \
            np.ndarray.__repr__(self)


def _view_as_tensorarray(arr, requires_grad=False):
    """把 ndarray 转为 TensorArray 视图。"""
    if isinstance(arr, TensorArray):
        arr.requires_grad = requires_grad or arr.requires_grad
        return arr
    try:
        ta = arr.view(TensorArray)
        ta.requires_grad = requires_grad
        return ta
    except Exception:
        return np.asarray(arr)


# ---------------------------------------------------------------------------
# 兼容对象：device / 未知值
# ---------------------------------------------------------------------------


class _DeviceCompat:
    """torch.device 的轻量替代（本实现忽略设备信息）。"""

    __slots__ = ("type", "index")

    def __init__(self, type_, index=None):
        self.type = str(type_)
        self.index = index

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self):
        return f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        if isinstance(other, _DeviceCompat):
            return (self.type, self.index) == (other.type, other.index)
        return str(self) == str(other)

    def __hash__(self):
        return hash((self.type, self.index))


class _UnknownValue:
    """未知 torch 属性的容错哨兵：可调用、可比较、可打印、可访问属性。"""

    __slots__ = ("qualname",)

    def __init__(self, qualname):
        self.qualname = qualname

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, item):
        return _UnknownValue(f"{self.qualname}.{item}")

    def __repr__(self):
        return f"<torch_compat unknown: {self.qualname}>"

    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, _UnknownValue) and self.qualname == other.qualname

    def __hash__(self):
        return hash(self.qualname)


# ---------------------------------------------------------------------------
# 重建函数（find_class 的返回目标）
# ---------------------------------------------------------------------------


# 当前解包使用的张量类（eager=TensorView；lazy=LazyTensor）
# 单线程场景可用模块级变量；load_pth_lazy 在解析期间临时切换。
_tensor_cls = TensorView  # noqa: E305  （定义在下方类之后赋值）


def _make_tensor(*args, **_kwargs):
    """真正构造张量视图（供 _rebuild_tensor_v2 等调用）。"""
    storage, storage_offset, size, stride, requires_grad = args[0], args[1], args[2], args[3], args[4]
    if not isinstance(storage, (_StorageRef, _InlineStorage, _ArrayStorageRef)):
        raise TorchCompatError(
            f"_rebuild_tensor_v2 收到无法识别的 storage 参数: "
            f"{type(storage).__name__}: {storage!r}"
        )
    rest = args[5:]
    # backward_hooks = rest[0]（忽略）；之后可能出现第 7 参：
    #   - 新版本 torch：metadata（dict，conj/neg 标记）-> 忽略
    #   - 旧版本 torch（1.12~1.13）：dtype -> 用作 dtype
    explicit_dtype = None
    if len(rest) >= 2:
        candidate = rest[1]
        if isinstance(candidate, dict):
            pass  # metadata
        else:
            explicit_dtype = candidate

    dtype = explicit_dtype
    if dtype is None and getattr(storage, "dtype", None) is not None:
        dtype = storage.dtype
    if dtype is None:
        # 旧格式且无法推断：默认 float32（RVC 权重以 float32 为主），并警告
        dtype = "float32"
        warnings.warn(
            f"张量 storage='{getattr(storage, 'key', '?')}' 的 dtype 无法从文件中确定，"
            "按 float32 处理",
            RuntimeWarning, stacklevel=2,
        )
    return _tensor_cls(storage, storage_offset, size, stride, dtype, requires_grad)


def _rebuild_tensor_v2(*args):
    return _make_tensor(*args)


def _rebuild_tensor_v3(*args):
    """(storage, offset, size, stride, rg, hooks, dtype, metadata=None)。"""
    if len(args) >= 7:
        dtype_arg = args[6]
    else:
        dtype_arg = None
    storage, offset, size, stride, rg = args[0], args[1], args[2], args[3], args[4]
    if dtype_arg is None and getattr(storage, "dtype", None) is not None:
        dtype_arg = storage.dtype
    if dtype_arg is None:
        dtype_arg = "float32"
    return TensorView(storage, offset, size, stride, dtype_arg, rg)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    """torch._utils._rebuild_parameter：nn.Parameter 的 pickle 重建。"""
    if isinstance(data, TensorView):
        return _ParameterView(data._storage, data._offset, data._size,
                              data._stride, data._dtype, requires_grad)
    arr = np.ascontiguousarray(data)
    return TensorView(_ArrayStorageRef(arr), 0, arr.shape, None, arr.dtype,
                      requires_grad)


def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    return _rebuild_parameter(data, requires_grad, backward_hooks)


def _rebuild_device_tensor_from_numpy(data, dtype, device=None, requires_grad=False):
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(_dtype_from_torch_name(dtype), copy=False)
    return TensorView(_ArrayStorageRef(arr), 0, arr.shape, None, arr.dtype,
                      requires_grad)


class _ArrayStorageRef:
    """把已物化的 numpy 数组伪装成 storage 占位。"""

    __slots__ = ("_arr", "key", "dtype", "location", "numel")

    def __init__(self, arr):
        self._arr = np.ascontiguousarray(arr)
        self.key = "<inline-array>"
        self.dtype = self._arr.dtype
        self.location = "cpu"
        self.numel = self._arr.size

    def get_bytes(self):
        return self._arr.tobytes()


class _ParameterView(TensorView):
    __slots__ = ()

    def __repr__(self):
        return f"ParameterView(dtype={self._dtype}, shape={self._size})"


class _InlineStorage:
    """_load_from_bytes 的产物：已物化的内联 storage。"""

    __slots__ = ("data", "numel", "dtype", "device", "key", "location")

    def __init__(self, data, size, dtype, device):
        self.data = data
        self.numel = size
        self.dtype = dtype
        self.device = device
        self.key = "<inline-bytes>"
        self.location = device

    def get_bytes(self):
        return self.data


def _load_from_bytes(b):
    """对应 torch.storage._load_from_bytes，解析内联 storage 字节。

    torch 2.6+ 中该函数等价于 ``torch.load(io.BytesIO(b))``（b 是内嵌的完整
    序列化文件）；更早版本中 b 是"未序列化 storage"：前 8 字节 int64 元素个数 +
    元数据长度 + 元数据文本 + 原始元素数据。此处两种都尽力兼容；该路径在标准
    .pth 权重文件中极少出现（RVC 权重不会走这里）。
    """
    if not isinstance(b, (bytes, bytearray, memoryview)):
        raise TorchCompatError(f"_load_from_bytes 需要 bytes，收到 {type(b)}")
    buf = bytes(b)
    if not buf:
        raise TorchCompatError("_load_from_bytes 收到空字节流")

    # 路径一：b 是完整 zip（新版 torch）
    if buf[:2] == b"PK":
        try:
            inner = _read_zip(buf)
        except TorchCompatError as e:
            raise TorchCompatError(f"_load_from_bytes 内嵌 zip 解析失败: {e}") from e
        # 期望顶层是 storage 引用；否则取对象图里第一个 storage
        if isinstance(inner, (_StorageRef, _InlineStorage)):
            return inner
        refs = _collect_storage_refs(inner)
        if refs:
            return list(refs)[0]
        raise TorchCompatError("_load_from_bytes 内嵌 zip 中没有 storage")

    # 路径二：旧式 header 布局
    #   [0:8]  int64 size（元素个数）
    #   [8:12] int32 元数据长度（老 torch 也有 8 字节长度变体，下面双重尝试）
    if len(buf) < 12:
        raise TorchCompatError(f"_load_from_bytes 字节流过短: {len(buf)}B")
    size = struct.unpack("<q", buf[0:8])[0]
    meta_len = struct.unpack("<i", buf[8:12])[0]
    if meta_len < 0 or 12 + meta_len > len(buf):
        meta_len = struct.unpack("<q", buf[8:16])[0]
        meta_start = 16
    else:
        meta_start = 12
    if meta_len < 0 or meta_start + meta_len > len(buf):
        raise TorchCompatError(f"_load_from_bytes 元数据长度非法: {meta_len}")
    meta = buf[meta_start:meta_start + meta_len]
    data = buf[meta_start + meta_len:]

    dtype = None
    location = "cpu"
    try:
        meta_text = meta.decode("utf-8", errors="replace")
    except Exception:
        meta_text = ""
    for tok in meta_text.replace("torch.", "").replace("'", "").split():
        if tok in _DTYPE_NAME_TO_NUMPY:
            dtype = _dtype_from_torch_name(tok)
            break
        if tok.endswith("Storage"):
            dt = _STORAGE_CLASS_TO_DTYPE.get("torch." + tok)
            if dt is not None:
                dtype = dt
                break
    if "cuda" in meta_text:
        location = "cuda"

    if dtype is not None:
        need = abs(size) * dtype.itemsize
        if len(data) < need:
            raise TorchCompatError(
                f"_load_from_bytes 数据不足：声明 {abs(size)} 个元素({need}B)，"
                f"实际 {len(data)}B"
            )
    return _InlineStorage(data, abs(size), dtype, location)


# ---------------------------------------------------------------------------
# 自定义 Unpickler
# ---------------------------------------------------------------------------

_STORAGE_CLASS_NAMES = frozenset(_STORAGE_CLASS_TO_DTYPE)


class _TorchUnpickler(pickle.Unpickler):
    """把 torch 张量/存储重建为 numpy 兼容对象。"""

    def __init__(self, file, zip_file, zip_prefix="", **kwargs):
        super().__init__(file, **kwargs)
        self._zip = zip_file
        self._zip_prefix = zip_prefix

    # -- persistent_load ----------------------------------------------------
    def persistent_load(self, pid):
        """
        torch 序列化 persistent_id 形式：
          新格式: ('storage', <storage类型类>, key, location, numel)
          旧格式: ('storage', <storage类型类>, root_key, location, numel, view_metadata)
          更旧 (torch<=1.5): 字符串 '<key> storage' / '<key>:storage' / 裸 key
        返回惰性 storage 占位（_StorageRef）。
        """
        key = None
        dtype = None
        location = None
        numel = None
        if isinstance(pid, tuple):
            if len(pid) >= 2 and pid[0] == "storage":
                # pid[1] 是 storage 类型类（find_class('torch','FloatStorage') 的产物）
                st = pid[1]
                if hasattr(st, "dtype"):
                    dtype = st.dtype
                if len(pid) >= 3:
                    key = pid[2]
                    if len(pid) >= 4:
                        location = pid[3]
                    if len(pid) >= 5:
                        numel = pid[4]
                else:
                    key = pid[1]
            else:
                for i, tok in enumerate(pid):
                    if tok == "storage" and i + 1 < len(pid):
                        key = pid[i + 1]
                        if i + 2 < len(pid):
                            st = pid[i + 2]
                            if hasattr(st, "dtype"):
                                dtype = st.dtype
                                location = pid[i + 3] if i + 3 < len(pid) else None
                                numel = pid[i + 4] if i + 4 < len(pid) else None
                        break
                if key is None and pid:
                    # 兜底：取最后一个像 key 的字符串
                    for tok in reversed(pid):
                        if isinstance(tok, (str, bytes)):
                            key = tok
                            break
        elif isinstance(pid, str):
            s = pid.strip()
            if s.endswith(" storage"):
                key = s[:-len(" storage")]
            elif ":storage" in s:
                key = s.split(":storage", 1)[0]
            elif s.endswith(":"):
                key = s[:-1]
            else:
                key = s
        if key is None:
            raise TorchCompatError(f"无法识别的 persistent id: {pid!r}")
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="replace")
        key = str(key)
        return _StorageRef(self._zip, self._zip_prefix, key, dtype, location, numel)

    # -- find_class ---------------------------------------------------------
    def find_class(self, module, name):
        qual = f"{module}.{name}"

        # ---- numpy：放行真实对象（numpy 有自己的 pickle 协议） ----
        if module == "numpy.core.multiarray" or module.startswith("numpy."):
            try:
                return super().find_class(module, name)
            except Exception:
                return _UnknownValue(qual)

        # ---- torch 相关映射 ----
        if qual in ("torch._utils._rebuild_tensor_v2",):
            return _rebuild_tensor_v2
        if qual in ("torch._utils._rebuild_tensor_v3",):
            return _rebuild_tensor_v3
        if qual == "torch._utils._rebuild_tensor":
            return _rebuild_tensor_v2  # 参数少，_make_tensor 用 *args 兼容
        if qual in ("torch._utils._rebuild_parameter",
                    "torch._utils._rebuild_parameter_v2",
                    "torch._utils._rebuild_parameter_with_state"):
            return {
                "torch._utils._rebuild_parameter": _rebuild_parameter,
                "torch._utils._rebuild_parameter_v2": _rebuild_parameter,
                "torch._utils._rebuild_parameter_with_state": _rebuild_parameter_with_state,
            }[qual]
        if qual == "torch._utils._rebuild_device_tensor_from_numpy":
            return _rebuild_device_tensor_from_numpy
        if qual in ("torch._utils._rebuild_sparse_tensor",
                    "torch._utils._rebuild_sparse_tensor_v2",
                    "torch._utils._rebuild_sparse_csr_tensor",
                    "torch._utils._rebuild_qtensor",
                    "torch._utils._rebuild_nested_tensor",
                    "torch._utils._rebuild_wrapper_subclass",
                    "torch._utils._rebuild_meta_tensor_no_storage"):
            raise UnsupportedTorchFeature(f"不支持的 torch 张量类型: {qual}")
        if qual == "torch.storage._load_from_bytes":
            return _load_from_bytes
        if qual in _STORAGE_CLASS_NAMES:
            return _storage_type_factory(qual)
        # untyped storage 的各种名字
        if qual in ("torch.UntypedStorage", "torch.storage.UntypedStorage",
                    "torch._UntypedStorage", "torch.storage._UntypedStorage",
                    "torch.storage.TypedStorage", "torch.TypedStorage"):
            return _storage_type_factory("torch.ByteStorage")
        if qual in ("torch.Size", "torch._C.Size"):
            return _make_size
        if qual == "torch.device":
            return _make_device
        if qual == "torch.dtype":
            return _make_dtype
        if qual in ("torch.nn.Parameter", "torch.Tensor",
                    "torch._C._TensorBase", "torch._C.TensorBase"):
            # 直接出现在流里无法还原存储关系，返回哨兵并提醒
            warnings.warn(
                f"pickle 流中直接引用了 {qual}（非标准张量保存路径），将被占位对象替代",
                RuntimeWarning, stacklevel=2,
            )
            return _UnknownValue(qual)

        # ---- torch 命名空间下其它名字：容错 ----
        if module == "torch" or module.startswith("torch."):
            if name in _DTYPE_NAME_TO_NUMPY or name in _UNSUPPORTED_NEW_DTYPES:
                return _make_dtype(name)
            return _UnknownValue(qual)

        # ---- 标准库 / 其它：正常 import ----
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            raise TorchCompatError(
                f"pickle 引用了无法解析的全局: {qual}（{e}）"
            ) from e


def _make_size(*args):
    """torch.Size 的 pickle 重建（返回 tuple）。"""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return tuple(args)


def _make_device(type_, index=None):
    """torch.device 的 pickle 重建。"""
    if isinstance(type_, _DeviceCompat):
        return type_
    return _DeviceCompat(type_, index)


# ---------------------------------------------------------------------------
# zip 读取
# ---------------------------------------------------------------------------


def _open_zip(path):
    try:
        return zipfile.ZipFile(path, "r")
    except FileNotFoundError:
        raise TorchCompatError(f"文件不存在: {path}") from None
    except zipfile.BadZipFile as e:
        raise TorchCompatError(f"不是有效的 zip/.pth 文件: {path}（{e}）") from None


def _find_data_pkl(zf):
    """在 zip 中定位 data.pkl 并返回其目录前缀（'' 表示无前缀）。"""
    names = zf.namelist()
    candidates = []
    for n in names:
        if n == "data.pkl" or (n.endswith("/data.pkl") and "/" in n):
            candidates.append(n)
    if not candidates:
        return None, None
    # 优先精确且不带深层嵌套的（archive/ 或 <basename>/）；取最短前缀
    candidates.sort(key=len)
    chosen = candidates[0]
    prefix = chosen[: -len("data.pkl")]
    return chosen, prefix


def _unpickle_zip(zf, data_name, prefix):
    raw = zf.read(data_name)
    unpickler = _TorchUnpickler(io.BytesIO(raw), zf, zip_prefix=prefix,
                                encoding="utf-8")
    try:
        return unpickler.load()
    except TorchCompatError:
        raise
    except Exception as e:
        raise TorchCompatError(
            f"解析 {data_name} 失败: {type(e).__name__}: {e}"
        ) from e


def _read_zip(data_or_path, is_path=True):
    """打开 zip，返回对象图（张量为 TensorView 惰性）。"""
    if is_path:
        zf = _open_zip(data_or_path)
    else:
        zf = zipfile.ZipFile(io.BytesIO(data_or_path), "r")
    data_name, prefix = _find_data_pkl(zf)
    if data_name is None:
        zf.close()
        raise TorchCompatError(
            "zip 内缺少对象图 'data.pkl'（或 archive/data.pkl）；"
            "该文件可能不是 PyTorch 1.6+ 的 .pth 格式"
        )
    try:
        obj = _unpickle_zip(zf, data_name, prefix)
        return obj
    except Exception:
        zf.close()
        raise


def _collect_storage_refs(obj):
    """递归收集对象图中的全部 _StorageRef（去重）。"""
    out = []
    seen = set()

    def visit(o):
        if isinstance(o, _StorageRef):
            if id(o) not in seen:
                seen.add(id(o))
                out.append(o)
        elif isinstance(o, TensorView):
            visit(o._storage)
        elif isinstance(o, dict):
            for v in o.values():
                visit(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                visit(v)
    visit(obj)
    return out


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def load_pth(path):
    """读取 .pth，返回与 torch.load(map_location='cpu') 等价的对象。

    张量为 ``TensorArray``（numpy.ndarray 子类）：与 ndarray 完全互通，
    另带 .numpy()/.item()/.tolist()/.numel()/.dim() 等兼容接口。
    eager 模式：读完即关闭文件，与 torch.load 语义一致。
    """
    zf = _open_zip(os.fspath(path))
    data_name, prefix = _find_data_pkl(zf)
    if data_name is None:
        zf.close()
        raise TorchCompatError(
            "zip 内缺少对象图 'data.pkl'（或 archive/data.pkl）；"
            "该文件可能不是 PyTorch 1.6+ 的 .pth 格式"
        )
    try:
        obj = _unpickle_zip(zf, data_name, prefix)
        result = materialize(obj)
        return result
    finally:
        zf.close()


def load_pth_lazy(path):
    """读取 .pth，返回同样结构但张量为 LazyTensor 惰性占位。

    不读取 storage 字节（仅解析对象图元数据），适合大模型/只读元数据场景。
    注意：文件句柄会保持打开，处理完后请调用 ``close_lazy(result)`` 释放。
    """
    obj = _read_zip(os.fspath(path), is_path=True)
    return obj


def materialize(obj):
    """递归把对象图中的 TensorView/LazyTensor 物化为 numpy ndarray（TensorArray）。"""
    if isinstance(obj, TensorView):
        src = obj.numpy()
        return _view_as_tensorarray(src, getattr(obj, "_requires_grad", False))
    if isinstance(obj, dict):
        return {k: materialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(materialize(v) for v in obj)
    if isinstance(obj, _StorageRef):
        return obj
    return obj


def _materialize(obj):
    return materialize(obj)


def close_lazy(obj):
    """释放 load_pth_lazy 结果占用的 zip 文件句柄（幂等，可对任意结构调用）。"""
    refs = _collect_storage_refs(obj)
    _close_storage_zip(refs)


# ---------------------------------------------------------------------------
# 便捷 dict 工具（对齐 RVC 常用调用）
# ---------------------------------------------------------------------------


def get_weight_dict(obj):
    """从 checkpoint 顶层 dict 取出状态字典（兼容 'weight' 键形态）。"""
    if isinstance(obj, dict):
        if "weight" in obj:
            return obj["weight"]
        # 一些模型把整个 state_dict 作为顶层
        return obj
    raise TorchCompatError("checkpoint 顶层不是 dict")