from .serialize import dask_serialize, dask_deserialize, register_generic
from .cuda import cuda_deserialize, cuda_serialize

import torch
import numpy as np


@cuda_serialize.register(cupy.ndarray)
def cuda_serialize_cupy_ndarray(x):
    # Making sure `x` is behaving
    if not (x.flags["C_CONTIGUOUS"] or x.flags["F_CONTIGUOUS"]):
        x = cupy.array(x, copy=True)

    header = x.__cuda_array_interface__.copy()
    header["strides"] = tuple(x.strides)
    header["lengths"] = [x.nbytes]
    frames = [
        cupy.ndarray(
            shape=(x.nbytes,), dtype=cupy.dtype("u1"), memptr=x.data, strides=(1,)
        )
    ]

    return header, frames


@cuda_deserialize.register(cupy.ndarray)
def cuda_deserialize_cupy_ndarray(header, frames):
    (frame,) = frames
    arr = cupy.ndarray(
        shape=header["shape"],
        dtype=header["typestr"],
        memptr=cupy.asarray(frame).data,
        strides=header["strides"],
    )
    return arr


@dask_serialize.register(torch.Tensor)
def serialize_torch_Tensor(t):
    header = {"shape": tuple(t.shape), "device": t.device.type}

    # Extract type via NumPy
    header["typestr"] = t[:0].cpu().numpy().dtype.str

    # Handle grad
    requires_grad_ = t.requires_grad
    if requires_grad_:
        header, frames = dask_serialize(t.detach().numpy())
    else:
        header, frames = dask_serialize(t.numpy())
    if t.grad is not None:
        grad_header, grad_frames = dask_serialize(t.grad.numpy())
        header["grad"] = {"header": grad_header, "start": len(frames)}
        frames += grad_frames
    header["requires_grad"] = requires_grad_

    # Making sure `t` is contiguous
    header["strides"] = t.stride()
    if t.is_contiguous():
        t = t.flatten()
    elif t.T.is_contiguous():
        t = t.T.flatten()
    else:
        t = t.contiguous()
        header["strides"] = t.stride()
        t = t.flatten()

    return header, frames


@dask_deserialize.register(torch.Tensor)
def deserialize_torch_Tensor(header, frames):
    if header.get("grad", False):
        i = header["grad"]["start"]
        frames, grad_frames = frames[:i], frames[i:]
        grad = dask_deserialize.dispatch(np.ndarray)(
            header["grad"]["header"], grad_frames
        )
    else:
        grad = None

    x = dask_deserialize.dispatch(np.ndarray)(header, frames)
    if header["device"] == "cpu":
        t = torch.from_numpy(x)
        if header["requires_grad"]:
            t = t.requires_grad_(True)
    else:
        t = torch.tensor(
            data=x, device=header["device"], requires_grad=header["requires_grad"]
        )
    if grad is not None:
        t.grad = torch.from_numpy(grad)
    return t


@dask_serialize.register(torch.nn.Parameter)
def serialize_torch_Parameters(p):
    header, frames = dask_serialize(p.detach())
    header["requires_grad"] = p.requires_grad
    return header, frames


@dask_deserialize.register(torch.nn.Parameter)
def deserialize_torch_Parameters(header, frames):
    t = dask_deserialize.dispatch(torch.Tensor)(header, frames)
    return torch.nn.Parameter(data=t, requires_grad=header["requires_grad"])


register_generic(torch.nn.Module)
