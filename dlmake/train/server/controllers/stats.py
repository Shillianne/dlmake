from ..models import (
    UpdateRequestModel,
    FloatResponseModel,
    IntegerResponseModel,
)
from fastapi import APIRouter, Request, HTTPException
from asyncio import Queue
from ..state import *
import torch
import io


router = APIRouter()


@router.post(
    "/update",
    status_code=200,
    summary="Updates the value of the given stats",
    description="""
    Enqueues the value given in the request to separate queues.
    """
)
async def update(request: UpdateRequestModel):
    if request.loss is not None:
        await loss.put(request.loss)
    if request.acc is not None:
        await acc.put(request.acc)
    if request.test_loss is not None:
        await test_loss.put(request.test_loss)
    if request.test_acc is not None:
        await test_acc.put(request.test_acc)
    if request.val_loss is not None:
        await val_loss.put(request.val_loss)
    if request.val_acc is not None:
        await val_acc.put(request.val_acc)
    if request.norm is not None:
        await norm.put(request.norm)
    if request.tkps is not None:
        await tkps.put(request.tkps)
    if request.mfu is not None:
        await mfu.put(request.mfu)
    if request.vram_usage is not None:
        await vram_usage.put(request.vram_usage)

@router.post(
    "/loop",
    status_code=200,
    summary="Updates the values of the looping model's exit probabilities",
    description="""
    Enqueues the exit probabilities of the model.
    """
)
async def update_exit_probs(request: Request):
    try:
        body_bytes = await request.body()
        buffer = io.BytesIO(body_bytes)
        unnormalized = torch.load(buffer, weights_only=True)
        for i, u in enumerate(unnormalized):
            if len(loop_exit_probs) != unnormalized.shape[0]:
                q: Queue[torch.Tensor] = Queue(maxsize=128)
                await q.put(u)
                loop_exit_probs.extend([q])
            else:
                await loop_exit_probs[i].put(u)
    except Exception as e:
        msg = str(e)
        raise HTTPException(status_code=500, detail=msg)

@router.get(
    "/loss",
    response_model=FloatResponseModel,
)
async def get_loss():
    if not loss.empty():
        return await loss.get()
    return None

@router.get(
    "/acc",
    response_model=FloatResponseModel,
)
async def get_acc():
    if not acc.empty():
        return await acc.get()
    return None

@router.get(
    "/test_loss",
    response_model=FloatResponseModel,
)
async def get_test_loss():
    if not test_loss.empty():
        return await test_loss.get()
    return None

@router.get(
    "/test_acc",
    response_model=FloatResponseModel,
)
async def get_test_acc():
    if not test_acc.empty():
        return await test_acc.get()
    return None

@router.get(
    "/val_loss",
    response_model=FloatResponseModel,
)
async def get_val_loss():
    if not val_loss.empty():
        return await val_loss.get()
    return None

@router.get(
    "/val_acc",
    response_model=FloatResponseModel,
)
async def get_val_acc():
    if not val_acc.empty():
        return await val_acc.get()
    return None

@router.get(
    "/norm",
    response_model=FloatResponseModel,
)
async def get_norm():
    if not norm.empty():
        return await norm.get()
    return None

@router.get(
    "/tkps",
    response_model=FloatResponseModel,
)
async def get_tkps():
    if not tkps.empty():
        return await tkps.get()
    return None

@router.get(
    "/mfu",
    response_model=FloatResponseModel,
)
async def get_mfu():
    if not mfu.empty():
        return await mfu.get()
    return None

@router.get(
    "/vram",
    response_model=IntegerResponseModel,
)
async def get_vram():
    if not vram_usage.empty():
        return await vram_usage.get()
    return None

@router.get(
    "/loop",
    response_model=FloatResponseModel,
)
async def get_loop_exit_probs():
    present = True
    for q in loop_exit_probs:
        present = present and not q.empty()
    if present:
        return [await q.get() for q in loop_exit_probs]
    return None
