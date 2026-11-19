from ..models import (
    UpdateMetricsRequestModel,
    FloatResponseModel,
    IntegerResponseModel,
    SeqLenResponseModel,
)
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from asyncio import Queue
from ..state import *
import pickle
import torch
import io


router = APIRouter()


@router.post(
    "/metrics/update",
    status_code=200,
    summary="Updates the value of the given stats",
    description="""
    Enqueues the value given in the request to separate queues.
    """
)
async def update(request: UpdateMetricsRequestModel):
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
    if request.gpu_usage is not None:
        await gpu_usage.put(request.gpu_usage)
    if request.epoch is not None:
        await epoch.put(request.epoch)
    if request.eta is not None:
        await eta.put(request.eta)
    if request.lr is not None:
        await lr.put(request.lr)
    if request.train_time is not None:
        await train_time.put(request.train_time)
    if request.test_time is not None:
        await test_time.put(request.test_time)
    if request.val_time is not None:
        await val_time.put(request.val_time)
    if request.benchmark_time is not None:
        await benchmark_time.put(request.benchmark_time)

    if request.train_precision is not None:
        await train_precision.put(request.train_precision)
    if request.train_recall is not None:
        await train_recall.put(request.train_recall)
    if request.train_specificity is not None:
        await train_specificity.put(request.train_specificity)
    if request.train_f1score is not None:
        await train_f1score.put(request.train_f1score)
    if request.train_perplexity is not None:
        await train_perplexity.put(request.train_perplexity)
    if request.train_calibration_error is not None:
        await train_calibration_error.put(request.train_calibration_error)

    if request.test_precision is not None:
        await test_precision.put(request.test_precision)
    if request.test_recall is not None:
        await test_recall.put(request.test_recall)
    if request.test_specificity is not None:
        await test_specificity.put(request.test_specificity)
    if request.test_f1score is not None:
        await test_f1score.put(request.test_f1score)
    if request.test_perplexity is not None:
        await test_perplexity.put(request.test_perplexity)
    if request.test_calibration_error is not None:
        await test_calibration_error.put(request.test_calibration_error)

    if request.val_precision is not None:
        await val_precision.put(request.val_precision)
    if request.val_recall is not None:
        await val_recall.put(request.val_recall)
    if request.val_specificity is not None:
        await val_specificity.put(request.val_specificity)
    if request.val_f1score is not None:
        await val_f1score.put(request.val_f1score)
    if request.val_perplexity is not None:
        await val_perplexity.put(request.val_perplexity)
    if request.val_calibration_error is not None:
        await val_calibration_error.put(request.val_calibration_error)

    if request.test_top5_acc is not None:
        await test_top5_acc.put(request.test_top5_acc)
    if request.test_top10_acc is not None:
        await test_top10_acc.put(request.test_top10_acc)
    if request.test_top50_acc is not None:
        await test_top50_acc.put(request.test_top50_acc)
    if request.val_top5_acc is not None:
        await val_top5_acc.put(request.val_top5_acc)
    if request.val_top10_acc is not None:
        await val_top10_acc.put(request.val_top10_acc)
    if request.val_top50_acc is not None:
        await val_top50_acc.put(request.val_top50_acc)

    if request.test_seqlen_loss is not None:
        for i, q in enumerate(test_seqlen_loss):
            await q.put(request.test_seqlen_loss[i])
    if request.val_seqlen_loss is not None:
        for i, q in enumerate(val_seqlen_loss):
            await q.put(request.val_seqlen_loss[i])

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
    "/metrics/loss",
    response_model=FloatResponseModel,
)
async def get_loss():
    if not loss.empty():
        return await loss.get()
    return None

@router.get(
    "/metrics/acc",
    response_model=FloatResponseModel,
)
async def get_acc():
    if not acc.empty():
        return await acc.get()
    return None

@router.get(
    "/metrics/test_loss",
    response_model=FloatResponseModel,
)
async def get_test_loss():
    if not test_loss.empty():
        return await test_loss.get()
    return None

@router.get(
    "/metrics/test_acc",
    response_model=FloatResponseModel,
)
async def get_test_acc():
    if not test_acc.empty():
        return await test_acc.get()
    return None

@router.get(
    "/metrics/val_loss",
    response_model=FloatResponseModel,
)
async def get_val_loss():
    if not val_loss.empty():
        return await val_loss.get()
    return None

@router.get(
    "/metrics/val_acc",
    response_model=FloatResponseModel,
)
async def get_val_acc():
    if not val_acc.empty():
        return await val_acc.get()
    return None

@router.get(
    "/metrics/norm",
    response_model=FloatResponseModel,
)
async def get_norm():
    if not norm.empty():
        return await norm.get()
    return None

@router.get(
    "/metrics/tkps",
    response_model=FloatResponseModel,
)
async def get_tkps():
    if not tkps.empty():
        return await tkps.get()
    return None

@router.get(
    "/metrics/mfu",
    response_model=FloatResponseModel,
)
async def get_mfu():
    if not mfu.empty():
        return await mfu.get()
    return None

@router.get(
    "/metrics/vram",
    response_model=IntegerResponseModel,
)
async def get_vram():
    if not vram_usage.empty():
        return await vram_usage.get()
    return None

@router.get(
    "/metrics/usage",
    response_model=IntegerResponseModel,
)
async def get_usage():
    if not gpu_usage.empty():
        return await gpu_usage.get()
    return None

@router.get(
    "/metrics/epoch",
    response_model=IntegerResponseModel,
)
async def get_epoch():
    if not epoch.empty():
        return await epoch.get()
    return None

@router.get(
    "/metrics/eta",
    response_model=FloatResponseModel,
)
async def get_eta():
    if not eta.empty():
        return await eta.get()
    return None

@router.get(
    "/metrics/lr",
    response_model=FloatResponseModel,
)
async def get_lr():
    if not lr.empty():
        return await lr.get()
    return None

@router.get(
    "/metrics/time",
    response_model=FloatResponseModel,
)
async def get_time():
    if not time.empty():
        return await time.get()
    return None

@router.get(
    "/metrics/precision",
    response_model=FloatResponseModel,
)
async def get_precision():
    if not train_precision.empty():
        return await train_precision.get()
    return None

@router.get(
    "/metrics/recall",
    response_model=FloatResponseModel,
)
async def get_recall():
    if not train_recall.empty():
        return await train_recall.get()
    return None

@router.get(
    "/metrics/specificity",
    response_model=FloatResponseModel,
)
async def get_specificity():
    if not train_specificity.empty():
        return await train_specificity.get()
    return None

@router.get(
    "/metrics/f1score",
    response_model=FloatResponseModel,
)
async def get_f1score():
    if not train_f1score.empty():
        return await train_f1score.get()
    return None

@router.get(
    "/metrics/perplexity",
    response_model=FloatResponseModel,
)
async def get_perplexity():
    if not train_perplexity.empty():
        return await train_perplexity.get()
    return None

@router.get(
    "/metrics/calibration_error",
    response_model=FloatResponseModel,
)
async def get_calibration_error():
    if not train_calibration_error.empty():
        return await train_calibration_error.get()
    return None

@router.get(
    "/metrics/test_precision",
    response_model=FloatResponseModel,
)
async def get_test_precision():
    if not test_precision.empty():
        return await test_precision.get()
    return None

@router.get(
    "/metrics/test_recall",
    response_model=FloatResponseModel,
)
async def get_test_recall():
    if not test_recall.empty():
        return await test_recall.get()
    return None

@router.get(
    "/metrics/test_specificity",
    response_model=FloatResponseModel,
)
async def get_test_specificity():
    if not test_specificity.empty():
        return await test_specificity.get()
    return None

@router.get(
    "/metrics/test_f1score",
    response_model=FloatResponseModel,
)
async def get_test_f1score():
    if not test_f1score.empty():
        return await test_f1score.get()
    return None

@router.get(
    "/metrics/test_perplexity",
    response_model=FloatResponseModel,
)
async def get_test_perplexity():
    if not test_perplexity.empty():
        return await test_perplexity.get()
    return None

@router.get(
    "/metrics/test_calibration_error",
    response_model=FloatResponseModel,
)
async def get_test_calibration_error():
    if not test_calibration_error.empty():
        return await test_calibration_error.get()
    return None

@router.get(
    "/metrics/val_precision",
    response_model=FloatResponseModel,
)
async def get_val_precision():
    if not val_precision.empty():
        return await val_precision.get()
    return None

@router.get(
    "/metrics/val_recall",
    response_model=FloatResponseModel,
)
async def get_val_recall():
    if not val_recall.empty():
        return await val_recall.get()
    return None

@router.get(
    "/metrics/val_specificity",
    response_model=FloatResponseModel,
)
async def get_val_specificity():
    if not val_specificity.empty():
        return await val_specificity.get()
    return None

@router.get(
    "/metrics/val_f1score",
    response_model=FloatResponseModel,
)
async def get_val_f1score():
    if not val_f1score.empty():
        return await val_f1score.get()
    return None

@router.get(
    "/metrics/val_perplexity",
    response_model=FloatResponseModel,
)
async def get_val_perplexity():
    if not val_perplexity.empty():
        return await val_perplexity.get()
    return None

@router.get(
    "/metrics/val_calibration_error",
    response_model=FloatResponseModel,
)
async def get_val_calibration_error():
    if not val_calibration_error.empty():
        return await val_calibration_error.get()
    return None

@router.get(
    "/metrics/test_topk_acc",
    response_model=list[FloatResponseModel],
)
async def get_test_topk_acc():
    a = not test_top5_acc.empty()
    b = not test_top10_acc.empty()
    c = not test_top50_acc.empty()
    if a and b and c:
        v1 = await test_top5_acc.get()
        v2 = await test_top5_acc.get()
        v3 = await test_top5_acc.get()
        return [v1, v2, v3]
    return None

@router.get(
    "/metrics/val_topk_acc",
    response_model=list[FloatResponseModel],
)
async def get_val_topk_acc():
    a = not val_top5_acc.empty()
    b = not val_top10_acc.empty()
    c = not val_top50_acc.empty()
    if a and b and c:
        v1 = await val_top5_acc.get()
        v2 = await val_top5_acc.get()
        v3 = await val_top5_acc.get()
        return [v1, v2, v3]
    return None

@router.get(
    "/metrics/test_seqlen_loss",
    response_model=list[FloatResponseModel],
)
async def get_test_seqlen_loss():
    present = True
    for q in test_seqlen_loss:
        present = present and not q.empty()
    if present:
        return [await q.get() for q in test_seqlen_loss]
    return None

@router.get(
    "/metrics/val_seqlen_loss",
    response_model=list[FloatResponseModel],
)
async def get_val_seqlen_loss():
    present = True
    for q in val_seqlen_loss:
        present = present and not q.empty()
    if present:
        return [await q.get() for q in val_seqlen_loss]
    return None

@router.get(
    "/metrics/loop",
)
async def get_loop_exit_probs():
    present = True
    for q in loop_exit_probs:
        present = present and not q.empty()
    if present:
        response = []
        for q in loop_exit_probs:
            buffer = io.BytesIO()
            t = await q.get()
            torch.save(t, buffer)
            buffer.seek(0)
            response.append(buffer)
        content = pickle.dumps(response)
        return StreamingResponse(
            content=io.BytesIO(content),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="data.txt"'
            }
        )
    return None
