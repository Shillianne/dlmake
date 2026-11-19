from ..models import UpdateBenchmarksRequestModel, FloatResponseModel
from fastapi import APIRouter
from ..state import *


router = APIRouter()

@router.post(
    "/benchmarks/update",
    status_code=200,
    summary="Updates the value of the given stats",
    description="""
    Enqueues the value given in the request to separate queues.
    """
)
async def update(request: UpdateBenchmarksRequestModel):
    if request.hellaswag is not None:
        await hellaswag.put(request.hellaswag)
    if request.arc_easy is not None:
        await arc_easy.put(request.arc_easy)
    if request.arc_challenge is not None:
        await arc_challenge.put(request.arc_challenge)
    if request.piqa is not None:
        await piqa.put(request.piqa)
    if request.sciq is not None:
        await sciq.put(request.sciq)
    if request.winogrande is not None:
        await winogrande.put(request.winogrande)
    if request.openbookqa is not None:
        await openbookqa.put(request.openbookqa)
    if request.commonsenseqa is not None:
        await commonsenseqa.put(request.commonsenseqa)
    if request.mmlu is not None:
        await mmlu.put(request.mmlu)


@router.get(
    "/benchmarks/hellaswag",
    response_model=list[FloatResponseModel],
)
async def get_hellaswag():
    if not hellaswag.empty():
        return await hellaswag.get()
    return None

@router.get(
    "/benchmarks/arc_easy",
    response_model=list[FloatResponseModel],
)
async def get_arc_easy():
    if not arc_easy.empty():
        return await arc_easy.get()
    return None

@router.get(
    "/benchmarks/arc_challenge",
    response_model=list[FloatResponseModel],
)
async def get_arc_challenge():
    if not arc_challenge.empty():
        return await arc_challenge.get()
    return None

@router.get(
    "/benchmarks/piqa",
    response_model=list[FloatResponseModel],
)
async def get_piqa():
    if not piqa.empty():
        return await piqa.get()
    return None

@router.get(
    "/benchmarks/sciq",
    response_model=list[FloatResponseModel],
)
async def get_sciq():
    if not sciq.empty():
        return await sciq.get()
    return None

@router.get(
    "/benchmarks/winogrande",
    response_model=list[FloatResponseModel],
)
async def get_winogrande():
    if not winogrande.empty():
        return await winogrande.get()
    return None

@router.get(
    "/benchmarks/openbookqa",
    response_model=list[FloatResponseModel],
)
async def get_openbookqa():
    if not openbookqa.empty():
        return await openbookqa.get()
    return None

@router.get(
    "/benchmarks/commonsenseqa",
    response_model=list[FloatResponseModel],
)
async def get_commonsenseqa():
    if not commonsenseqa.empty():
        return await commonsenseqa.get()
    return None

@router.get(
    "/benchmarks/mmlu",
    response_model=list[FloatResponseModel],
)
async def get_mmlu():
    if not mmlu.empty():
        return await mmlu.get()
    return None

