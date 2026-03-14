import asyncio
from fastapi import APIRouter, HTTPException
from api.schemas.travel import TravelRequest, TravelResponse
from src.main import run

router = APIRouter()
'''
 Por qué asyncio.to_thread(): graph.invoke() es sincrónico y puede tardar 1-3 minutos. Llamarlo directamente desde un
 endpoint async bloquea el event loop de FastAPI. asyncio.to_thread() lo ejecuta en un thread separado sin bloquear.
'''

@router.post("/travel", response_model=TravelResponse)
async def create_travel_plan(request: TravelRequest):
    try:
        answer = await asyncio.to_thread(
            run,
            request.user_query,
            request.origin_city,
            request.travel_date,
            request.trip_duration,
        )
        return TravelResponse(
            answer=answer,
            origin_city=request.origin_city,
            user_query=request.user_query
        )
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg or "429" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Rate limit de OpenAI alcanzado. Esperá un minuto y reintentá"
            )
        raise HTTPException(status_code=500, detail=str(e))
        