from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import travel

@asynccontextmanager
async def lifespan(app:FastAPI):
    yield
    
app = FastAPI(
    title="Travel Assistant API",
    version="1.0.0",
)

app.include_router(travel.router)