from pydantic import BaseModel, Field

class TravelRequest(BaseModel):
    user_query:str = Field(..., min_length=10)
    origin_city:str = Field(..., min_length=2)
    travel_date:str = Field(default="")
    trip_duration: str = Field(default="")
    
class TravelResponse(BaseModel):
    answer:str
    origin_city:str
    user_query: str
    
    