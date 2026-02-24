"""Travel Assistant - Entry point."""
from dotenv import load_dotenv
load_dotenv()

import uuid
from .graph import graph


def run(user_query:str, travel_date: str = '', trip_duration: str = '') -> str:
    config = {"configurable": {'thread_id':str(uuid.uuid4())}}
    result = graph.invoke({'user_query':user_query, 'travel_date': travel_date, 'trip_duration':trip_duration}, config=config)
    return result['final_answer']

if __name__ == "__main__":
    query = input("Where would you like to travel? ")
    travel_date = input('When are you planning to travel? (press Enter to skip) ')
    trip_duration = input('How many days? (press Enter to skip) ')
    print('\nResearching your trip...\n')
    answer = run(query, travel_date, trip_duration)
    print(answer)