"""Travel Assistant - Entry point."""
import uuid
from .graph import graph

def run(user_query:str) -> str:
    config = {"configurable": {'thread_id':str(uuid.uuid4())}}
    result = graph.invoke({'user_query':user_query}, config=config)
    return result['final_answer']

if __name__ == "__main__":
    query = input("Where would you like to travel?")
    print('\nResearching your trip...\n')
    answer = run(query)
    print(answer)