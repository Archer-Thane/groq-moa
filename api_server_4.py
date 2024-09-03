from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import copy

# Ensure these imports are correct based on your project's structure
from groq import Groq
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk

# Default and recommended configurations as defined in the Streamlit app
default_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 3,
    "layer_agent_config": {}
}

layer_agent_config_def = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama3-8b-8192"
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "gemma-7b-it",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3-8b-8192"
    },
}

rec_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 2,
    "layer_agent_config": {}
}

layer_agent_config_rec = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.2
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.4
    },
    "layer_agent_4": {
        "system_prompt": "You are an expert planner agent. Create a plan for how to answer the human's query. {helper_response}",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.5
    },
}

# Pydantic models for request and response
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]
    config: Dict[str, Any] = None

class QueryResponse(BaseModel):
    response: str

app = FastAPI()

# Initialize MOAgent with default settings
def set_moa_agent(
    main_model: str = default_config['main_model'],
    cycles: int = default_config['cycles'],
    layer_agent_config: dict = copy.deepcopy(layer_agent_config_def),
    main_model_temperature: float = 0.1,
):
    moa_agent = MOAgent.from_config(
        main_model=main_model,
        cycles=cycles,
        layer_agent_config=layer_agent_config,
        temperature=main_model_temperature
    )
    return moa_agent

moa_agent = set_moa_agent()

@app.post("/query", response_model=QueryResponse)
async def query_moa(request: QueryRequest):
    global moa_agent
    try:
        if request.config:
            # Update MOAgent configuration based on request
            config = request.config
            moa_agent = set_moa_agent(
                main_model=config.get('main_model', default_config['main_model']),
                cycles=config.get('cycles', default_config['cycles']),
                layer_agent_config=config.get('layer_agent_config', copy.deepcopy(layer_agent_config_def)),
                main_model_temperature=config.get('main_model_temperature', 0.1),
            )

        # Convert the messages history to the format expected by the MOAgent
        conversation_history = [{'role': msg.role, 'content': msg.content} for msg in request.messages]

        response_stream = moa_agent.chat(conversation_history, output_format='json')
        response_chunks = list(response_stream)
        response = ''.join(chunk['delta'] for chunk in response_chunks if chunk['response_type'] != 'intermediate')
        
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
