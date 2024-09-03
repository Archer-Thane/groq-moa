from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import copy
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk

app = FastAPI()

# Default configuration from the Streamlit script
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

class RequestModel(BaseModel):
    query: str
    main_model: str = default_config['main_model']
    cycles: int = default_config['cycles']
    layer_agent_config: Dict[str, Any] = layer_agent_config_def
    main_model_temperature: float = 0.1

class ResponseModel(BaseModel):
    response: str

def stream_response(messages: List[ResponseChunk]) -> str:
    response = []
    for message in messages:
        response.append(message['delta'])
    return ''.join(response)

def set_moa_agent(
    main_model: str,
    cycles: int,
    layer_agent_config: Dict[str, Any],
    main_model_temperature: float
) -> MOAgent:
    return MOAgent.from_config(
        main_model=main_model,
        cycles=cycles,
        layer_agent_config=copy.deepcopy(layer_agent_config),
        temperature=main_model_temperature
    )

@app.post("/process", response_model=ResponseModel)
def process(request: RequestModel):
    try:
        moa_agent = set_moa_agent(
            main_model=request.main_model,
            cycles=request.cycles,
            layer_agent_config=request.layer_agent_config,
            main_model_temperature=request.main_model_temperature
        )
        response_chunks = moa_agent.chat(request.query, output_format='json')
        response = stream_response(response_chunks)
        return ResponseModel(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
