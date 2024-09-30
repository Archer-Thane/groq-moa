from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator
import copy
import json
from groq import Groq
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk

from functools import lru_cache

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

# Pydantic models for request and response
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]
    config: Dict[str, Any] = None
    reference_system_prompt: str = None

class LayerResponse(BaseModel):
    layer: str
    agent_responses: List[str]

class QueryResponse(BaseModel):
    intermediate_responses: List[LayerResponse]
    final_response: str

app = FastAPI()

@lru_cache(maxsize=1)
def get_moa_agent():
    return set_moa_agent()

# Initialize MOAgent with default settings
def set_moa_agent(
    main_model: str = default_config['main_model'],
    cycles: int = default_config['cycles'],
    layer_agent_config: dict = copy.deepcopy(layer_agent_config_def),
    main_model_temperature: float = 0.1,
    reference_system_prompt: str = None
):
    moa_agent = MOAgent.from_config(
        main_model=main_model,
        cycles=cycles,
        layer_agent_config=layer_agent_config,
        temperature=main_model_temperature,
        reference_system_prompt=reference_system_prompt
    )
    return moa_agent

moa_agent = set_moa_agent()

# Corrected: Define stream_response as an async function
async def stream_response(messages: AsyncGenerator[ResponseChunk, None]):
    layer_outputs = {}
    final_output = ""
    async for message in messages:
        if message['response_type'] == 'intermediate':
            layer = str(message['metadata']['layer'])  # Convert layer to string
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            # Accumulate final output
            final_output += message['delta']

    return layer_outputs, final_output

@app.post("/query", response_model=QueryResponse)
async def query_moa(request: QueryRequest):
    global moa_agent
    try:
        if request.config:
            # Update MOAgent configuration based on request
            config = request.config
            reference_system_prompt = request.reference_system_prompt
            moa_agent = set_moa_agent(
                main_model=config.get('main_model', default_config['main_model']),
                cycles=config.get('cycles', default_config['cycles']),
                layer_agent_config=config.get('layer_agent_config', copy.deepcopy(layer_agent_config_def)),
                main_model_temperature=config.get('main_model_temperature', 0.8),
                reference_system_prompt=reference_system_prompt
            )

        # Convert the messages history to the format expected by the MOAgent
        conversation_history = [{'role': msg.role, 'content': msg.content} for msg in request.messages]

        # Use the async chat method
        response_stream = moa_agent.chat(
            input=conversation_history[-1]['content'],  # Assuming last message is the user's input
            messages=conversation_history[:-1],  # Previous messages
            output_format='json'
        )

        # Corrected: Await the async function
        layer_outputs, final_output = await stream_response(response_stream)
        
        intermediate_responses = [
            LayerResponse(layer=layer, agent_responses=outputs)
            for layer, outputs in layer_outputs.items()
        ]
        
        return QueryResponse(intermediate_responses=intermediate_responses, final_response=final_output)

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
