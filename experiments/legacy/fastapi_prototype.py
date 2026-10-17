from fastapi import FastAPI
from pydantic import BaseModel
import fastapi
import uvicorn
import os
import signal
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://0.0.0.0:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define your AI model function here
def my_ai_model(data):
    # Dummy model function, replace with your actual model inference code
    return {"result": "AI model output based on input data"}

# Define a request body model using Pydantic
class InputData(BaseModel):
    input_text: str


# Add a shutdown event
def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return fastapi.Response(status_code=200, content='Server shutting down...')

def hello():
    return fastapi.Response(status_code=200, content='Hello, world!')

@app.on_event('shutdown')
def on_shutdown():
    print('Server shutting down...')

app.add_api_route('/hello', hello, methods=['GET'])
app.add_api_route('/shutdown', shutdown, methods=['GET'])


# Define a route that uses the AI model
@app.post("/predict")
async def predict(data: InputData):
    result = my_ai_model(data.input_text)
    return result

# Function to run the FastAPI app with Uvicorn
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)