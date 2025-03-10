import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from stream_utils import Streaming

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

streaming = Streaming()

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/devices")
def devices():
    return streaming.list_available_devices()

@app.get("/start")
def devices():
    return 
    




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
