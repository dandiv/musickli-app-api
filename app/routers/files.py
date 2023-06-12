from fastapi import APIRouter, FastAPI, UploadFile, HTTPException

from app.model_loader import ModelLoader

router = APIRouter(
    prefix="/files",
    tags=["files"],
)
app = FastAPI()


@app.post("/loadfile/")
async def create_upload_file(audio_file: UploadFile):
    if audio_file.size > 100 or not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=500, detail="Musickli doesn't support")
    musickli_model = ModelLoader()
    result = musickli_model.predict_file(audio_file)
    return result
