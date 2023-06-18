from fastapi import APIRouter, UploadFile, HTTPException
from model_loader import ModelLoader

router = APIRouter(
    prefix="/files",
    tags=["files"],
)

@router.post("/loadfile/")
async def create_upload_file(audio_file: UploadFile):
    if audio_file.size > 100 or not audio_file.filename.endswith(".wav"):
        print(audio_file.size)
        print(audio_file.filename)
        raise HTTPException(status_code=500, detail="Musickli doesn't support")
    musickli_model = ModelLoader()
    result = musickli_model.predict_file(audio_file)
    return result