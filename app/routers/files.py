from fastapi import APIRouter, UploadFile, HTTPException
from model_loader import ModelLoader
from preprocessing.utils import save_file, normalize_result
from preprocessing.feature_extraction import get_audio_data

router = APIRouter(
    prefix="/files",
    tags=["files"],
)

@router.post("/loadfile/")
async def create_upload_file(audio_file: UploadFile):
    if not audio_file.filename.endswith(".wav"):
        raise HTTPException(status_code=500, detail="Musickli doesn't support")
    
    file_path = await save_file(audio_file)
    file_data = get_audio_data(file_path)

    musickli_model = ModelLoader()
    result = musickli_model.predict_file(file_data)
    client_result = normalize_result(result)

    return client_result