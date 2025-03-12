import os
import uuid
import shutil
import asyncio

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse

from api.my_env import my_env
from src.wound_image import WoundImage

TEMPLATES = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "templates")
TEMP_DIR = os.path.join("output", "api", "temp")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
EXPECTED_FORMATS = {
    "segmentation_mask": "save_segmentation_mask",
    "segmentation_semantic": "save_segmentation_semantic",
    "mask_wound": "save_mask_wound",
    "mask_peri_wound": "save_mask_peri_wound",
    "masked_wound": "save_masked_wound",
    "masked_peri_wound": "save_masked_peri_wound",
    "pwat_estimation": "save_pwat_estimation",
}


def gen_id():
    return str(uuid.uuid4())


async def auto_delete_file(file_path: str, delay: float):
    """Delete the file after `delay` seconds."""
    await asyncio.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)


async def auto_delete_dir(dir_path: str, delay: float):
    """Delete the directory after `delay` seconds."""
    await asyncio.sleep(delay)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events in a single function."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    yield  # here the app running
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


@app.get("/", response_class=RedirectResponse)
async def root():
    """
    Redirect from the root ("/") to the "/docs" path.
    """
    return RedirectResponse(url="/docs")


@app.get("/valid_extensions")
async def get_valid_extensions():
    return list(VALID_EXTENSIONS)


@app.get("/expected_formats")
async def get_expected_formats():
    return list(EXPECTED_FORMATS.keys())


@app.get("/upload")
async def get_upload():
    return FileResponse(os.path.join(TEMPLATES, 'upload.html'))


@app.post("/upload")
async def upload_image(expected_format: str,
                       file: UploadFile = File(...)) -> FileResponse:
    """Upload and process an image based on the expected format. In the header, x-predicted-pwat is the predicted PWAT score."""
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in VALID_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid image format. Use one of: {await get_valid_extensions()}.")

    if expected_format not in EXPECTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid expected format. Use one of: {await get_expected_formats()}")

    file_uuid = gen_id()
    file_path = os.path.join(TEMP_DIR, f"{file_uuid}{file_ext}")
    file_new_path = os.path.join(TEMP_DIR, f"{file_uuid}_processed{file_ext}")

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    wi = WoundImage(image_path=file_path, logging=my_env.is_dev())
    predicted_pwat = wi.get_predicted_pwat()
    getattr(wi, EXPECTED_FORMATS[expected_format])(file_new_path)

    # Schedule auto-delete
    asyncio.create_task(auto_delete_file(file_path, delay=1))
    asyncio.create_task(auto_delete_file(file_new_path, delay=1))

    return FileResponse(file_new_path, media_type=file.content_type, headers={
                        "predicted_pwat": str(predicted_pwat)})


@app.get("/upload/pwat")
async def get_upload_pwat():
    return FileResponse(os.path.join(TEMPLATES, 'upload_pwat.html'))


@app.post("/upload/pwat")
async def pwat_from_image(file: UploadFile = File(...)) -> JSONResponse:
    """Upload and process an image to get the predicted PWAT score."""
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in VALID_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid image format. Use one of: {await get_valid_extensions()}.")

    file_path = os.path.join(TEMP_DIR, f"{gen_id()}{file_ext}")

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    wi = WoundImage(image_path=file_path, logging=my_env.is_dev())
    predicted_pwat = wi.get_predicted_pwat()

    # Schedule auto-delete
    asyncio.create_task(auto_delete_file(file_path, delay=1))

    return JSONResponse(content={"predicted_pwat": predicted_pwat})
