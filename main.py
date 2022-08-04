from fastapi import FastAPI, File, Request, UploadFile
# import logging
from segmentation import get_yolov5, get_image_from_bytes, getInnerBoxes, saveFile
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Command to start server
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

class ImageCrop(BaseModel):
		unit: str
		x: float
		y: float
		width: float
		height: float

model = get_yolov5()

app = FastAPI(
		title="Custom YOLOV5 Machine Learning API",
		description="""Obtain object value out of image
										and return image and json result""",
		version="0.0.1",
)

# logger = logging.getLogger("api")
# logger.setLevel(logging.DEBUG)

origins = [
		"*"
]

app.add_middleware(
		CORSMiddleware,
		allow_origins=origins,
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
)

# @app.get('/')
# def test():
# 	return detect_food_return_json_result(file=File(""))


@app.get('/health')
def get_health():
		"""
		Usage on K8S
		readinessProbe:
				httpGet:
						path: /notify/v1/health
						port: 80
		livenessProbe:
				httpGet:
						path: /notify/v1/health
						port: 80
		:return:
				dict(msg='OK')
		"""
		return dict(msg='OK')


@app.post("/object-to-json")
async def detect_digit_return_json_result(
	request: Request,
	# file: bytes = File(...),
	file: UploadFile
	):
		form = await request.form()
		crop = None
		file_bytes = file.file.read()
		input_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
			
		results = model(input_image)
		
		detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
		detect_res = json.loads(detect_res)
		# get results.save() text file
		detect_txt = results.save()
		
		if "crop" in form.keys() and form["crop"] is not None:
			crop = ImageCrop(**json.loads(form["crop"]))
		cropped_res = getInnerBoxes(detect_res, crop)
		
		saveFile(
			input_image=input_image,
			filename=file.filename,
			crop=crop,
			result_json=detect_res,
			result_pandas=results.pandas().xyxy[0],
			result_image=Image.fromarray(results.imgs[0]),
		);
		
		return {"result": cropped_res}


@app.post("/object-to-img")
async def detect_digit_return_base64_img(
	request: Request,
	# file: bytes = File(...),
	file: UploadFile
	):
		form = await request.form()
		crop = None
		file_bytes = file.file.read()
		input_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
			
		results = model(input_image)
		results.render()  # updates results.imgs with boxes and labels
		for img in results.imgs:
			if "crop" in form.keys() and form["crop"] is not None:
				crop = ImageCrop(**(json.loads(form["crop"])))
				
			cropped_result = get_image_from_bytes(
				Image.fromarray(img),
				crop=crop
				)
				
			bytes_io = io.BytesIO()
			cropped_result.save(bytes_io, format="jpeg")
			
		saveFile(
			input_image=input_image,
			filename=file.filename,
			crop=crop,
			result_json=results.pandas().xyxy[0].to_json(orient="records"),
			# result_pandas=results.pandas().xyxy[0],
			result_image=Image.fromarray(results.imgs[0]),
		);
		
		return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
