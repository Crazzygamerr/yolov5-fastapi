from tkinter import N
import torch
from PIL import Image
import io
from pydantic import BaseModel
import os
import pandas
import json

class ImageCrop(BaseModel):
		unit: str
		x: float
		y: float
		width: float
		height: float

def get_yolov5():
		# local best.pt
		model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local')  # local repo
		model.conf = 0.25
		return model


def get_image_from_bytes(
	# binary_image, 
	input_image,
	# max_size=1024, 
	crop: ImageCrop = None,
	):
		# input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
		# width, height = input_image.size
		# resize_factor = min(max_size / width, max_size / height)
		# resized_image = input_image.resize(
		#     (
		#         int(input_image.width * resize_factor),
		#         int(input_image.height * resize_factor),
		#     )
		# )
		if crop is not None:
			resized_image = input_image.crop((
				crop.x, 
				crop.y, 
				crop.x + crop.width, 
				crop.y + crop.height
				))
		else:
			resized_image = input_image
		return resized_image
		
def getInnerBoxes(
	result,
	crop: ImageCrop = None,
):
	if crop is None:
		return result
	
	inner_boxes = []
	for box in result:
		if (float(box['xmin']) > crop.x 
			and float(box['ymin']) > crop.y 
			and float(box['xmax']) < crop.x + crop.width 
			and float(box['ymax']) < crop.y + crop.height):
			
			inner_boxes.append(box)
			
	return inner_boxes

def saveFile(
	input_image: Image,
	filename: str,
	crop: ImageCrop = None,
	result_json: str = None,
	result_pandas: pandas.DataFrame = None,
	result_image: Image = None,
	):
	
	if filename is None:
		filename = "new"
		
	filename_without_extension = ''.join(filename.split(".")[:-1])
	filename_extension = ''.join(filename.split(".")[-1:])
			
	if not os.path.exists(f"predictions/images/{filename_without_extension}_1.{filename_extension}"):
		save_filename = filename_without_extension + "_1"
	else:
		files = [f for f in os.listdir(f"predictions/images/") if f.startswith(filename_without_extension)]
		files.sort(key=lambda x: os.path.getmtime(f"predictions/images/{x}"))
		latest_file = ''.join(files[-1].split(".")[:-1])
		n = int(latest_file.split("_")[-1])
		n += 1
		save_filename = f"{filename_without_extension}_{n}"
		
	input_image.save("predictions/images/" + save_filename + "." + filename_extension)
	if crop is not None:
		with open(f"predictions/crop/{save_filename}.txt", "w") as f:
			f.write(f"{crop.x},{crop.y},{crop.width},{crop.height}\n")
	if result_json is not None:
		with open(f"predictions/predicted_json/{save_filename}.json", "w") as f:
			f.write(json.dumps(result_json))
	if result_pandas is not None:
		annotations = []
		for row in result_pandas.itertuples():
			row_str = "" + str(row.xmin) + " " + str(row.ymin) + " " + str(row.xmax) + " " + str(row.ymax) + " " + str(row.label) + "," + str(row.confidence) + "\n"
			annotations.append(row_str)
		print(annotations)
		
	if result_image is not None:
		result_image.save(f"predictions/predicted_images/{save_filename}.png")		