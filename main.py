from flask import Flask, request, jsonify
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO("models/yolo_v8.onnx")

def yolo_v8(image_path):
    results = model(source=image_path, task="detect")
    boxes = []
    for result in results:
        boxes = result.boxes.xyxy
    list_of_lists = boxes.tolist()
    yoloprediction = np.array(list_of_lists)
    return yoloprediction

@app.route('/get_yolo_predictions', methods=['POST'])
def get_yolo_predictions():
    try:
        # Get the URL from the request
        image_url = request.json.get('image_url')

        # Call your YOLO prediction function
        yolopredictions = yolo_v8(image_url)

        # Return the YOLO predictions as JSON response
        return jsonify({"predictions": yolopredictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
