import onnxruntime as ort
import numpy as np
import csv
import pandas as pd

def read_csv_file(file_path):
    data = pd.read_csv(wine_path)


if __name__ == "__main__":
    file_path = 'samples.csv'

    # Download the model from mantik and insert the path to it here
    onnx_model = 'models/mnist2.onnx/model.onnx'

    # change this to predict other datapoints

    for predicted_data_point in range(20):
        labels, test_data = read_csv_file(file_path)
        ort_sess = ort.InferenceSession(onnx_model)
        outputs = ort_sess.run(None, {"x": test_data[predicted_data_point]})

        predicted, actual = outputs[0][0].argmax(0), labels[predicted_data_point]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')