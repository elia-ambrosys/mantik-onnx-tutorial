import onnxruntime as ort
import pandas as pd
import numpy as np

def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    x = data.drop(["quality"], axis=1)
    y = data[["quality"]]
    return x, y


if __name__ == "__main__":
    file_path = 'samples.csv'

    # Download the model from mantik and insert the path to it here
    onnx_model = 'models/wine-quality-estimator.onnx/model.onnx'

    for predicted_data_point in range(20):
        test_data, labels = read_csv_file(file_path)
        ort_sess = ort.InferenceSession(onnx_model)
        data_point = test_data.iloc[predicted_data_point].to_numpy(dtype="float32")
        data_point = data_point.reshape(1, len(data_point))
        outputs = ort_sess.run(None, {"x": data_point})

        predicted, actual = outputs[0][0][0], labels.iloc[predicted_data_point].to_numpy(dtype="float32")[0]
        print(f'Predicted: {predicted}, Actual: {actual}')