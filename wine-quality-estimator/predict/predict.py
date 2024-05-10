import onnxruntime as ort
import pandas as pd

def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    x = data.drop(["quality"], axis=1)
    y = data[["quality"]]
    return x, y


if __name__ == "__main__":
    file_path = 'samples.csv'

    # Download the model from mantik and insert the path to it here
    onnx_model = 'models/mnist2.onnx/model.onnx'

    for predicted_data_point in range(20):
        test_data, labels = read_csv_file(file_path)
        ort_sess = ort.InferenceSession(onnx_model)
        outputs = ort_sess.run(None, {"x": test_data[predicted_data_point]})

        predicted, actual = outputs[0][0].argmax(0), labels[predicted_data_point]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')