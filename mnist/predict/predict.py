import onnxruntime as ort
import numpy as np
import csv

def read_csv_file(file_path):
    labels = []
    test_data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            labels.append(row[0])
            data_point = np.array(row[1:])
            data_point = data_point.reshape(
                1, 28, 28, 1
            )
            data_point = data_point.astype("float32")
            data_point /= 255
            test_data.append(data_point)
    return labels, test_data


if __name__ == "__main__":
    file_path = 'mnist_test.csv'

    # Download the model from mantik and insert the path to it here
    onnx_model = 'models/mnist.onnx/model.onnx'

    # change this to predict other datapoints
    predicted_data_point = 0

    labels, test_data = read_csv_file(file_path)
    ort_sess = ort.InferenceSession(onnx_model)
    outputs = ort_sess.run(None, {"x": test_data[predicted_data_point]})

    predicted, actual = outputs[0][0].argmax(0), labels[predicted_data_point]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')