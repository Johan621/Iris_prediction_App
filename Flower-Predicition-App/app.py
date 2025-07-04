#Step 1: Train the model
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import gradio as gr
iris = load_iris()
x = iris.data
y = iris.target
model = DecisionTreeClassifier()
model.fit(x, y)

target_names = iris.target_names


#Define prediction function
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    return f" 🌺🌼Predicted Species: **{target_names[prediction].capitalize()}**"

#Create Gradio UI

iface = gr.Interface(
    fn = predict_iris,
    inputs=[
        gr.Number(label="Sepal Length (cm)", value=5.1),
        gr.Number(label="Sepal width (cm)", value=3.5),
        gr.Number(label="Petal Length (cm)", value=1.4),
        gr.Number(label="Petal Width (cm)", value=0.2),
    ],
    outputs="markdown",
    title="🌸🌻Iris Flower Classifier",
    description="Enter flower measurements to predict its species using a Decision Tree model.",
    theme="soft"
)
#Launch the app
iface.launch(share=True)
