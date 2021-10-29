import flask
import pickle
import pandas as pd
# Use pickle to load in the pre-trained model.
with open(f'irisModel.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        sepal_length = flask.request.form['sepal_length']
        sepal_width = flask.request.form['sepal_width']
        petal_length = flask.request.form['petal_length']
        petal_width = flask.request.form['petal_width']
        
        
        input_variables = pd.DataFrame([[sepal_length, sepal_width, petal_length,petal_width]],
                                       columns=['sepal_length', 'sepal_width', 'petal_length','petal_width'], dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('index.html',
                                     original_input={'sepal_length':sepal_length,
                                                     'sepal_width':sepal_width,
                                                     'petal_length':petal_length,
                                                     'petal_width':petal_width,
                                                   },
                                     result=prediction,
                                     )
