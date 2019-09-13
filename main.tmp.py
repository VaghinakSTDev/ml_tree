from io import BytesIO
import numpy as np

from flask import Flask, render_template, request, send_file
from wtforms import Form,  SubmitField, FloatField, SelectField, FileField
from wtforms.validators import DataRequired
# import tensorflow
from tensorflow.keras.models import load_model
from tree import Tree
# from neural_network import Network
import pickle
import pandas as pd

from utils import handle_data, INPUT_COLUMNS

app = Flask(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.debug = True


def get_model():
    return load_model('model_data/STdevChurnData_binary_model.h5')


breads_encoder = np.load('model_data/breads_encoder.npy')
states_encoder = np.load('model_data/states_encoder.npy')


class SampleForm(Form):
    Age = FloatField('Age', validators=[DataRequired()])
    DurationPet = FloatField('DurationPet', validators=[DataRequired()])
    BreedName = SelectField(
        'BreedName', choices=([(bread_name, bread_name) for bread_name in breads_encoder]),
        validators=[DataRequired()])
    TotalClaimsDenied = FloatField('TotalClaimsDenied', validators=[DataRequired()])
    TotalClaimsPaid = FloatField('TotalClaimsPaid', validators=[DataRequired()])
    TotalClaims = FloatField('TotalClaims', validators=[DataRequired()])
    ControllingStateCd = SelectField(
        'ControllingStateCd', choices=([(state_name, state_name) for state_name in states_encoder]),
        validators=[DataRequired()])
    PolicyForm = FloatField('PolicyForm', validators=[DataRequired()])
    LastAnnualPremiumAmt = FloatField('LastAnnualPremiumAmt', validators=[DataRequired()])
    CopayPct = FloatField('CopayPct', validators=[DataRequired()])
    InitialPremiumAmtPriorYr = FloatField('InitialPremiumAmtPriorYr', validators=[DataRequired()])
    InitialWrittenPremiumAmt = FloatField('InitialWrittenPremiumAmt', validators=[DataRequired()])
    LastAnnualPremiumAmtPriorYr = FloatField('LastAnnualPremiumAmtPriorYr', validators=[DataRequired()])

    submit = SubmitField('Submit')


class SampleUploadForm(Form):
    file = FileField()


@app.route('/')
def index():
    sample_form = SampleForm(request.form)
    upload_form = SampleUploadForm(request.form)
    return render_template('index.html', sample_form=sample_form, upload_form=upload_form)


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # data = request.form
        # input_data = []
        # for column in tree.columns:
        #     input_data.append(float(data[column]))
        #
        # for i in range(len(tree.bread_names) - 1):
        #     if data['BreedName'] == tree.bread_names[i]:
        #         input_data.append(1)
        #     else:
        #         input_data.append(0)
        # for j in range(len(tree.state_names) - 8):
        #     if data["ControllingStateCd"] == tree.state_names[j]:
        #         input_data.append(1)
        #     else:
        #         input_data.append(0)
        #
        # result_bool = tree.predict_proba(input_data)[0]
        #
        # churn_percentage = result_bool[1]
        #
        # result = "{}% ".format(int(churn_percentage * 100))
        #
        # return render_template("result.html", result=result)

        pass
@app.route('/result_file', methods=['POST'])
def result_file():
    if request.method == 'POST':

        df = pd.read_excel(request.files.get('file'))

        X = handle_data(df[INPUT_COLUMNS], breads_encoder, states_encoder)

        model = get_model()
        result = model.predict_proba(X)
        df['prediction'] = result.round(2)
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        df.to_excel(writer, startrow=0, merge_cells=False)
        writer.close()
        output.seek(0)
        return send_file(output, attachment_filename="testing.xlsx", as_attachment=True)


if __name__ == "__main__":

    app.run(host='0.0.0.0')