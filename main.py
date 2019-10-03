from io import BytesIO
import numpy as np

from flask import Flask, render_template, request, Response
from wtforms import Form,  SubmitField, FloatField, SelectField, FileField
from wtforms.validators import DataRequired
from werkzeug.wsgi import FileWrapper
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
    return load_model('data/STdevChurnData_binary_model.h5')


breads_encoder = np.load('data/breads_encoder.npy')
states_encoder = np.load('data/states_encoder.npy')


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
        import pdb
        data = request.form.to_dict(flat=False)
        data.pop('submit')
        data = handle_data(pd.DataFrame(data), breads_encoder, states_encoder)
        model = get_model()
        result_bool = model.predict_proba(data)
        churn_percentage = result_bool[0][0]
        result = "{}% ".format(int(churn_percentage * 100))

        return render_template("result.html", result=result)


@app.route('/result_file', methods=['POST'])
def result_file():
    if request.method == 'POST':

        df = pd.read_excel(request.files.get('file'))

        data = handle_data(df[INPUT_COLUMNS], breads_encoder, states_encoder)

        model = load_model('data/STdevChurnData_binary_model.h5')
        a = model.predict_classes(data)
        # df['prediction'] = [1 if i > 0.7 else 0 for i in a]
        df['prediction'] = model.predict_proba(data).round()


        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')

        df.to_excel(writer, startrow=0, merge_cells=False)
        writer.close()
        output.seek(0)


        return Response(output, mimetype="application/vnd.ms-excel")


if __name__ == "__main__":
    app.run(host='0.0.0.0')