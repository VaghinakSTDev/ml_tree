from flask import Flask, render_template, request
from wtforms import Form,  SubmitField, FloatField, SelectField
from wtforms.validators import DataRequired

from tree import Tree, INPUT_COLUMNS, PATH

app = Flask(__name__)

app.debug = True
tree = Tree('STdevChurnData.xlsx')
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class SampleForm(Form):
    Age = FloatField('Age', validators=[DataRequired()])
    DurationPet = FloatField('DurationPet', validators=[DataRequired()])
    BreedName = SelectField(
        'BreedName', choices=([(bread_name, bread_name) for bread_name in tree.bread_names]),
        validators=[DataRequired()])
    TotalClaimsDenied = FloatField('TotalClaimsDenied', validators=[DataRequired()])
    TotalClaimsPaid = FloatField('TotalClaimsPaid', validators=[DataRequired()])
    TotalClaims = FloatField('TotalClaims', validators=[DataRequired()])
    ControllingStateCd = SelectField(
        'ControllingStateCd', choices=([(state_name, state_name) for state_name in tree.state_names]),
        validators=[DataRequired()])
    PolicyForm = FloatField('PolicyForm', validators=[DataRequired()])
    LastAnnualPremiumAmt = FloatField('LastAnnualPremiumAmt', validators=[DataRequired()])
    CopayPct = FloatField('CopayPct', validators=[DataRequired()])
    InitialPremiumAmtPriorYr = FloatField('InitialPremiumAmtPriorYr', validators=[DataRequired()])
    InitialWrittenPremiumAmt = FloatField('InitialWrittenPremiumAmt', validators=[DataRequired()])
    LastAnnualPremiumAmtPriorYr = FloatField('LastAnnualPremiumAmtPriorYr', validators=[DataRequired()])

    submit = SubmitField('Submit')


@app.route('/')
def index():

    form = SampleForm(request.form)
    return render_template('index.html', form=form)


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        data = request.form
        input_data = []
        for column in tree.columns:
            input_data.append(float(data[column]))

        for i in range(len(tree.bread_names) - 1):
            if data['BreedName'] == tree.bread_names[i]:
                input_data.append(1)
            else:
                input_data.append(0)
        for j in range(len(tree.state_names) - 8):
            if data["ControllingStateCd"] == tree.state_names[j]:
                input_data.append(1)
            else:
                input_data.append(0)

        result_bool = tree.predict_proba(input_data)[0]

        churn_percentage = result_bool[1]

        result = "{}% ".format(int(churn_percentage*100))

        return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0')