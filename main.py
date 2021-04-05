import utils
import pandas as pd
import numpy as np
import NaiveBayes as nb
from flask import Flask, request, jsonify, render_template, url_for, session, redirect
from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm
from wtforms import TextAreaField
from wtforms.widgets import TextArea


VOCAB_SIZE = 2500

vocab_df = pd.read_csv('resources/word-by-id.csv')
word_index = pd.Index(vocab_df.VOCAB_WORD)

message = "yOU want some free viagra"


app = Flask(__name__)
app.config['SECRET_KEY']='mysecretkey' #allowas the form to work, makes sure it is not being hacked

class EmailForm(FlaskForm):


    Email_message = TextAreaField('Enter you email message body you want to verify:', widget=TextArea())
    submit = SubmitField("Verify")



@app.route('/', methods=['GET','POST'])

def index():
    form = EmailForm()

    if form.validate_on_submit():
        session['email_msg'] = form.Email_message.data


        return redirect(url_for("prediction"))
        #only redirect to prediction if the form is validared upon submission
    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():
    content = {}
    content['message'] = str(session['email_msg'])

    word_list = utils.clean_message(content['message'])
    print(word_list)
    word_df = utils.make_dataframe([word_list])
    print(word_df.index[0])
    sparse_df = utils.make_sparse_matrix(word_df, word_index)
    sparse_df = np.array(sparse_df)
    full_df = nb.make_full_matrix(sparse_df, VOCAB_SIZE)
    full_df = np.array(full_df)
    output = nb.predict(full_df)



    return render_template('prediction.html', results=output)


if __name__ =="__main__":
    app.run()


#
# word_list = utils.clean_message(message)
#
# print(word_list)
#
# word_df = utils.make_dataframe([word_list])
#
# sparse_df = utils.make_sparse_matrix(word_df,word_index)
# print(sparse_df)
# sparse_df = np.array(sparse_df)
#
# print(sparse_df)
# full_df = nb.make_full_matrix(sparse_df,VOCAB_SIZE)
#
# full_df = np.array(full_df)
#
# output = nb.predict(full_df)
# print(output)

