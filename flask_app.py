from flask import Flask
from flask import render_template, request
import pandas as pd
import numpy as np 

num_songs = 30 # controls how many songs are in the dataset
x = np.load('./x.npy')[:, 0:num_songs]
z = np.load('./z.npy')[:, 0:num_songs]
S = np.where(x!=0, 1, 0)
df = pd.read_hdf('./flask_app_df.h5', 'df').head(num_songs)
app = Flask(__name__)

def app2(a,A):
    A = A.T
    sq_dist = (A**2).sum(0) + a.dot(a) - 2*a.dot(A) 
    return sq_dist.argmin()

def get_recommendation(songs):
    idx = [df.index[df.artsong==x][0] for x in songs]
    print(len(idx))
    if len(idx) == 0:
        return df.sample(1).iloc[0].artsong
    new_user = np.ones((len(idx),), dtype=int)
    S_temp = S[:, idx]
    rec = z[app2(new_user, S_temp), :]
    return df.ix[rec.argmax()].artsong

def get_dropdown_list():
    return list(df.artsong)

def get_youtube_link(artsong):
    row = df.loc[df.artsong == artsong]
    return row.youtube_id.iloc[0]

@app.route('/')
def index():
    return render_template("index.html", songs=get_dropdown_list())

@app.route('/index', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        songs = request.form.getlist('songselect')
        if songs!=None:
            song = get_recommendation(songs)
            return render_template("index.html", songs=get_dropdown_list(), youtube_id=get_youtube_link(song))
    return render_template("index.html", songs=get_dropdown_list())

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)