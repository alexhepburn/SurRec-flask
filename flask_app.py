from flask import Flask
from flask import render_template, request
import pandas as pd
import numpy as np 

num_songs = 100 # controls how many songs are in the dataset
x = np.load('./x.npy')[:, 0:num_songs]
z = np.load('./z.npy')[:, 0:num_songs]
znorm = np.load('./znorm.npy')[:, 0:num_songs]
S = np.where(x!=0, 1, 0)
df = pd.read_hdf('./flask_app_df.h5', 'df').head(num_songs)
app = Flask(__name__)

def app2(a,A):
    A = A.T
    sq_dist = (A**2).sum(0) + a.dot(a) - 2*a.dot(A) 
    return sq_dist.argmin()

def get_recommendation(songs):
    idx = [df.index[df.artsong==x][0] for x in songs]
    if len(idx) == 0:
        return df.sample(1).iloc[0].artsong
    new_user = np.ones((len(idx),), dtype=int)
    S_temp = S[:, idx]
    rec = z[app2(new_user, S_temp), :]
    rec2 = znorm[app2(new_user, S_temp), :]
    rec[idx] = 0
    rec2[idx] = 0
    return (df.ix[rec.argmax()].artsong, df.ix[rec2.argmax()].artsong,
        list(df.ix[(-rec).argsort()[:5]].artsong), list(df.ix[(-rec2).argsort()[:5]].artsong))

def get_dropdown_list():
    return sorted(list(df.artsong))

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
            song, song2, rec, rec2 = get_recommendation(songs)
            print(rec)
            print(rec2)
            return render_template("index.html", songs=get_dropdown_list(), youtube_id=get_youtube_link(song), 
                youtube_idtwo=get_youtube_link(song2), rec=rec, rec2=rec2)
    return render_template("index.html", songs=get_dropdown_list())

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)