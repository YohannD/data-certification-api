
from io import StringIO
from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel


# load model
clf = load('model.joblib')


def get_prediction(acousticness, danceability, duration_ms, energy, explicit, id, instrumentalness, \
    key, liveness, loudness, mode, name, release_date, speechiness, tempo, valence, artist):

    x = [[acousticness, danceability, duration_ms, energy, explicit, id, instrumentalness, key, liveness, loudness, mode, name, release_date, speechiness, tempo, valence, artist]]

    y = clf.predict(x)

    return {'artist': f'{artist}', 'name': '{name}', 'popularity': y}


#initiate API
app = FastAPI()

# define model for post request.
class ModelParams(BaseModel):
    acousticness: float
    danceability: float
    duration_ms: int
    energy: float
    explicit: int
    id: str
    instrumentalness: float
    key: int
    liveness: float
    loudness: float
    mode: int
    name: str
    release_date: str
    speechiness: float
    tempo: float
    valence: float
    artist: str


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

# Implement a /predict endpoint
@app.get("/predict/{acousticness}/{danceability}/{duration_ms}/{energy}/\
    {explicit}/{id}/{instrumentalness}/{key}/{liveness}/{loudness}/{mode}/\
        {name}/{release_date}/{speechiness}/{tempo}/{valence}/{artist}")
def predict(params: ModelParams):

    pred = get_prediction(f'{acousticness}'/f'{danceability}'/f'{duration_ms}'/f'{energy}'/\
    f'{explicit}'/f'{id}'/f'{instrumentalness}'/f'{key}'/f'{liveness}'/f'{loudness}'/f'{mode}'/\
        f'{name}'/f'{release_date}'/f'{speechiness}'/f'{tempo}'/f'{valence}'/f'{artist}'
)

    return pred



