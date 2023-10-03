import uvicorn
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost.tiangolo.com","https://localhost.tiangolo.com","http://localhost","http://localhost:8080","http://localhost:3000",]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

model = pickle.load(open("model.pkl", "rb"))

class Candidate(BaseModel):
    pclass: int
    sex: int
    age: int

@app.get("/")
def read_root():
    return {"data":"welcom to the titanic survived"}

@app.post("/prediction/")
async def get_predict(data: Candidate):
    sample = [[data.pclass, data.sex, data.age]]
    hired = model.predict(sample).tolist()[0]
    return {
        "data":{
            'prediction':hired,
            'interpretation':'vous avez servecu' if hired == 1 else 'vous n avez pas survecu'
        }
    }

if __name__ == '__main__':
    uvicorn.run(app, port=8080,host='0.0.0.0')
