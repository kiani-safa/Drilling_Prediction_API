from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1) load the saved model
model=joblib.load("rop_random_forest.pkl")

# 2) Create FastAPI app
app=FastAPI(
    title="ROP Prediction API",
    description="An API to predict rate of penetration(ROP) " \
    "using random forest model provide drilling parameters as input" \
    "and get the predicted ROP as output"
    
)

# input schema with pydantic
class ROPInput(BaseModel):
    hole_depth:float
    hook_load:float
    rotary_rpm:float
    rotary_torque:float
    weight_on_bit:float
    differential_pressure:float
    gamma_at_bit:float


@app.get("/",tags=['General'],
         summary="Home page")
def home():
    """
    this is a simple welcome endpoint,
    it just returns a  message to confirm that the API is working.
    """
    return {"message":"welcome to the ROP Prediction API"}
@app.post("/predict",tags=['Prediction'],
          summary='Predict ROP',
          response_description='the predicted ROP value')


def predict(data:ROPInput):
    """
     provide drilling parameters as input
     and get the predicted ROP.

     -**hole_depth**: curent hole depth(ft)
     -**hook_load**: Hook Load(klbf)
     -**rotary_rpm **: Rotary speed(minute)
     -**rotary_torque **:(ft-lb)
     -**weight_on_bit **:(klb)
     -**differential_pressure**:(psi)
     -** gamma_at_bit **:()
     
     returns the predicted ROP value.

    """
    # convert input into numpy array
    features=np.array([[
        data.hole_depth,
        data.hook_load,
        data.rotary_rpm,
        data.rotary_torque,
        data.weight_on_bit,
        data.differential_pressure,
        data.gamma_at_bit
    ]])

    prediction=model.predict(features)

    return {
        "inputs":data.model_dump(),
        'predicted_rop':float(prediction[0])

    }