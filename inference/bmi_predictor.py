from joblib import load
import numpy as np
from typing import Tuple, Dict
from fastapi import FastAPI
from pydantic import BaseModel



# Map the gender to integer values
GENDER_CATEGORICAL_MAP = {"Female": 0, "Male": 1}
# insurance quote categories under following business rules
# 0. (AGE >= 18 and AGE <= 39) and (BMI < 17.49 or BMI >= 38.5) : 750
# 1. (AGE >= 40 and AGE <= 59) and (BMI < 18.49 or BMI >= 38.5) : 1000
# 2. (AGE >= 60) and (BMI < 18.49 or BMI >= 38.5) : 2000
# 3. else: 500
BMI_CATEGORICAL_MAP = {
    0: (750, "Age is between 18 to 39 and 'BMI' is either less than 17.49 or greater than 38.5"),
    1: (1000, "Age is between 40 to 59 and 'BMI' is either less than 18.49 or greater than 38.5"),
    2: (2000, "Age is greater than 60 and 'BMI' is either less than 18.49 or greater than 38.5"),
    3: (500, "BMI is in right range")
}

def quote_final(bmi_category:int, gender_discount:bool) -> Tuple[float, str]:
    quote, reason = BMI_CATEGORICAL_MAP[bmi_category]
    return quote * 0.9 if gender_discount else float(quote), reason

def weight_imperial_to_metric(weight_imperial: str) -> float:
    # weight_imperial is in pounds
    return int(weight_imperial) * 0.453592


def bmi_metric(height: float, weight: float) -> float:
    return weight / (height**2)

def height_imperial_to_metric(height_imperial: str) -> float:
    # height_imperial is in foot concatenated with inches
    # e.g. '510' is 5 feet 10 inches
    # e.g. '503' is 5 feet 3 inches
    # note that the human height realistically shouldn't be more than 9 feet
    # therefore we ignore the cases where input string has length that is not 3
    assert len(height_imperial) == 3, "height_imperial string must be 3 characters long"
    feet, inches = int(height_imperial[0]), int(height_imperial[1:])
    inches += feet * 12
    return inches * 0.0254

class Model(object):
    def __init__(self, model_path: str):
        self.model = load(model_path)
    
    def predict(self, dct:Dict) -> Tuple[float, str]:
        age = dct["age_int"]
        gender = dct["gender_c_int"]
        bmi = dct["bmi_float"]
        data = np.array([age, gender, bmi])[np.newaxis]
        cls = self.model.predict(data)
        cls = int(cls[0])
        quote, reason = quote_final(cls, gender == 0) # 0 is Female
        return quote, reason

class ApplicantInfo(BaseModel):
    app_id: int
    age: int
    gender: str
    ht: str
    wt: str
    issue_date: str | None = None

    def parse(self) -> Dict:
        ret = {}
        ret['app_id_int'] = self.app_id
        ret['age_int'] = self.age
        ret['gender_c_int'] = GENDER_CATEGORICAL_MAP[self.gender]
        ret['weight_metric_float'] = weight_imperial_to_metric(self.wt)
        ret['height_metric_float'] = height_imperial_to_metric(self.ht)
        ret['issue_date'] = self.issue_date
        ret['bmi_float'] = bmi_metric(ret['height_metric_float'], ret['weight_metric_float'])
        return ret


app = FastAPI()
MODEL_PATH = "clf_gdbt_synthetic_9111.joblib"
model = Model(MODEL_PATH)

@app.post("/bmi_predict/")
async def bmi_predict(applicant_info:ApplicantInfo):
    try:
        x = applicant_info.parse()
        quote, reason = model.predict(x)
        return {"appid": x['app_id_int'], "quote": quote, "reason": reason}
    except Exception as e:
        return {"quote": None, "reason": "Internal Error"}



