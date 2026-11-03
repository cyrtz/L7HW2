from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from ml.engine import read_csv,profile,calculate
import json
import math
import numpy as np

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

app=FastAPI(title='Regression Lab API')
DEFAULT=Path(__file__).resolve().parent.parent/'data'/'train.csv'

@app.get('/api/health')
def health(): return {'status':'ok'}

@app.get('/api/default')
def default_profile():
    df=read_csv(DEFAULT.read_bytes())
    return json_safe(profile(df))

@app.post('/api/profile')
async def upload_profile(file:UploadFile=File(...)):
    try: return json_safe(profile(read_csv(await file.read())))
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/analyze')
async def analyze(file:UploadFile|None=File(None),target:str=Form('SalePrice'),features:str=Form('[]'),k:int=Form(20),method:str=Form('kbest')):
    try:
        df=read_csv(await file.read() if file else DEFAULT.read_bytes())
        requested=json.loads(features)
        if not isinstance(requested,list) or any(not isinstance(x,str) for x in requested): raise ValueError('features 必須是欄位名稱陣列')
        result=calculate(df,target,requested,k,method)
        return JSONResponse(content=json_safe(result))
    except (ValueError,TypeError,KeyError) as e: raise HTTPException(422,str(e))
