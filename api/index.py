from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from ml.engine import read_csv,profile,calculate
import json

app=FastAPI(title='Regression Lab API')
DEFAULT=Path(__file__).resolve().parent.parent/'data'/'train.csv'

@app.get('/api/health')
def health(): return {'status':'ok'}

@app.get('/api/default')
def default_profile():
    df=read_csv(DEFAULT.read_bytes())
    return profile(df)

@app.post('/api/profile')
async def upload_profile(file:UploadFile=File(...)):
    try: return profile(read_csv(await file.read()))
    except ValueError as e: raise HTTPException(422,str(e))

@app.post('/api/analyze')
async def analyze(file:UploadFile|None=File(None),target:str=Form('SalePrice'),features:str=Form('[]'),k:int=Form(20),method:str=Form('kbest')):
    try:
        df=read_csv(await file.read() if file else DEFAULT.read_bytes())
        requested=json.loads(features)
        if not isinstance(requested,list) or any(not isinstance(x,str) for x in requested): raise ValueError('features 必須是欄位名稱陣列')
        result=calculate(df,target,requested,k,method)
        return JSONResponse(content=result)
    except (ValueError,TypeError,KeyError) as e: raise HTTPException(422,str(e))
