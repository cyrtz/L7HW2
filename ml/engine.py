import io
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAX_ROWS=12000
MAX_COLS=180

def read_csv(raw: bytes):
    if len(raw)>4_000_000: raise ValueError('CSV 檔案不得超過 4 MB')
    try: df=pd.read_csv(io.BytesIO(raw),low_memory=False)
    except Exception as e: raise ValueError('CSV 格式無法解析') from e
    if len(df)<80 or len(df)>MAX_ROWS or len(df.columns)>MAX_COLS: raise ValueError('資料需有 80–12000 筆，且不超過 180 欄')
    if df.columns.duplicated().any(): raise ValueError('欄位名稱不可重複')
    return df.replace([np.inf,-np.inf],np.nan)

def profile(df):
    nums=[c for c in df.select_dtypes(include='number').columns if df[c].notna().sum()>50 and df[c].nunique(dropna=True)>3]
    head=df.head(8)
    # astype(object) first: a float64 column coerces None back to NaN, which is not valid JSON.
    return {'rows':len(df),'columns':len(df.columns),'numericColumns':nums,'columnInfo':[{'name':str(c),'type':str(df[c].dtype),'missing':int(df[c].isna().sum())} for c in df.columns], 'preview':head.astype(object).where(pd.notnull(head),None).to_dict('records')}

def calculate(df,target,features,k=20,method='kbest',seed=42):
    if target not in df or target in features: raise ValueError('目標欄位設定不正確')
    if len(set(features))!=len(features) or not 13<=len(features)<=39: raise ValueError('候選特徵必須為 13–39 個且不得重複')
    if any(f not in df for f in features): raise ValueError('特徵欄位不存在')
    if not 13<=k<=len(features): raise ValueError('選取特徵數須介於 13 與候選特徵數之間')
    if method not in ('kbest','rfe'): raise ValueError('不支援的特徵選擇方法')
    # Numeric-only model: avoids silently inflating one-hot feature count.
    if any(not pd.api.types.is_numeric_dtype(df[f]) for f in features+[target]): raise ValueError('目前僅支援數值型特徵與目標')
    clean=df[features+[target]].copy()
    clean=clean.dropna(subset=[target])
    clean=clean.loc[clean[target].apply(np.isfinite)]
    if len(clean)<80: raise ValueError('有效樣本不足 80 筆')
    X=clean[features].astype(float)
    y=clean[target].astype(float)
    if y.nunique()<10: raise ValueError('目標欄位必須是連續數值')
    if any(X[f].notna().sum()<30 for f in features): raise ValueError('某些特徵有效值不足 30 筆')
    # 60% train, 20% calibration, 20% holdout test.
    X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.4,random_state=seed)
    X_cal,X_test,y_cal,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=seed)
    def make_model(n,selection_method):
        selector=SelectKBest(score_func=f_regression,k=n) if selection_method=='kbest' else RFE(LinearRegression(),n_features_to_select=n,step=.2)
        return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('select',selector),('model',LinearRegression())])
    cv=KFold(n_splits=5,shuffle=True,random_state=seed)
    candidate_ks=sorted(set([v for v in (13,15,20,25,30,35,39,k) if v<=len(features)]))
    comparisons=[]
    for m in ('kbest','rfe'):
        for n in candidate_ks:
            model=make_model(n,m)
            scores=-cross_val_score(model,X_train,y_train,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=1)
            comparisons.append({'method':m,'features':n,'cvRmse':round(float(scores.mean()),3),'cvStd':round(float(scores.std()),3)})
    # User-selected configuration is evaluated on holdout; CV chart supports exploration.
    model=make_model(k,method)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    cal_errors=np.abs(y_cal.to_numpy()-model.predict(X_cal))
    alpha=.05
    rank=min(len(cal_errors),math.ceil((len(cal_errors)+1)*(1-alpha)))
    q=float(np.sort(cal_errors)[rank-1])
    actual=y_test.to_numpy()
    rmse=float(np.sqrt(mean_squared_error(actual,pred)))
    selected=[f for f,flag in zip(features,model.named_steps['select'].get_support()) if flag]
    coefficients=model.named_steps['model'].coef_
    results={'mae':float(mean_absolute_error(actual,pred)),'mse':float(mean_squared_error(actual,pred)),'rmse':rmse,'r2':float(r2_score(actual,pred)),'coverage':float(np.mean((actual>=pred-q)&(actual<=pred+q))),'intervalWidth':2*q}
    points=[{'actual':float(a),'predicted':float(p),'lower':float(p-q),'upper':float(p+q),'residual':float(a-p)} for a,p in zip(actual[:250],pred[:250])]
    # Baseline comparison on same holdout, no selection.
    baseline=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('model',LinearRegression())])
    baseline.fit(X_train,y_train)
    baseline_pred=baseline.predict(X_test)
    return {'metrics':results,'baseline':{'mae':float(mean_absolute_error(actual,baseline_pred)),'rmse':float(np.sqrt(mean_squared_error(actual,baseline_pred))),'r2':float(r2_score(actual,baseline_pred))},'selected':selected,'coefficients':[{'feature':f,'coefficient':float(c)} for f,c in zip(selected,coefficients)],'comparisons':comparisons,'points':points,'split':{'train':len(X_train),'calibration':len(X_cal),'test':len(X_test)},'intervalMethod':'Split conformal absolute residual, finite-sample quantile (95% nominal)','note':'Test set is for final assessment; choose method and feature count using training cross-validation, not test results.'}
