from pathlib import Path
from ml.engine import read_csv,profile,calculate

def test_ames():
    df=read_csv((Path(__file__).resolve().parents[1]/'data/train.csv').read_bytes())
    assert len(df)==1460
    assert len(profile(df)['numericColumns'])>=36
    features=[c for c in df.select_dtypes('number').columns if c not in ('Id','SalePrice')]
    result=calculate(df,'SalePrice',features,20)
    assert len(result['selected'])==20
    assert 0<=result['metrics']['coverage']<=1
    assert result['metrics']['rmse']>0
