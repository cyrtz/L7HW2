# Regression Lab — Vercel + Python

Ames Housing 互動式多元線性回歸研究平台。資料來源：https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

## 功能
- 兩個研究問題（下拉選單）：特徵與預測；特徵選擇對效能影響
- 內建 `data/train.csv`，或上傳其他 CSV（數值型欄位；4 MB 以內、80–12000 筆）
- 13–39 個候選特徵；SelectKBest / RFE；5-fold CV
- LinearRegression；MAE/MSE/RMSE/R²；殘差、預測圖、95% split-conformal 預測區間
- 匯出 JSON、測試集預測 CSV

## 本機啟動
需 Node 20+、Python 3.12。

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
npm install
# terminal A
uvicorn api.index:app --reload --port 8000
# terminal B
npm run dev
```

本機 Next.js 需要代理 API：請參照 `next.config.mjs`，預設開發模式會將 `/api/*` 導到 `localhost:8000`。

## Vercel
1. 推送整個專案到 GitHub。
2. Vercel 匯入 repo，Framework Preset 選 Next.js，Root Directory 選專案根目錄。
3. Python FastAPI 位於 `api/index.py`，Vercel 自動辨識 Python Function；不需要設定額外後端網址。
4. 確認 Functions 包含 `/api/default`、`/api/analyze` 等路徑。先測試 `/api/health`。
5. Vercel Python 打包體積與執行時間因方案及 Runtime 變化；若超限請將 Python API 部署至獨立服務並調整前端代理。

## 研究設計
- 60/20/20 train/calibration/test split；固定 random_state=42。
- Pipeline 內補缺值與特徵選擇；交叉驗證僅用訓練集。
- 使用 calibration 絕對殘差有限樣本分位數建立名目 95% split-conformal 區間；在 exchangeability 假設下有邊際涵蓋率保證。
- 比較不同 k 應以 CV RMSE 為準，不能依測試集挑選 k。
- 僅支援數值型特徵；不聲稱所有 CSV 都可成功訓練。
- 不儲存上傳資料，單次請求內處理。

## CRISP-DM
1. Business Understanding：預測連續數值，分析特徵選擇效益。
2. Data Understanding：資料型態、筆數、缺失值。
3. Data Preparation：欄位驗證、缺失值填補、切分。
4. Modeling：多元線性回歸、SelectKBest/RFE。
5. Evaluation：MAE/MSE/RMSE/R²、殘差與 95% 預測區間。
6. Deployment：Next.js + FastAPI on Vercel。

## 注意
圖表僅於模型實際訓練成功後呈現。報告請將網站真實執行結果、圖表截圖、AI 協作紀錄另行整理。
