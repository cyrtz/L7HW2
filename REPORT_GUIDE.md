# CRISP-DM 研究報告草稿架構

**研究題目：** 基於多元線性回歸與特徵選擇的房價預測分析平台

**資料來源：** Kaggle House Prices – Advanced Regression Techniques (Ames Housing), https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

## 1. Business Understanding
問題一：哪些數值型住宅特徵可用於預測成交價？問題二：特徵選擇方法與特徵數量如何影響預測效能？

## 2. Data Understanding
原始資料 1460 筆、81 欄（79 個解釋特徵、Id、SalePrice）。請附上網站資料預覽、缺失值統計、欄位說明。

## 3. Data Preparation
排除 Id、選擇 36 個數值型候選特徵；60/20/20 分為訓練、校準、測試集。Pipeline 內以中位數填補缺失值，避免資料洩漏。只使用訓練集選擇特徵。

## 4. Modeling
使用 LinearRegression + SelectKBest / RFE，並在訓練集上進行 5-fold CV，比較 13–39 個特徵範圍內的候選數量。

## 5. Evaluation
測試配置：36 個候選數值特徵、SelectKBest k=20、seed=42。
- MAE: 22452.07 USD
- MSE: 907623872.42 USD²
- RMSE: 30126.80 USD
- R²: 0.8144
- 95% split conformal 測試集涵蓋率：96.23%
- 平均區間寬度：136081.82 USD

請從網站實際執行後截取實際值與預測值、殘差圖、預測區間、特徵選擇比較圖，插入本章。上述結果為指定設定的測試值，其他設定應以實際運算為準。

## 6. Deployment
Next.js + TypeScript 前端、FastAPI Python API、Pandas / Scikit-learn，部署到 Vercel。提供 CSV 上傳、研究問題下拉選單、模型參數調整、JSON / CSV 結果匯出。

## 7. 研究限制
數值型欄位限定；單次請求處理；非所有任意 CSV 都符合迴歸需求。迴歸係數不能直接解釋為因果。95% 區間為邊際涵蓋率，須符合資料可交換性假設。

## 附錄：AI 協作紀錄
對話脈絡：從 Kaggle 主題選擇、決定 Ames Housing、確認 13–39 特徵限制、決定兩個研究問題、CSV 任意上傳、Vercel + Python 技術選型，到專案實作。**完整逐字對話請由 ChatGPT 匯出原始紀錄後附入；本檔不是逐字稿。**
