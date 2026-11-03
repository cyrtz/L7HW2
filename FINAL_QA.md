# Regression Lab — 最終驗收紀錄（2026-11-03）

## 已驗證（本地 Python）
- `PYTHONPATH=. pytest -q tests/test_engine.py`：1 passed。
- FastAPI `GET /api/health`：HTTP 200，`status=ok`。
- FastAPI `GET /api/default`：HTTP 200，預設資料摘要成功回傳。
- FastAPI `POST /api/analyze`：HTTP 200，36 個數值候選特徵、SelectKBest 20；RMSE=30,126.796584147512。
- 修正版 `api/index.py` 於回傳前清理非有限浮點數，避免 NaN JSON 500。
- CRISP-DM 報告已產生、渲染檢查，包含 4 張模型圖與 2 張使用者提供的網站截圖。

## 尚需在正式 Vercel 環境確認
- `npm install && npm run build`（此執行環境未進行 Next.js 正式 build）。
- 兩個研究問題的實際操作流程、CSV/PNG 下載、任意 CSV 上傳、瀏覽器 E2E。
- 正式站點的 API 狀態、函式運算時間與大小限制。

## 注意
- `results.json` 與報告數據基於內附 `data/train.csv`，36 個數值候選特徵，SelectKBest k=20，random_state=42。
- 使用者網站截圖的勾選欄位可能不同，因此畫面數值不必等於報告的本地重現數值。
- 最佳 CV 組合 RFE 25 是在訓練集交叉驗證得到，**不是**其獨立測試表現。
- 完整 AI 對話逐字稿未納入；報告附錄提供可核對的歷程摘要。

## 部署差異
如果線上已是 V2 且已套用 NaN 修正，本套件無需再次覆蓋程式；僅提供已整合版本備份。
如果要對照，V2 到本版的必要後端變更為 `api/index.py` 的 `json_safe` 處理；`vercel.json` 保留有效 rewrite。
