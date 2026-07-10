# ChartGenEval 發佈 Checklist

目標狀態：`chartgeneval` 套件可安裝可重現、評估論文（arXiv → ISMIR）與
artifact 同步發佈。上游研究紀錄在 SoftChart repo 的
`papers/01-evaluation-protocol/OUTLINE.md`。

## A. 實驗／分析（數字凍結前）

- [ ] **既有模型評估（taiko 域）**：Mapperatorinator v32＋TaikoNation
      對 40 首測試歌生成 → 全 profile 評估（管線 agent 執行中，
      GPU 等 v1.6 佇列收官）
- [ ] **跨遊戲組**：ddc_onset／GenéLive!／AutoOsu（CPU 推論，
      管線 agent 執行中）→ 只評遊戲無關家族，產出
      **指標適用性矩陣**（家族 × 遊戲詞彙）
- [ ] **suite v2 納入前的三件修復**：
  - [ ] `official_manifold_gap` 用真 LM-C5 重測（現只有 color-flatten
        proxy 證據）
  - [ ] `energy_peak_support` 按有效時長正規化（|ρ| vs 音符數 0.77 → <0.3）
  - [ ] `call_response` 以 reciprocity 子葉為 primary（rate 有 C3 漏）
- [ ] **gauntlet 'mid' 預期類別拆帶號重報**（28% pass 靠它、4 個
      「劣化使讀數變好」也算過——不修不得引用 dose 計數）
- [ ] **C2 雙列重跑呈現**：authored grid（最利偵測）＋estimated grid
      （reference-free）並排；跨遊戲組時值評估改用 authored grid
- [ ] （選配）boredom heuristic 修復或維持 diagnostic 降級說明
- [ ] （選配）seed／解碼穩健性小跑

## B. 論文寫作（英文版：SoftChart/papers/01-evaluation-protocol/paper.tex）

- [ ] §3 改寫為 **5＋1 家族 × 3 層角色** 矩陣呈現＋探針覆蓋表
      （互補性的證明）
- [ ] §timing 寫入**格線參照層級**（authored > estimated > 永不用
      自宣稱 grid）＋fairness 區分（禁音符位置、可用 timing metadata）
- [ ] **指標定義表**（tokenization／LM 階數／band 衰減／F1 容忍度）
      ——自足性
- [ ] 應用節：撤下自家模型內容 → 換既有模型結果＋適用性矩陣
- [ ] suite v2 四指標寫入對應家族（待 A 組修復完成）
- [ ] 系統性敵意審稿輪（論證鏈→主張校準→敵意→句法→一致性 多 pass）
- [ ] 中文版（本 repo `PAPER.zh-TW.md`）與英文版同步更新
- [ ] 圖打磨（fig1 圖例位置、fig3 字級）＋6 頁排版

## C. 套件發佈（本 repo）

- [x] 保真驗證通過（2026-07-10，verdict **sound**）：乾淨 venv 安裝、
      34 pytest 全過、三個 golden 錨 max diff = 0.0（3,528＋28,350 值＋
      校準逐位元）、探針決定性雙跑一致
- [ ] 驗證員 MINOR 修復：pytest 入 [dev] 說明（README 已補）、
      `--limit` 貫穿 LM 建置（現全量 ~9 分鐘）、
      **P9 後以 clean split 重校準並替換隨附 artifact**（同時消除
      「預設資料集≠golden 語料」的陷阱，README 已先加警語）
- [ ] README 補：格線參照層級與適用時機、自宣稱 grid 警告、
      指標適用性矩陣（哪些家族需要哪種輸入/語料）
- [ ] suite v2 指標隨 A 組修復同步進套件（含 gauntlet 通過紀錄）
- [ ] LICENSE 定案（現為 MIT 佔位，待作者確認）
- [ ] 版本號 0.1.0＋CHANGELOG
- [ ] （選配）CI（pytest＋pip install 煙霧測試）
- [ ] （選配）PyPI 發佈 vs 只走 `pip install git+`——決策
- [ ] GitHub repo 公開＋URL 寫進論文（兌現「We release」）

## D. 資料與 artifact

- [x] `taiko-1000-parsed-clean` 已釋出（public＋gated manual，
      813/80/120 群組切分、跨切分 SHA 重疊=0）
- [ ] 論文用 records（探針 JSONL、meta-eval 表、耦合矩陣）上傳
      （隨 repo 或 HF dataset 附件——決策）
- [ ] 校準 artifact 版本標記（band 統計對應 clean split 或原 split
      要寫清楚；終版建議以 clean split 重校準一次）

## E. 決策（需作者拍板）

- [ ] 作者署名／單位（paper.tex 現為 TODO 佔位）
- [ ] arXiv 分類（cs.SD＋eess.AS？）與授權（CC BY？）
- [ ] TaikoNation 無 license 宣告——推論評估的引用措辭確認
- [ ] checklist.md 與 PAPER.zh-TW.md 是否留在公開 artifact repo
      （驗證員建議公開發佈前移出或 gitignore；作者拍板）
- [ ] Mapperatorinator 非同儕評審的定位措辭確認
      （"widely-used community system"）

## 發佈順序建議

1. A 組全綠 → 數字凍結
2. B 組寫作＋敵意審稿輪
3. C/D 收尾（repo 公開、URL 進文）
4. E 拍板 → 編譯終版 → arXiv
