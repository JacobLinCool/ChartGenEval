# ChartGenEval 發佈 Checklist

目標狀態：`chartgeneval` 套件可安裝可重現、評估論文（arXiv → ISMIR）與
artifact 同步發佈。論文 LaTeX 原始檔與研究紀錄在本 repo `paper/`
（`paper/paper.tex`＋`paper/OUTLINE.md`，2026-07-10 自 SoftChart 遷入）。

## A. 實驗／分析（數字凍結前）

- [x] **既有模型評估（taiko 域）**（2026-07-11，clean-40）：
      Mapperatorinator 170 譜「語言對、難度錯（easy 超帶 1.5–2.5×）、
      相位歪（26ms）」；TaikoNation 40 譜「密度對、語言錯、時鐘歪」
      ——互補失效對，2021→2024 clean rate 同分而機制不同
- [x] **跨遊戲組生成**（2026-07-10）：ddc_onset／GenéLive!／AutoOsu
      三系統 40/40 零失敗（CPU）；GenéLive 公開版實測為 timing-only
      （sym stub 全 lane 0，密度條件化真實）——適用性矩陣照此標註
- [x] **跨遊戲組評估（timing）**（2026-07-11 終版）：六系統雙格線
      1,320 records；ddc=管線對照組（2.5ms）；適用性矩陣熱圖待組裝
      （B 組應用節件）
- [x] **suite v2 修理票包**（2026-07-11，fixes_20260710.md，五 commits）：
  - [x] `official_manifold_gap` 真 LM-C5 重測＋φ v2（v1 低劑量可操縱
        已修；ρ −0.317、逐譜 84.4%、C1–C8 零迴歸）
  - [x] `energy_peak_support` run-head 時長正規化（長度 ρ 0.77→0.30、
        C6 反應保持、AUC 0.728）
  - [x] `call_response` → reciprocity v2 primary（C3 漏轉正、C6 歸正、
        8/8 pass；舊 leaf ABAB=0.0 缺陷一併修）
  - [x] boredom → **boredom_v2**（TaikoNation 反例 1.000→0.458、
        C4 ρ −0.932、AUC 0.860 全候選最強）
  - [x] MDI → **DROP**（修復版更糟＋雙形式被攻擊者打穿＋構念由
        reciprocity v2 覆蓋）
- [x] **gauntlet 'mid' 預期類別拆帶號重報**（2026-07-10，
      mid_rereport.md）：嚴格版 33/48=69%（原 74%）；mid 拆為
      directional-down ×5（post-hoc 輔助）/flat ×6/mild-counter ×3
      （ε 級須揭露）；ADOPT 四指標全存活；dose 計數以重報為準
- [x] **C2 雙列呈現＋witness 修復**（2026-07-10，c2_flip_dual.csv）：
      witness 依層分派（authored=piecewise lattice 三尺度、estimated=
      常數合成）→ **兩層×三劑量 100% 逐譜回收**（authored Δ 精確
      15/30/60.00、IQR 0.00）；matching 雙層皆盲；wrap 缺陷已修除，
      無需任何範圍註記；跨遊戲組時值評估已用 authored grid ✓
- [x] **timing 尾端修訂＋認證**（2026-07-10）：deadzone 統一 6ms（知覺
      優先，非遊戲判定窗）、三階超越率（1×/2×/3×）、clean_rate、C1s
      稀疏離群探針。C1@6ms 認證 PASS（clean_rate 100% 譜嚴格降 ×3 劑量、
      dose1 違規率 0.40=理論值）；**C1s 發現結構性盲區**（72.6% 被密格
      re-matching 吸收，timing 家族僅 34–45% 譜觸發；segments 版
      authored grid 重跑後數字，結論不變）——舊 10ms 紀錄
      不可與新 6ms 紀錄混讀（P9 全面重跑後統一）
- [x] **C1s 家族歸屬**（2026-07-10）：grammar 家族強接（淨方向 d3：
      ioi_entropy +0.87、pattern_nll +0.86、4gram ±0.85、
      window_nll_p95 +0.82），density/structure 沉默——耦合矩陣補
      C1s 行；「timing 被吸收/grammar 接住」與 C5、C2 構成互補性
      雙向展品組（進 §complementarity）
- [x] boredom heuristic → boredom_v2 修復完成（見上）
- [ ] （選配）seed／解碼穩健性小跑

## B. 論文寫作（英文版：本 repo paper/paper.tex）

- [ ] §3 改寫為 **5＋1 家族 × 3 層角色** 矩陣呈現＋探針覆蓋表
      （互補性的證明）
- [ ] §timing 寫入**格線參照層級**（authored > estimated > 永不用
      自宣稱 grid）＋fairness 區分（禁音符位置、可用 timing metadata）
- [ ] **指標定義表**（tokenization／LM 階數／band 衰減／F1 容忍度）
      ——自足性
- [ ] 應用節：撤下自家模型內容 → 換既有模型結果＋適用性矩陣
- [ ] suite v2 四指標寫入對應家族（待 A 組修復完成）
- [ ] 系統性敵意審稿輪（論證鏈→主張校準→敵意→句法→一致性 多 pass）
- [ ] 中文版（`paper/PAPER.zh-TW.md`）與英文版同步更新
- [ ] 圖選定與打磨：新圖組已產（figA 互補性／figB 三階堆疊／figC
      耦合+C1s 欄，`experiments/make_paper_figures.py`）——與 fig1/fig2
      的取捨定案＋6 頁排版；figB 待系統重評估後換終版數字

## C. 套件發佈（本 repo）

- [x] 保真驗證通過（2026-07-10，verdict **sound**）：乾淨 venv 安裝、
      34 pytest 全過、三個 golden 錨 max diff = 0.0（3,528＋28,350 值＋
      校準逐位元）、探針決定性雙跑一致
- [x] **視覺化庫**（2026-07-10）：`chartgeneval.plots`——L0–L3 視覺協定
      （profile 熱圖含 n/a＋constraint 玻片／雷達僅 demo 不填色、timing
      三階堆疊條＋逐譜分位帶、淨方向劑量圖、official-manifold 散點、
      單譜診斷條）；`[plots]` extra、10 項煙霧測試（共 44 過）
- [ ] 驗證員 MINOR 修復：pytest 入 [dev] 說明（README 已補）、
      `--limit` 貫穿 LM 建置（現全量 ~9 分鐘）、
      **P9 後以 clean split 重校準並替換隨附 artifact**（同時消除
      「預設資料集≠golden 語料」的陷阱，README 已先加警語）
- [ ] **bpm≤0 防呆進 grammar 家族**（新指標已循 NaN 哨兵契約；
      殘餘：既有 grammar 家族在 bpm 無效時仍會出垃圾 token——
      `evaluate_chart_quality` 補 NaN 短路；腳本層 --bpm-map 已補）
- [ ] README 補：格線參照層級與適用時機、自宣稱 grid 警告、
      指標適用性矩陣（哪些家族需要哪種輸入/語料）
- [x] **suite v2 五指標進套件**（2026-07-11，commit 4ab1f9a）：
      manifold_gap(φv2)→gap、reciprocity v2＋boredom_v2→structure、
      density_energy＋energy_peak(run-head)→coupling；等價驗證
      **1,302 key 逐位 0.0**（11 合成＋12 真實譜）；pytest 60 過
      （44 舊＋16 新）；docstring 含知覺錨＋gauntlet 數字；無效輸入
      一律 NaN 哨兵（兼銷 bpm≤0 防呆項的套件層行為）
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
- [ ] checklist.md 與 paper/{OUTLINE.md,PAPER.zh-TW.md} 是否留在公開 artifact repo
      （驗證員建議公開發佈前移出或 gitignore；作者拍板）
- [ ] Mapperatorinator 非同儕評審的定位措辭確認
      （"widely-used community system"）

## 發佈順序建議

1. A 組全綠 → 數字凍結
2. B 組寫作＋敵意審稿輪
3. C/D 收尾（repo 公開、URL 進文）
4. E 拍板 → 編譯終版 → arXiv
