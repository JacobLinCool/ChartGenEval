# 01 評估協定篇大綱（v2，2026-07-10 重構）

**Working title**: *Calibrate Before You Rank: Corruption-Validated Quality
Metrics for Rhythm-Game Chart Generation*

**這篇論文「只」講評估。** 候選場館：ISMIR（主）、TISMIR、arXiv 先行。
人類效度**只放 future work**（04 篇的資料未來回填，不是本篇依賴）。

**定位修正（2026-07-10，使用者指示）**：
1. **不引入我們自己的生成模型**——應用節改為以本方法評估「既有已發表
   模型」（TaikoNation 為首選：taiko 域、開源；DDC/GenéLive 視詞彙
   相容性與可跑性）。主證據（劣化探針施於官方譜）本就不依賴任何模型，
   不受影響；自家消融階梯與 mx 系統的所有內容從正文撤下（可留作
   內部驗證，不進論文）。順帶消滅「自評偏誤」攻擊面
2. **Artifact＝可安裝的 Python 套件**：新 repo ~/Documents/GitHub/
   ChartGenEval——`pip install chartgeneval`＋最小重現實驗碼
   （校準→探針→評估→產表圖）。timing 家族的格線來源做成可插拔
   （metadata／librosa／madmom／任意 beat tracker），論文中改用
   off-the-shelf tracker 跑 C2（去除對我們 beat 模型的依賴，
   需小規模重跑）

**應用節被評對象（2026-07-10 偵察定案）**：
- **Mapperatorinator v32**（主）：MIT、HF 權重 41k 下載、原生
  taiko（gamemode=1）、條件式（difficulty/style）、現代 stack
  ——~1 天端到端。論文中定位為「widely-used community system」
  （非同儕評審，須明寫）
- **TaikoNation**（次，加分）：FDG 2021 同儕評審＋taiko 專用先行
  系統，repo 內含 TF1 checkpoint——環境成本 1–2 天（py3.7+TF1+
  essentia），審稿人要「已發表系統」時的正牌答案
- 兩者輸出皆 osu! Mode 1 `.osu`，共用一個 hitsound-flag→don/ka
  解析器（映射已在 TaikoNation 自帶的 osu_to_bin.py 驗證）
- **跨遊戲組（2026-07-10 修正：不降級，改測適用子集）**——使用者
  正確指出詞彙不相容只殺語料校準家族（3 語法/5 流形/約束帶），
  時值對齊與音樂耦合家族**遊戲無關**、結構家族 raw 層對任意 token
  字母表可算：
  - DDC → 用維護中的 ddc_onset（PyTorch 移植＋權重）取 placement
    onset 序列 → 家族 1/2
  - GenéLive!（MIT、model.pth 在 repo）→ 對我們 40 首音訊生成
    （StepMania 域）→ 家族 1/2＋結構 raw
  - AutoOsu（mania、權重在 repo）→ 同上
  - 三者皆小模型 **CPU 推論**，不與 GPU 排程衝突
  - **生成已完成（2026-07-10）**：三系統 40/40 零失敗，統一 schema 落地
    eval/ext_{ddconset,autoosu,genelive}/samples/（provenance 含 BPM 表
    與 driver 腳本）。發現：GenéLive 公開版無 symbol 模型
    （官方 stub 全放 lane 0）＝ **timing-only baseline**（含五段難度
    密度條件化，單調驗證過）；適用性矩陣其結構欄改「不適用
    （degenerate vocab by design）」而非 raw
  產出「**指標適用性矩陣**」（家族 × 遊戲詞彙）——直接回答審稿人
  「只是太鼓？」：audio-anchored/coupling 零修改跨遊戲、語料家族
  需換語料（scope 陳述而非缺陷）。跨遊戲分數以 raw/z 呈現並明註
  「taiko 帶不適用於他遊戲」

## 核心論證結構（v2 的關鍵改變）

~~用品質階梯的相關係數證明指標有用~~（數字對數字，說服力弱）→
**改為劣化探針（corruption probes）的直覺論證**：

1. **每個維度配一個針對性劣化操作**，施加在官方（人類）譜上——
   方向由建構**在群體層級**保證（兩側帶使 7–14% 個別譜可能靠近帶心，
   已實測揭露），無需 F1 錨、無循環性
2. **劑量反應（敏感性）＝主證據**：劣化強度 {弱, 中, 強} → 目標分數
   單調下降——實證 6/8 通過（ρ −0.33~−0.80）
3. **歸謬（基線在同一套探針下方向反轉）＝Figure 1**：複讀化讓
   structureness/self-BLEU「變好」、平庸化讓 perplexity「變好」而
   band 分數不被騙（抗操縱）——讀者可目視的機制展示
4. **特異性降為「耦合結構」誠實陳述（v2.1 修正）**：對角優勢實測
   不成立（0/8；rank-based 僅 C7 目標居首）——共享 tokenization 使
   時值劣化傳染 pattern 分數，部分是領域真相（節奏遊戲的時值傷害
   本就是 pattern 傷害）、部分是套件限制（全 280 格矩陣公開為
   指標耦合圖）。**不得宣稱「只有它掉」**
5a. **格線參照層級（2026-07-10 使用者修正——寫進 §timing 與套件文件）**：
   (1) **authored grid**（官方譜 timing metadata，不含音符位置）
   ＝benchmark 模式首選，dose-0 相位誤差 0.165ms、C2 dose1
   unsupported 0.0003→0.483 最利偵測器；跨遊戲組的時值評估直接用它
   （40 首的 authored grid 對任何遊戲的生成譜都適用，免音訊估計）；
   (2) **estimated grid**（音訊）＝reference-free 模式（新歌無譜），
   已對 authored 驗證（週期 31/31 精確、相位 ~10ms）；
   (3) **生成譜自宣稱的 grid 永不作對齊參照**（自我一致≠對齊，
   C2 的博弈向量），僅限內部一致性指標。
   1×2 互為 cross-check：嚴重不合＝該歌 metadata 可疑的資料稽核旗標。
   C2 exhibit 改雙列呈現（authored 最利／estimated reference-free
   仍抓到）。

5. **C2 錨定偏移：盲區已閉環（2026-07-10 邊界決定：timing 家族納入
   本篇）**：35 個 chart-only 指標全盲（照舊誠實展示）→ 本篇的
   audio-anchored timing 家族（experiments/timing_integration_v1）
   以估計格線錨定，`grid_phase_offset` 精確回收 +15.08/30.11/60.01ms
   （誤差 <0.2ms）、校準分數 0.542→0.011。beat **估計器**本身仍是
   02 篇貢獻，本篇以元件引用。兩個必須保留的工程誠實註記：
   (a) matching-based signed_offset **本身也是 C2 盲的**（密格重配對）
   ——fixed-grid support maximization 是必要的非顯然設計；
   (b) 估計格線在單首歌可有 ~15ms 相位差（test_00017）使該歌
   unsupported_rate collapse 到 1.0——est-grid unsupported 非 C2
   主偵測器、官方支撐基線要報分佈不報均值
6. **最後**才是應用節：真實系統（消融家族、v1 vs v1.5 世代）評分
   ——meta-eval 降級為應用案例，不再是主證據

## 劣化探針套件（設計稿，待實作定案）

| # | 劣化操作（施於官方譜） | 目標維度/分數 | 應不動的 | 基線的預期洋相 |
|---|---|---|---|---|
| C1 | 時值抖動（±1 格 / ±10–30ms，逐劑量） | timing／tuplet 家族 | pattern、density | — |
| C1s | 稀疏重度離群（0.5/1/2% 音符 ±60ms＝10×deadzone） | timing 尾端（mean 類應全盲） | 其餘 | mean/percentile 全盲；lattice 吸收 72% |
| C2 | 整體錨定偏移（+15/30/60ms 常數） | 錨定/timing | 其餘全部 | 多數基線全盲 |
| C3 | 音色隨機置換（don↔ka，p 遞增） | transition validity／pattern IC | timing、density | entropy 反而上升=「更好」 |
| C4 | 複讀化（以最常見 4-gram 循環填充，保密度） | repetition／variety／boredom | density、timing | **structureness 上升**、Self-BLEU 上升 |
| C5 | 平庸化（LM 最大機率重取樣局部片段） | pattern IC（低於帶） | timing | **perplexity 下降=「更好」** |
| C6 | 密度縮放（×0.5／×1.5／×2 抽刪/複製音符） | density/strain 約束層 | pattern 結構 | distinct-n 被長度拖著走 |
| C7 | 爆發插入（隨機高密叢集） | overload／spike／chaos | 全域統計 | — |
| C8 | 小節洗牌（保局部、毀全局結構） | 結構/repetition 家族 | local pattern（特異性關鍵格） | — |

實測結果（experiments/corruption_probes_v1，run 20260710，
4,175 張、dose-0 與既有 official 分數逐位一致 7,348 值、
對抗驗證 sound-with-caveats、算子逐一逆向核實）：
- **敏感 6/8 ✅**（C2 為設計上的全盲對照；C5 目標「不跌」是抗操縱
  特性，標 n/a 而非 FAIL）
- **特異 ✗**（見上第 4 點——改為耦合結構陳述）
- **歸謬 ✅✅**：C4 structureness 0.096→0.144、self-BLEU 0.860→0.892；
  C5 perplexity 7.98→5.48 而 pattern_ic band 分數 0.892→0.900 不動
- **C5 公平性註記**：劣化用的 LM 與量 perplexity 的 LM 同一顆——
  論文明寫為「in-sample worst-case attack：即使是對 perplexity
  最有利的攻擊也騙不過 band 分數」。**P1 已完成（experiments/lm_provenance_v1）**：
  異 LM 版本（held-half LM 驅動 blandification、full-train LM 量 perplexity）
  perplexity 仍隨劑量下降 7.98→5.55，Spearman(dose,perp) −0.324（in-sample
  −0.339），band pattern_ic 不被驅低（−0.122）——C5 歸謬升級為 cross-LM 證據
- **LM 出處稽核（P1，2026-07-10 實查，推翻既有假設）**：LM 訓練語料（train）
  與受評 official（test）**並非**歌曲層互斥——三把獨立鍵（TITLE／音訊路徑／
  音訊內容 SHA-256，三者完全一致）實查 **25/116 test 歌（21.6%）也在 train**，
  受評 40 首集內 7/40 歌＝29/167 course-chart（17.4%）洩漏。故「held-out by
  construction」**不成立**，改述為「已量測 ~17–22% 歌曲層重疊」。穩健性由 held-half
  LM 佐證：系統排序完全不變（Spearman 1.000）、各 AUC 位移 ≤0.078，故洩漏存在
  但不承重

### Timing 家族尾端修訂與認證（2026-07-10 第二波，
experiments/timing_integration_v1/runs/raw/records/certify_timing_tail_6ms.jsonl，
2,072 records，167 官方譜 × C1/C1s × 3 劑量 × 雙格線）

**設計修訂（使用者定案，知覺優先原則）**：
- deadzone 統一為 **6ms**（與相對失真下限同一 JND 常數；「deadzone 內＝感知
  不到」）；閾值一律知覺文獻錨定，**不用遊戲判定窗**
- 主呈現＝**三階超越率**（>1×/2×/3× deadzone）＋ **clean_rate**
  （配對成功且 deadzone 內 ÷ 全部音符；unsupported 算分母——堵 selection
  effect）；deadzone 殘差 p90/p99 降為輔助診斷；原始（未扣 deadzone）分佈
  不進呈現
- 相對偏差：先以原始 matched error 計算相對失真、再套 θ 包絡（實作原本
  即如此，已加註解固定）
- 聚合協定：逐譜統計 → 跨譜分佈（median＋尾端），絕不 pool-then-mean

**C1 重新認證 @6ms（authored grid）——PASS**：
- dose1（±10ms）從舊制全盲變為違規率 0.40，與理論值
  P(|U(−10,10)|>6)=0.4 精確吻合
- clean_rate 於全部三劑量 **100% 譜嚴格下降**；absolute_violation_rate
  逐譜方向一致率 1.00/1.00/1.00；2x 階於 dose2/3 觸發（嚴重度分級有效）
- magnitude 在 dose3 有吸收回落（0.60→0.43，密格 re-matching），但方向
  保證不受影響——寫入論文時以「方向一致率」為準、不宣稱 magnitude 單調

**C1s 稀疏重度離群——發現結構性盲區（重要負結果）**：
- 位移音符 **72.6% 被密格 re-matching 吸收**為 clean（±60ms 後落在別的
  合法格點 6ms 內）；timing 家族僅在 34–45% 譜上觸發（clean_rate 通道：
  violation↑ 30.5–43.1%、unsupported↑ 5.4–12%）
- mean/percentile 類全盲（median 全 0.0000，dir ≤0.44）——歸謬成立：
  「平均很小」不保證尾端乾淨
- 機制：lattice matching 量「離格程度」不量「錯格位置」——被移到另一
  合法節拍位置的音符，reference-free 格線指標無法與有意音符區分。與 C2
  re-matching 失明同機制的第二實例
- **C1s 家族歸屬（已定案，experiments/corruption_probes_v1/runs/raw/
  records/c1s_full_profile.jsonl，668 records）**：grammar 家族強接
  （逐譜淨方向 d3：ioi_entropy +0.87、pattern_nll +0.86、
  unique/repeat_4gram ±0.85、window_nll_p95 +0.82、invalid_transition
  +0.82；連 d1＝每譜 1–3 顆也有 0.67–0.78）；density/structure 正確
  沉默（0.00–0.02）。機制：錯格位移在格線層被吸收，但必然翻轉相鄰
  IOI 的 token 類別。→ 耦合矩陣 C1s 行：主接收者 grammar（原始診斷層），
  次接收者 timing clean_rate（35–45%）；band 校準分數在微劑量下反應弱
  （pattern_ic_adequacy −0.25）——與角色分層一致（診斷層偵測、band 層
  排序）。**互補性最乾淨展品**：同一劣化對一家族隱形、對另一家族刺眼，
  且雙向有實例（C5：grammar 被騙/band 撐住；C2：chart-only 全盲/timing
  接住；C1s：timing 被吸收/grammar 接住）

**Estimated grid @6ms**：絕對通道被 estimator 誤差淹沒（官方譜 baseline
違規率 97.7%）→ 適用性矩陣定案：**絕對通道＝authored grid 專屬；estimated
tier 僅相對通道可信**（C1s 於 est 上唯一有反應者＝relative_violation
dir 0.79）

**Authored grid 的建構教訓（2026-07-10，變拍歌事故→§timing 素材）**：
初版認證腳本把 authored grid 扁平化為 downbeat 列表（單一拍號推斷），
變拍/變速官方譜被誤判 31% 違規（《誘惑》3/4↔4/4、《シン・ゾンビ》5/16
＋多段變速；原始建構下兩者皆 0.000ms 完美在格）。修正＝逐小節 TJA
segments（每小節自帶拍號）＋chart_duration=max(音訊, 末音符+1s)；修正後
逐位重現參考建構（n_anchors 一致），官方 baseline 最差譜歸位 0.970。
結論入文：**authored tier 必須用完整逐小節 timing metadata——扁平化
downbeat 列表會弄丟拍號資訊而誤錨變拍譜**。套件 `timing.compute` 現支援
`grid["segments"]`（優先於 downbeats）。全部認證數字已以修正版重跑，
結論不變（C1 100% 嚴格降 ×3；C1s 吸收 72.6%）

### Clean-40 外部系統 timing 結果（2026-07-10 初步，
experiments/timing_integration_v1/runs/raw/records/ext_clean_timing_20260710.jsonl，
980 records；clean test 前 40 canonical 歌、clean 校準時代首批數字；
Mapperatorinator 待 GPU 批次收官後補）

| system（authored grid，跨譜 median） | clean | viol1× | unsup | 相對viol | grid_phase\|off\| | 逐譜 worst |
|---|---|---|---|---|---|---|
| official | 1.000 | 0.000 | 0.000 | 0.000 | 0.00ms | 0.955 |
| ddc_onset | 0.729 | 0.245 | 0.027 | 0.311 | 3.50ms | 0.229 |
| taikonation | 0.581 | 0.325 | 0.139 | 0.414 | **49.75ms** | 0.344 |
| genelive | 0.526 | 0.398 | 0.102 | **0.099** | 12.67ms | 0.046 |
| autoosu | 0.380 | 0.456 | 0.233 | 0.232 | 16.45ms | 0.000 |

三通道分解讓每個系統得到**機制診斷**而非單一分數（應用節主敘事）：
- **ddc_onset**「audio-follower＋偵測散佈」：錨定準（3.5ms）、6–12ms 帶
  抖動（viol 24.5% 但 2×/3× 階≈0）
- **genelive**「內部一致、相位錯錨」：相對通道乾淨（0.099，五難度中
  beginner 0.000）但 grid phase 偏 12.7ms——**C2 教訓在真實系統上重演**：
  matching-median 只顯示 0.49ms（re-matching 吸收），fixed-grid witness
  抓到 12.67ms。真實已發表系統上的活例證，比探針更有說服力
- **taikonation**「訊框量化雙重症」：相位不相干（49.8ms）＋音程抖動
  （0.414）
- **autoosu**「偏移＋散佈＋災難尾」：35% 譜 bias>6ms、逐譜 worst=0.000
- 全系統 2×/3× 階≈0：偏差集中在 6–12ms 可感知帶，粗大離群走 unsupported
  通道——三階＋unsupported 的分工實測成立
- estimated grid 欄相對通道與 authored 一致（genelive 0.109 vs 0.099）——
  相對通道 grid-robust 的系統級佐證

## 目標貢獻（v2.1）

1. **劣化驗證的指標套件**：每個維度都有「攻擊它、它就掉」的劑量反應
   證據＋band 分數的抗操縱性；特異性以耦合矩陣誠實公開（非正交是
   實測限制，不是宣稱）
2. **兩層協定**：raw 診斷 ＋ corpus 校準帶（p10–p90、course 條件化）；
   三層角色分級由劣化實驗實證指派
3. **基線失效的建構性演示**：perplexity／structureness／entropy／
   distinct-n 在針對性劣化下方向反轉——借用指標的風險從統計主張
   變成可目視的機制展示
4. 應用案例：真實系統評分（消融家族含設計失效、跨世代進步）

## 現有資產（可重用）

- `scripts/validate_metrics.py`：v1 時代已做過「腐化響應」驗證
  （energy_align 抓到「差半格」）——同一思想的前身，探針實作可參考
- `src/softchart/train.py` 的 `corrupt_events`（DPO 負樣本：jitter＋swap）
- 04 篇的操縱設計（時值劣化/無 plan/偏 30ms）——與 C1/C2 同源，
  未來人類效度可直接對接
- experiments/chart_quality_metrics_v1（校準帶＋指標管線）、
  experiments/metric_baselines_v1（基線實作＋公平性規範）——
  meta-eval 結果降級為應用節素材

## Release Gate（v2 修訂）

承重（不做主張不成立）：
- [x] **P1** pattern LM 出處稽核＋異 LM 穩健性（experiments/lm_provenance_v1）：
  實查 train/test 有 ~17–22% 歌曲層重疊（非 held-out by construction）；held-half
  LM 重算 pattern 家族——系統排序 Spearman 1.000、AUC 位移 ≤0.078、C5 歸謬 cross-LM
  仍成立。結論措辭（驗證員校準）：**在 50% 語料擾動下排序不變（held-half 仍含
  13/25 洩漏歌），故洩漏對排序不承重；嚴格 leak-free LM（剔除 25 首）為未跑的
  更強版本**——非「已證無洩漏」
- [x] **timing 家族整合（原 C2 盲區）**：experiments/timing_integration_v1——
  估計格線錨點與 metadata 一致（週期 31/31 精確、相位 ~10ms）、C2 閉環、
  耦合矩陣 timing 列乾淨（C3/C4/C5 恰為 0.0）、決定性重跑逐位一致；
  C1 劑量反應以 relative_interval（ρ +0.67）/relative_error（ρ +0.45）為
  正字標記，absolute_offset（ρ −0.41，matcher re-snap）與 unsupported
  （dose1 後非單調）為已註記的 matcher 交互 artifact——論文寫劑量表時
  逐子指標帶號呈現，不得聚合掉
- [x] **C 套件實作＋執行**（corruption_probes_v1，見上）
- [ ] **P3** 解碼對照併入 C5（平庸化＝受控版的 greedy 效應）
- [ ] **P8** artifact 實際釋出（「We release」兌現）
- [ ] 系統描述小節（應用節自足，不依賴未發表的 03 篇）
- [ ] 指標定義表（tokenization／LM 階數／band 衰減／F1 容忍度）

打磨：
- [ ] **P4** seed 穩健性、**P5** level 條件帶
- [ ] **P7** Figure 1（=特異性矩陣熱圖）＋劑量曲線圖、作者欄、書目核實
- [ ] boredom heuristic 修復或明文降級
- [ ] 跨世代比較的解碼混淆 caveat（或改用 base_greedy 同解碼樣本）
- [ ] proxy＝screening 聚合的表述統一

## Suite v2 候選指標（2026-07-10，experiments/metric_candidates_v1，
對抗驗證 sound-with-caveats、全部統計獨立重算吻合）

五視角發想 26 個 → 評審選 8 個實作 → 預註冊 gauntlet 考核（劣化探針
對答案＋AUC＋長度正交＋新軸性）→ 裁決：

**ADOPT（v2 核心 ranker）**：
- `official_manifold_gap`——距官方流形距離（反平庸軸，攻病理 2）：
  8/8 探針方向正確、平庸化使 gap **上升**（6.72→10.30，抗操縱）、
  C2 精確不動、與全部既有 ranker |r|≤0.30。
  ⚠️ 抗平庸目前僅以 color-flatten proxy 證明，真 LM-C5 重測前
  主張保持「suggestive」
- `call_response_reciprocity`——呼應句式（長程對話結構，攻病理 6）：
  AUC 0.693 且 **official vs 最佳健康系統 0.681**（殘餘品質軸）；
  primary leaf 用 C3-穩健的 reciprocity 子葉（rate 有 C3 漏）
- `density_energy_response`——密度跟隨能量（音訊耦合）：全套件最獨立
  （max|r|=0.11）、C6 等比縮放不變（設計特性）
- `energy_peak_support_rate`——音符落在能量峰上（音訊耦合，
  screening 級）：at_hit_z 子葉是免估計格線的 C1/C2 見證者
  （C2 ρ −0.83）；須按時長正規化（|ρ|len 0.77）

**HOLD 為診斷**：`motif_development_index`——全場最強分離器
（AUC 0.754、official 0.76 ≫ 最佳生成 0.56）但**可被操縱**
（部分複讀與平庸化都使它上升——V=0.5 鐘形峰漏洞），加 V→0
collapse 項並重測前不得當 ranker。`syncopation_contour`（弱分離，
但 C3 精確不動證明時值-pattern 可解耦——病理 5 的反例）。

**REVISE**：`grid_accent_divergence`（C2/C7 漏＋長度耦合，
需 re-anchor＋長度控制）。**DROP**：`self_similarity_form_score`
（在自己的兩個目標探針上反向——正是它聲稱要修的 self-BLEU 病理）。

待辦（便宜）：'mid' 預期類別要拆帶號版本重報（28% 的 pass 靠它，
4 個是「劣化使讀數變好」也算過）；manifold_gap 真 LM-C5 重測；
MDI 修復重測；reciprocity 升 primary。

## 風險

- 「劣化是人造的」→ 劣化操作與 04 篇的人類操縱同源（時值抖動/錨偏），
  未來人類資料直接驗證同一批探針；且應用節有真實系統的野外失效對照
- 「探針挑軟柿子」→ 特異性矩陣全格公開（含不動的格）；基線與我們的
  分數跑完全相同的探針
- 與 02/03/04 不重疊：02 佔估時鐘、03 引用分數當 gate、04 佔人類感知
