# 先校準，再排名：以劣化探針驗證的節奏遊戲譜面品質指標

> 英文版（投稿版）：SoftChart repo `papers/01-evaluation-protocol/paper.tex`。
> 本文為中文對照版，隨英文版同步更新；標 ⏳ 處為進行中實驗，數字凍結後回填。
> 最後同步：2026-07-10。

## 摘要

節奏遊戲譜面是設計出來的創作物，同一首歌有許多合法譜面；然而生成譜面至今
仍以借來的指標評估：對單一參照譜的 F-measure、語料語言模型（LM）的
perplexity、以及 entropy、distinct-n、自相似度等無參照統計量。我們用「測試
煙霧偵測器就放受控的火」的方式驗證指標：對 167 張人類（官方）譜施加八種
針對性劣化、各三個劑量（共 4,175 張評分變體），發現借來的指標以肉眼抽查
看不見的方式失效——把譜面壓平向 LM 眾數模式時，n-gram perplexity 反而
**改善 31%**；複讀化崩潰使自相似結構度**上升 50%**；而我們測試的全部 35 個
chart-only 指標對「整譜對音訊平移 60ms」這種在遊玩中致命的損壞**完全全盲**。
一個兩層協定修復了這些失效：raw 診斷量對官方譜語料的每難度目標帶
（p10–p90）正規化，加上以格線錨定的時值家族——六個有校準目標的探針
全部呈單調劑量反應（Spearman ρ −0.33 至 −0.80）、band 分數抵抗騙過
perplexity 的攻擊、fixed-grid 時值偵測器將注入的 15/30/60ms 平移回收至
0.2ms 以內。我們釋出探針、校準 artifact、全部逐譜紀錄，以及可安裝的
`chartgeneval` 評估套件。

## 1. 引言

節奏遊戲的自動譜面生成——把音訊映射為可遊玩的定時音符序列——已從 onset
偵測管線進展到條件式序列模型（DDC、TaikoNation、GenéLive!、TCP）。評估
卻沒有跟上。譜面是創作物而非標註，同一首歌容許多種合法譜面；領域的主流
工具「對單一官方譜的 F-measure」對每一種合法變體都同等扣分，而且對沒有
譜面的新歌完全無法計算。自然的補充是無參照測量，於是研究者向符號音樂與
文字生成借指標：LM perplexity、pattern 空間一致度、事件 entropy、
distinct-n、Self-BLEU、特徵分佈比較、自相似結構度。

本文問的是：**這些借來的指標到底量不量得到譜面品質？** 我們用一種不需要
聽測的驗證方法回答——**針對性劣化（targeted corruption）**。從人類官方譜
出發，施加八種損壞算子（時值抖動、整體錨定偏移、音色置換、複讀化、LM
平庸化、密度縮放、爆發插入、小節洗牌），各三個劑量。在群體層級，劣化
人類譜不可能讓它變好，所以可用的品質指標必須朝正確方向反應、且反應隨
劑量增長。指標驗證由此成為完全自動化、可重現的程序——不需要評分者、
不需要模型訓練，且可原樣套用於任何新提出的指標。

考核給出一個帶著建設性核心的負結果。借來的指標**結構性失效**（§5）：
perplexity 獎勵它該懲罰的平庸、結構度把退化的重複誤認為形式、多個統計量
是譜長的近似代理、而全部 chart-only 指標對整體音訊錯位全盲。但底層的
raw 量並非無用；缺的是**校準與接地**。我們證明：(i) 把每個診斷量對官方
語料的每難度目標帶正規化，就恢復了一致的「越高越好」方向並抵抗博弈；
(ii) 時值家族錨定在外部格線上（見 §4.3 的參照層級），以零參照譜的代價
關閉錯位盲區。

**貢獻**：

1. **劣化探針考核 harness**：八個劑量控制的損壞算子＋驗收準則（單調劑量
   反應、抗博弈、公開耦合結構）＋公平性規範——任何未來指標都能套用（§4）
2. **14 個借用指標的結構性失效分析**：含兩個歸謬展示（平庸化使 perplexity
   變好；複讀化使結構度變好）與 chart-only 指標的錨定偏移全盲結果（§5）
3. **兩層、無參照譜的評估協定**：raw 診斷＋語料校準帶分數＋實證指派的
   角色分級，並以格線錨定時值家族延伸——fixed-grid 偏移偵測器以次毫秒
   精度回收注入平移（§3、§5.3）
4. **釋出 artifact**：探針實作、校準帶、全部 4,175 筆逐譜紀錄、
   `pip install chartgeneval` 套件（本 repo）、以及歌曲層群組切分的資料集

## 2. 相關工作

**譜面生成與其評估。** DDC 以對參照譜的 F-measure 評 step placement、以
LM perplexity 評 step selection；後續系統沿用此模板。TaikoNation 增加了
對人類譜的 pattern 一致度測量。這些指標從未被受控損壞考核過。

**生成音樂評估。** Yang & Lerch 比較生成集與參照集的特徵分佈；MusPy 普及
了符號統計量；Jazz Transformer 引入自相似矩陣的結構度指標。這些是集合
層級或方向模糊的量；我們考核的是它們被當作逐譜品質訊號時的行為。

**指標批判。** Theis et al. 證明了似然與樣本品質可任意背離；我們的平庸化
結果是譜面域的語料 LM 尖銳實例。NLG 領域記錄了 distinct-n 與 Self-BLEU
的病理。與純批判不同，我們提供修復；與人類錨定的指標研究不同，我們的
harness 不需要評分者。

## 3. 兩層評估協定

**資料。** 1,155 張 taiko 風格譜面。歌曲以音訊內容與正規化標題群組化，於歌曲層切分為 813/80/120
（train/validation/test），任何歌曲不跨切分出現
（`taiko-1000-parsed-clean`）。
校準用訓練切分的 4,361 張官方 chart-course；評估用 40 首保留歌全難度
（167 張）。

**Layer 1：raw 診斷。** 每張譜是定時、定型的事件序列。我們計算帶原生
單位、**不宣稱方向**的量：音符密度與局部密度百分位、IOI entropy、變色率、
n-gram LM 負對數似然（trigram、訓練於校準語料）、未見轉移率、
獨特/重複 4-gram 率、視窗化 LM surprisal 百分位。這些**刻意與借用指標
重合**：考核比較的是正規化策略，不是測量選擇。

**Layer 2：校準帶分數。** 對每個診斷量、每個難度 course，在校準語料上
估計目標帶 [p10, p90]。帶內得 1、帶外平滑衰減；誤差型量用指數懲罰。這
以建構解決方向模糊——「更多 entropy」無所謂好壞，「entropy 落在人類
譜師於此難度使用的範圍內」則定義良好——且在歌曲層級無參照：為新歌評分
只需要固定的語料校準。

**角色分級（由考核實證指派，非先驗）。** *Rankers* 對損壞有反應且能分離
系統（pattern-IC、transition validity、repetition/variety adequacy、
rhythm complexity）；*Constraints* 對所有合理譜面近飽和、只抓粗暴違規
（density、strain、overload）；*Diagnostics* 有訊號但校準未熟（現行
boredom heuristic、超敏感的 rare-n-gram）。我們報告**分數 profile 而非
單一總分**：軸之間的權衡權重正是該留給人類資料的判斷——且 §5.4 顯示
各家族的盲區模式互斥，任何加權總分都會摧毀「哪個軸壞了」的資訊。

**五＋一家族與互補結構。** Rankers 依錨定源分為五家族：

| 家族 | 錨定於 | 代表 leaf | 量什麼 |
|---|---|---|---|
| 1 時值對齊 | 外部格線 | grid_phase_offset、unsupported、interval distortion | 與音樂時間格的關係 |
| 2 音樂耦合 ⏳ | 音訊能量 | energy_peak_support、density_energy_response | 音符/密度呼應音樂內容 |
| 3 局部語法 | 語料帶 | transition_validity、pattern_ic | 局部片語像不像人寫 |
| 4 全局結構 | 譜內＋語料 | repetition/variety、call_response_reciprocity ⏳ | 形式、對話、發展 |
| 5 人類差距 ⏳ | 語料流形 | official_manifold_gap | 整體離人類分佈多遠（單向、反平庸） |
| ＋約束層 | 語料帶 | density/strain/overload | 只做 gate 不排名 |

（⏳ 家族 2/4/5 的新指標已通過考核裁決，納入前的三件修復進行中，
見 checklist.md A 組。）

### 3.1 格線參照層級（時值家族的錨定）

時值對齊需要一個外部格線，來源有嚴格的優先序：

1. **Authored grid**（官方譜的 timing metadata：BPM/offset/barlines，
   **不含音符位置**）——benchmark 模式首選。譜師的 barline 是毫秒級的
   authored 時間戳：dose-0 相位誤差 0.165ms，對 C2 的偵測最鋒利
   （unsupported rate 0.0003→0.483 @+15ms）。公平性註記：禁用官方音符
   位置、可用官方 timing metadata——這個區分比籠統的 reference-free
   更精確
2. **Estimated grid**（音訊 → 任意 beat tracker，套件提供可插拔
   adapter）——reference-free 模式（新歌無任何譜）。已對 authored 驗證：
   週期 31/31 首精確、相位中位 ~10ms
3. **生成譜自宣稱的 grid 永不作對齊參照**——自我一致 ≠ 對齊，生成器
   可以宣稱錯的格線然後自洽（C2 的博弈向量）；只能用於內部一致性指標

層級 1×2 互為 cross-check：兩者嚴重不合＝該歌 metadata 可疑的資料稽核
旗標。

## 4. 劣化探針 harness

| 探針（劑量旋鈕） | 損壞什麼 | 指定目標 |
|---|---|---|
| C1 時值抖動（±10/20/30ms） | 微時值 | rhythm complexity |
| C2 錨定偏移（+15/30/60ms） | 音訊對齊 | 時值家族（§5.3） |
| C3 音色置換（p=.2/.4/.8） | 音符語法 | transition validity |
| C4 複讀化（p=.3/.6/1） | 結構 | repetition adequacy |
| C5 平庸化（p=.3/.6/1） | 趣味性 | （抗博弈展示） |
| C6 密度縮放（×.5/1.5/2） | 難度適配 | density adequacy |
| C7 爆發插入（2/4/8 叢） | 可玩性 | density spike |
| C8 小節洗牌（p=.3/.6/1） | 全局形式 | repetition adequacy |

167 張官方譜 × 8 探針 × 3 劑量（4,175 張含原譜；決定性 seed；算子經
獨立重執行逐位核實）。每個探針保存它不攻擊的東西：C3–C5 精確保留全部
音符時間；C4/C6 保存或控制密度。

**方向保證是群體層級的**：band 分數是雙側的，劣化可能把個別離帶譜推向
帶心（實測每探針最大劑量 4–14% 的譜），所以全部準則在 167 張的均值上
評估。驗收要求：(a) **敏感性**——指定目標分數隨劑量單調下降；
(b) **抗博弈**——任何分數家族都不得獎勵損壞；(c) **公開耦合**——完整的
探針×指標反應矩陣公開，不以對角線摘要。

**公平性規範（事前固定）**：基線與校準分數跑完全相同的探針；跨層同源
指標對（distinct-4 ≡ unique-4-gram；transition validity ↔ 未見轉移率）
永不計為獨立佐證；全部指標檢查譜長混淆；LM 出處經稽核（見 §6），且
平庸化攻擊另跑 **cross-LM 版本**（驅動與測量的 LM 訓練於不相交的語料
半），排除「攻擊自己的模型」的質疑。

## 5. 結果

### 5.1 敏感性：針對的損壞移動對的分數

六個有校準目標的探針全過：Spearman ρ(劑量, 目標分數) 皆為負且顯著——
C1 rhythm complexity（−0.36）、C3 transition validity（−0.33）、C4
repetition adequacy（−0.52）、C6 density/strain（−0.65）、C7 density
spike（−0.78）、C8 repetition（−0.42），均值軌跡單調（p<10⁻¹⁵）。C2 的
目標是時值家族（§5.3）；C5 的指定 band 目標穩住不動正是抗博弈特性，
見下節。

### 5.2 歸謬：借來的指標獎勵損壞

**Perplexity 獎勵平庸。** 官方譜比 greedy 機器輸出更「令人意外」
（perplexity 7.98），而把人類譜壓平向 LM 眾數模式使 perplexity **下降
31%**（7.98→5.48）。按「越低越好」的標準讀法，perplexity 偏好被損壞的
譜——似然/品質背離（Theis et al.）的語料 LM 實例。建立在**同一個** LM
統計量上的 band 分數不被騙（0.892→0.900）：官方譜本就位於帶上緣，壓平
把 NLL 推向帶心而非帶外；同時 repetition adequacy 抓到攻擊真正造成的
損壞（誘發的重複）。此穩健性在 cross-LM 版攻擊下不變（驅動用 held-half
LM、測量用 full LM：7.98→5.55）——perplexity 的失效不是「攻擊自己的
模型」的 artifact。

**結構度獎勵崩潰。** 把譜面換成自己最常見 4-gram 的循環，使自相似結構度
單調**上升 50%**（0.096→0.144）——copy-paste 是最大自相似——集合層級
的 Self-BLEU 同步上升（0.860→0.892）。我們的 repetition/variety adequacy
正確下跌。

**在野外同樣成立。** 對品質範圍已知的八個生成器，perplexity 分離官方譜
與生成譜的 AUC 是 **0.32**（反向）；多個借用統計量是譜長的近似代理
（vs 音符數 Spearman 高至 0.97；校準 rankers ≤0.09）。

### 5.3 不用參照譜關閉錨定盲區

整譜對音訊平移 60ms 在遊玩中是致命的，但**全部 35 個 chart-only 指標
全盲**（最大反應 0.0004 SD）。兩個超越 headline 的發現：

1. **顯然的修法也是盲的**：matching-based signed offset（每音符配對最近
   錨點、平均殘差）本身 C2 全盲——細分格線 ~14ms 的間距讓每個平移音符
   被重新配對到鄰近錨點。偵測整體錯位需要 **fixed-grid support
   maximization**（在八分音符格上最大化支撐的相位搜尋、±75ms 無繞回）
   ——它把注入的 +15/30/60ms 回收為 +15.08/+30.11/+60.01ms（誤差
   <0.2ms，estimated grid、est_ok 129 張），校準分數 0.542→0.011
2. **雙模式閉環**：authored grid（有官方 timing metadata 時）是最鋒利的
   偵測器；estimated grid（純音訊）在 reference-free 模式下仍精確抓到
   ——兩種部署場景各自閉環

誠實註記：單首歌的估計格線可有 ~15ms 相位差，使該歌的 unsupported rate
飽和到 1.0——官方支撐基線要報分佈不報均值，且 grid_phase_offset（而非
unsupported rate）才是 C2 的主偵測器。

### 5.4 耦合結構：我們不宣稱的特異性

我們明確**不**宣稱每個探針只動自己的目標。完整反應矩陣公開：沒有探針
通過嚴格的對角優勢檢定，原因可辨識——共享 tokenization 使時值抖動改變
IOI bin、連動下游所有 n-gram 統計（部分是領域真相：打錯時間的音符就是
打壞的 pattern；部分是套件限制），而 diagnostic 層的 rare-n-gram 對幾乎
任何 token 編輯都反應（分級因此把它排除在 ranker 層外）。使用者應把
套件讀作**帶公開反應圖的耦合 profile**，而非獨立正交軸。反過來，各家族
的盲區互斥（時值家族在 C3/C4/C5 恰為 0.0；語料家族在 C2 恰為 0.0004 SD
以下）正是 profile 不可壓縮為單一分數的證明。

### 5.5 應用：評估既有系統 ⏳

⏳ 進行中——本節數字待生成管線完成後回填。設計：

- **taiko 域（全 profile）**：Mapperatorinator v32（廣用社群系統，MIT、
  原生 taiko、條件式——非同儕評審，如實定位）＋TaikoNation（FDG 2021，
  同儕評審的 taiko 專用先行系統）
- **跨遊戲組（遊戲無關家族）**：DDC/ddc_onset（DDR）、GenéLive!
  （StepMania）、AutoOsu（osu!mania）——時值與音樂耦合家族零修改適用、
  結構家族 raw 層可算、語料校準家族（語法/流形/約束帶）明註不適用
- 產出**指標適用性矩陣**（家族 × 遊戲詞彙）：audio-anchored 與 coupling
  家族跨遊戲零修改、語料家族需目標遊戲語料重校準——scope 陳述而非缺陷
- 跨世代比較（同一把校準尺）：較舊系統世代的 transition validity 生成
  均值 0.37 → 現世代 0.75–0.82——單參照 F-measure 無法做的跨研究陳述

### 5.6 Suite v2：以本 harness 篩選的新指標 ⏳

為示範 harness 的用途，我們以五視角設計了 26 個候選指標、實作 8 個、
全部通過同一套預註冊考核。裁決：ADOPT 4（official_manifold_gap——
反平庸的流形距離軸；call_response_reciprocity——長程對話結構、分離
official 與最佳健康系統 AUC 0.681；density_energy_response——全套件最
獨立 max|r|=0.11；energy_peak_support——免格線的音訊耦合），HOLD 1
（motif development：最強分離器 AUC 0.754 但可被部分複讀操縱——正是
harness 該攔的），DROP 1（自相似形式分數在自己的目標探針上反向——
得了它聲稱要修的病）。⏳ 三件納入修復進行中（checklist A 組）。

## 6. 限制與未來工作

**尚無人類效度。** 探針驗證的是損壞偵測能力，而非玩家體驗的預測力；
對玩家判斷的效度驗證留待未來工作。
**群體層級方向。** 4–14% 個別譜在劣化下向帶心移動；逐譜判定需要
profile 而非單一分數。
**耦合。** 套件是耦合 profile（§5.4）；時值與 pattern 損壞的歸因被共享
tokenization 塗抹。
**範圍。** 單一遊戲域的校準語料（跨遊戲組僅遊戲無關家族）、course 層級
（未到 level）的帶、（clean split 下 LM 校準語料與受評歌曲於建構上不相交）。
**天花板。** 健康系統之間的 profile 差異在評估噪聲內；「好與更好」仍是
人類問題（家族 4/5 的新指標把這條線往上推了一格：reciprocity 分離
official 與最佳健康系統 AUC 0.681）。

## 7. 結論

譜面生成需要的是可信的評估，多於需要另一個指標。劣化探針使指標驗證
系統化且可重現：借來的指標不只是表現不佳——它們獎勵平庸、獎勵崩潰、
並完全錯過致命的整體錯位。同樣的 raw 量經語料校準帶正規化、加上外部
格線錨定的時值家族，以單調劑量反應與次毫秒平移回收通過同一套探針，
其耦合結構公開而非隱藏。harness、校準 artifact、逐譜紀錄與 `chartgeneval`
套件全部釋出——使後續的譜面品質指標能在同一協定下接受驗證。

---

## 附錄 A：與英文版的對應

| 本文 | paper.tex |
|---|---|
| §3 家族表＋§3.1 參照層級 | §3（家族矩陣改寫待辦，checklist B）|
| §5.5 應用 ⏳ | §Application（撤自家模型改寫待辦）|
| §5.6 suite v2 ⏳ | 待 A 組修復後併入 |
| 其餘各節 | 一一對應，數字相同 |

## 附錄 B：數字出處（SoftChart repo）

- 探針：`experiments/corruption_probes_v1/runs/reports/corruption_probes_20260710/`
- meta-eval：`experiments/metric_baselines_v1/runs/reports/metric_baselines_20260709/`
- 時值整合：`experiments/timing_integration_v1/runs/reports/timing_integration_20260710/`
- LM 出處稽核：`experiments/lm_provenance_v1/`
- suite v2 考核：`experiments/metric_candidates_v1/`
- 乾淨切分：`experiments/lm_provenance_v1/clean_split_manifest.json`＋
  HF `JacobLinCool/taiko-1000-parsed-clean`
