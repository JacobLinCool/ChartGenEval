# Claim–evidence map

- Experiment: `experiments/confirmatory_holdout_v1/SPEC.md`
- Phase: `confirmatory`
- Spec snapshot hash: `db06188e0308215902cbb96724de6f0160c5ade89de1ef510c43f3cec22a920a`
- Config snapshot hash: `66dab335bbd6bfd48374b9e17a8590f7ae6d7da7aca98c2aa0bd139e41927653`
- Contract snapshot hash: `1e40735d93d88444730523de7e77d101cbf59a6a033cf9c533e241b5e1286390`
- Freeze hash: `20b319cbfdfaae24c2a13babe2e58243159227056013655ece318cbb85362c38`
- Confirmatory anchor hash: `eaa55490e9ca377e1238df25e4e420c87d670c184a3900ac26347c0a08ae0d0f`
- Code-bundle hash: `d31f0811650a0c48aadd8299f292cff6f5507bb7db4ee13bf5b5737a50c0f760`
- Runtime: `{"packages":{"chartgeneval":"0.1.0","datasets":"5.0.0","numpy":"2.2.6","scipy":"1.17.1"},"platform":"macOS-26.4-arm64-arm-64bit","python":"3.11.14","python_executable":"/Users/jacoblincool/Documents/GitHub/ChartGenEval/.venv/bin/python","score_version":"calibrated_diagnostics_v1_symmetric_halfwidth"}`
- Language-model contract: `{"alpha":0.05,"dataset_id":"JacobLinCool/taiko-1000-parsed-clean","dataset_revision":"b72da4616d643018e81f372cea06ce51349285e0","expected_canonical_groups":813,"expected_charts":3880,"expected_source_rows":924,"order":3,"range":"full_split","row_policy":"all_source_rows_including_aliases","split":"train"}`
- Language-model fingerprints: `{"model_state_sha256":"07b82322f447a4a8442c3d19af216bf9d822d08b197c3b442273082fe90f3f6a","n_train_charts":3880,"n_train_rows":924,"n_train_unique_group_ids":813,"ordered_training_input_sha256":"fcc45c12bba23e4fcecdf6da6a126685b08d69690e89e0ff7351cf2320e3a51e"}`
- Raw run manifest: `experiments/confirmatory_holdout_v1/runs/raw/confirmatory-primary-20260712/MANIFEST.json`
- Raw execution command: `["/Users/jacoblincool/.local/share/uv/python/cpython-3.11.14-macos-aarch64-none/bin/python3.11","experiments/confirmatory_holdout_v1/run.py","--config","experiments/confirmatory_holdout_v1/configs/confirmatory.json","--freeze","experiments/confirmatory_holdout_v1/frozen/confirmatory_freeze.json","--confirm-untouched-panel","--run-id","confirmatory-primary-20260712"]`
- Raw evidence hashes: `{"manifest_events.jsonl":"98b5d956ff09e9a53ae8973b8501c371a9c69e07f898473f5f707048efc83b6f","records.jsonl":"dc2f67931f10544bf0840c05ce1d11d8b2596302d1845f6d33751d5280ed6c8e","sampling_context.jsonl":"1e1d8ab69833e9844986665d5bf1a44a91c933787d53b3e742f3485b8833f640","source_scan.jsonl":"2f45ca7ba058775226668684689da9d520121a351aea6adc7cb33658a6c48cce"}`
- Canonical manifest: `experiments/confirmatory_holdout_v1/runs/raw/confirmatory-primary-20260712/tables/MANIFEST.json`
- Transformation command: `["/Users/jacoblincool/.local/share/uv/python/cpython-3.11.14-macos-aarch64-none/bin/python3.11","experiments/confirmatory_holdout_v1/transform.py","--run-dir","experiments/confirmatory_holdout_v1/runs/raw/confirmatory-primary-20260712"]`
- Canonical table hashes: `{"chart_dose_means.jsonl":"7338f2cd8104c716c9408767857830e5e4ff4cd2d8ac174e476518f6190ad828","exclusions.jsonl":"e72b21fa1ea8a59bed81ee1c92d59295ec41729370fa1faf779e2478349e2e58","observations.jsonl":"98cde119b4e0273ee7d9e030802fadb6cae54b93d6a39349a34cb0de3c26fe99","panels.json":"d5547df2401612e81d1387b8c6ac936405d4d641545eefd9a6741610f0b073ba","sampling_context_songs.jsonl":"1e1d8ab69833e9844986665d5bf1a44a91c933787d53b3e742f3485b8833f640","source_scan.jsonl":"2f45ca7ba058775226668684689da9d520121a351aea6adc7cb33658a6c48cce"}`
- Analysis command: `["/Users/jacoblincool/.local/share/uv/python/cpython-3.11.14-macos-aarch64-none/bin/python3.11","experiments/confirmatory_holdout_v1/analyze.py","--table-dir","experiments/confirmatory_holdout_v1/runs/raw/confirmatory-primary-20260712/tables"]`
- Dataset revision: `b72da4616d643018e81f372cea06ce51349285e0`
- Clean split manifest hash: `ba55473803771eea6670529e329a7c09e9e23ecd41da4f865bc2ae309128b7e5`
- Datasets fingerprints: `{"test":{"dataset_revision":"b72da4616d643018e81f372cea06ce51349285e0","filtered_alias_rows":20,"kind":"ordered_canonical_chart_content_v1","n_canonical_songs":120,"range":[0,120],"scanned_source_rows":140,"sha256":"65dbeb371c2c24f9d9adaec58959a5800421276474e4d9cad59132c84dcaa75b","source_scan_kind":"ordered_raw_source_identity_v1","source_scan_sha256":"7f2225e19a69137af566dde1bb969d2a7f5eaeb5512504b8d30a6ccac821bb31","split":"test"},"train":{"dataset_revision":"b72da4616d643018e81f372cea06ce51349285e0","kind":"ordered_parsed_chart_content_v1","n_charts":3880,"n_source_rows":924,"n_unique_group_ids":813,"range":"full_split","sha256":"02faa003514b7f9de952d4cc6bf4e5ea7f709e6159e459e233238414e99fb4b0","split":"train"}}`
- Raw/canonical/alias scan counts: `140` / `120` / `20`
- Selected group_id hash: `7aee2fa11d12ab8a03f53bfc1c416aa3ba8f80c6a2001f5d7cff6e265eb7570e`
- Selected audio_sha256 hash: `a6dbb219dafa57268a54b2ee04b8380080a9c786ebc7c4b4d50192d78336a671`
- Selected panel fingerprint: `1dcbd4315f1269c585a69d120daee642d9e899856a5b89aa0748ea346dbda848`

Sampling-context comparisons are descriptive only and cannot revise hypotheses.

## C1__timing_clean_rate

- Probe / metric: `C1_timing_jitter` / `timing.clean_rate`
- Scoped sham: `CTRL_color_bijection` dose `0`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.7441132117688026`
- Mean target-minus-scoped-sham contrast: `-0.5016433031483702`
- Nominal 95% intervals: `{"dose_statistic":[-0.7938632117688018,-0.6932379170630224],"target_minus_sham":[-0.5407179741221122,-0.4628631503540896]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.8151255955294215,-0.6669993160588435],"target_minus_sham":[-0.557667712201931,-0.44600393967586843]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"timing.clean_rate","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"timing.clean_rate","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_color_bijection","max_absolute_delta":0.0,"metric":"timing.clean_rate","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C1s__timing_p99

- Probe / metric: `C1s_sparse_jitter` / `timing.absolute_error_p99_ms`
- Scoped sham: `CTRL_color_bijection` dose `0`
- group_id clusters: 80
- Complete courses: 332
- Mean within-chart dose statistic: `-0.23409768705365647`
- Mean target-minus-scoped-sham contrast: `-0.3060017175597708`
- Nominal 95% intervals: `{"dose_statistic":[-0.30987492396303445,-0.1625491402234616],"target_minus_sham":[-0.4409903198859265,-0.1862299270975617]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.34350979893178796,-0.13367948855510595],"target_minus_sham":[-0.5083130236821816,-0.14415103892388312]}`
- Required controls / pass: `[{"absolute_tolerance":1e-09,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"timing.absolute_error_p99_ms","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-09,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"timing.absolute_error_p99_ms","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-09,"control":"CTRL_color_bijection","max_absolute_delta":0.0,"metric":"timing.absolute_error_p99_ms","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density"]}]` / `True`
- Pair exclusions: 1; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C2__timing_phase_abs

- Probe / metric: `C2_anchor_shift` / `timing.grid_phase_offset_abs_ms`
- Scoped sham: `CTRL_joint_time_origin_shift` dose `0`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.9157499999999998`
- Mean target-minus-scoped-sham contrast: `-55.5765625`
- Nominal 95% intervals: `{"dose_statistic":[-0.9650000000000001,-0.85549375],"target_minus_sham":[-59.696953125,-51.5374609375]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.98125,-0.8257487499999997],"target_minus_sham":[-61.42675390625,-49.831187499999984]}`
- Required controls / pass: `[{"absolute_tolerance":1e-06,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"timing.grid_phase_offset_abs_ms","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-06,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"timing.grid_phase_offset_abs_ms","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-06,"control":"CTRL_color_bijection","max_absolute_delta":0.0,"metric":"timing.grid_phase_offset_abs_ms","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C3__transition

- Probe / metric: `C3_type_shuffle` / `transition_validity_score`
- Scoped sham: `C2_anchor_shift` dose `3`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.8001085412256312`
- Mean target-minus-scoped-sham contrast: `-0.23063982786336065`
- Nominal 95% intervals: `{"dose_statistic":[-0.8403240606493791,-0.7569628670717283],"target_minus_sham":[-0.25304096865472114,-0.20844485818184164]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.8558586769021637,-0.7371079573725673],"target_minus_sham":[-0.26223657334175937,-0.19853695189110443]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"transition_validity_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"transition_validity_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"C2_anchor_shift","max_absolute_delta":0.0,"metric":"transition_validity_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":4995,"n_expected_control_replicates":4995,"n_missing_control_replicates":0,"passes":true,"scoped_families":["density","grammar","structure"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C4__repetition

- Probe / metric: `C4_loop_collapse` / `repetition_adequacy_score`
- Scoped sham: `C2_anchor_shift` dose `3`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.4446630320197601`
- Mean target-minus-scoped-sham contrast: `-0.1654512415443341`
- Nominal 95% intervals: `{"dose_statistic":[-0.5359186256660557,-0.3467810699911918],"target_minus_sham":[-0.20761703700957349,-0.12440007128280017]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.573406539674599,-0.31188201488547185],"target_minus_sham":[-0.22564482104497666,-0.10766323668084578]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"C2_anchor_shift","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":4995,"n_expected_control_replicates":4995,"n_missing_control_replicates":0,"passes":true,"scoped_families":["density","grammar","structure"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C5__repetition

- Probe / metric: `C5_blandification` / `repetition_adequacy_score`
- Scoped sham: `C2_anchor_shift` dose `3`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.35828658870992514`
- Mean target-minus-scoped-sham contrast: `-0.1309997576445224`
- Nominal 95% intervals: `{"dose_statistic":[-0.4288667319289085,-0.2846771553859691],"target_minus_sham":[-0.17147261547126558,-0.09127425388250933]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.4589517664030271,-0.2584138708188906],"target_minus_sham":[-0.18854931204998251,-0.07759928844076433]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"C2_anchor_shift","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":4995,"n_expected_control_replicates":4995,"n_missing_control_replicates":0,"passes":true,"scoped_families":["density","grammar","structure"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C5__surface_variety

- Probe / metric: `C5_blandification` / `surface_variety_adequacy_score`
- Scoped sham: `C2_anchor_shift` dose `3`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.35828658870992514`
- Mean target-minus-scoped-sham contrast: `-0.13099975764452249`
- Nominal 95% intervals: `{"dose_statistic":[-0.42906886234642483,-0.2864302944297231],"target_minus_sham":[-0.17102351305242405,-0.0915696827654272]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.4558409955389629,-0.2546490062961315],"target_minus_sham":[-0.18648616394824086,-0.07385252360647898]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"surface_variety_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"surface_variety_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"C2_anchor_shift","max_absolute_delta":0.0,"metric":"surface_variety_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":4995,"n_expected_control_replicates":4995,"n_missing_control_replicates":0,"passes":true,"scoped_families":["density","grammar","structure"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C6__density

- Probe / metric: `C6_density_scale` / `density_adequacy_score`
- Scoped sham: `CTRL_color_bijection` dose `0`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.6930019703762486`
- Mean target-minus-scoped-sham contrast: `-0.581617642873763`
- Nominal 95% intervals: `{"dose_statistic":[-0.7601519901750258,-0.6196335705101712],"target_minus_sham":[-0.6454660684495344,-0.5155329873564477]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.7843720629843747,-0.5793850835055417],"target_minus_sham":[-0.671926612077162,-0.48258032134837703]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"density_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"density_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_color_bijection","max_absolute_delta":0.0,"metric":"density_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density"]},{"absolute_tolerance":1e-12,"control":"C2_anchor_shift","max_absolute_delta":0.0,"metric":"density_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":4995,"n_expected_control_replicates":4995,"n_missing_control_replicates":0,"passes":true,"scoped_families":["density","grammar","structure"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C7__density_spike

- Probe / metric: `C7_burst_insert` / `density_spike_score`
- Scoped sham: `CTRL_color_bijection` dose `0`
- group_id clusters: 80
- Complete courses: 333
- Mean within-chart dose statistic: `-0.940032218397946`
- Mean target-minus-scoped-sham contrast: `-0.6191632365398558`
- Nominal 95% intervals: `{"dose_statistic":[-0.9472105649522324,-0.9327763899516123],"target_minus_sham":[-0.6369252036441853,-0.6011538300206211]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.9500500553130761,-0.9296850729698268],"target_minus_sham":[-0.6436178260189218,-0.5932441716725135]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"density_spike_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"density_spike_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_color_bijection","max_absolute_delta":0.0,"metric":"density_spike_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density"]},{"absolute_tolerance":1e-12,"control":"C2_anchor_shift","max_absolute_delta":0.0,"metric":"density_spike_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":4995,"n_expected_control_replicates":4995,"n_missing_control_replicates":0,"passes":true,"scoped_families":["density","grammar","structure"]}]` / `True`
- Pair exclusions: 0; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C8__repetition

- Probe / metric: `C8_bar_shuffle` / `repetition_adequacy_score`
- Scoped sham: `C2_anchor_shift` dose `3`
- group_id clusters: 80
- Complete courses: 330
- Mean within-chart dose statistic: `-0.5890921661118667`
- Mean target-minus-scoped-sham contrast: `-0.22439737088329573`
- Nominal 95% intervals: `{"dose_statistic":[-0.6898261482400336,-0.4845507921430419],"target_minus_sham":[-0.2730490745425668,-0.17870924013623407]}`
- Family-wise 99.5% intervals: `{"dose_statistic":[-0.7283928387350407,-0.4354523529602096],"target_minus_sham":[-0.2926127443315933,-0.15586982275199532]}`
- Required controls / pass: `[{"absolute_tolerance":1e-12,"control":"CTRL_identity","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"CTRL_joint_time_origin_shift","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":1665,"n_expected_control_replicates":1665,"n_missing_control_replicates":0,"passes":true,"scoped_families":["timing","density","grammar","structure"]},{"absolute_tolerance":1e-12,"control":"C2_anchor_shift","max_absolute_delta":0.0,"metric":"repetition_adequacy_score","n_baseline_evaluable_charts":333,"n_evaluable_control_replicates":4995,"n_expected_control_replicates":4995,"n_missing_control_replicates":0,"passes":true,"scoped_families":["density","grammar","structure"]}]` / `True`
- Pair exclusions: 3; rule: official + all target doses/replicates + fixed sham + scoped controls must be complete
- Verdict: **SURVIVES**
- Evidence: `primary_results.json`, `control_diagnostics.json`, `../chart_dose_means.jsonl`, `../source_scan.jsonl`, `../../records.jsonl`
- Exclusions: `pair_exclusions.json`

## C5__repetition_and_variety

- Combination rule: `all_pairs_must_survive`
- Pair verdicts: `{"C5__repetition":"SURVIVES","C5__surface_variety":"SURVIVES"}`
- Co-primary verdict: **SURVIVES**

## C5 interpretation boundary

C5 is co-primary across repetition and surface variety using an all-pairs-survive rule. Pattern-IC and raw pattern NLL are secondary ambiguity/diagnostic outputs with no directional gate.

## Limitations

- Controlled corruption establishes sensitivity to the frozen edits, not player or expert criterion validity.
- Authored timing metadata is an external timing map, not direct audio-onset annotation.
- Courses are collapsed within song; remaining author-level dependence is not modeled.
- Deterministic probes can produce identical five-seed replicates and never gain artificial sample size.
- Sampling-context comparisons are descriptive and cannot modify the frozen hypotheses or panel.
