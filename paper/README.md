# Paper

`paper.tex` is the source for *ChartGenEval: A Multi-Dimensional Metric Suite
for Rhythm-Game Chart Generation*. `paper.pdf` is the release build.
`paper.zh-TW.tex` is the Traditional Chinese companion version.

Build from this directory with:

```bash
latexmk -pdf paper.tex
latexmk -xelatex paper.zh-TW.tex
```

arXiv submission metadata:

- Primary category: `cs.SD` (Sound)
- Cross-list: `eess.AS` (Audio and Speech Processing)
- Submission license: arXiv.org perpetual, non-exclusive license 1.0

Plain-text abstract for the arXiv metadata field (identical to the PDF
abstract; 1,438 characters against the 1,920-character field limit):

> A rhythm-game chart must fit its music and play well, yet one song admits
> many valid note sequences. Agreement with one official chart therefore
> conflates valid alternatives with errors. We introduce ChartGenEval, a
> six-dimensional metric suite that uses the matched official chart only for
> its authored timing map, never its note placements. For injected whole-chart
> shifts of 15, 30, and 60 ms, its median grid-phase estimates match the true
> shifts. By contrast, every primary chart-only output changes by less than
> 0.03 human standard deviations.
>
> ChartGenEval reports this timing profile alongside five higher-level
> dimensions: note sequence, repetition and form, response to music,
> same-difficulty human-chart distance, and difficulty and playability limits.
> The dimensions remain separate, with no overall total. Nine nonredundant
> corruption--measurement pairs met prespecified sensitivity and invariance
> criteria on 80 held-out song groups; all 32 applicable control checks
> passed. The
> stress tests also show why isolated proxies can be misleading:
> common-pattern rewriting lowers language-model perplexity by 37%, while loop
> collapse raises self-similarity by 62%. Together, this automated
> six-dimensional profile can support fast comparison and iteration of
> chart-generation models. Depending on their roles, its outputs can serve as
> rewards, constraints, or monitoring signals in training methods such as
> reinforcement learning.

The public code and records artifact is
<https://github.com/JacobLinCool/ChartGenEval>. Paper licensing is described in
the repository's `ARTIFACT_LICENSE.md`.
