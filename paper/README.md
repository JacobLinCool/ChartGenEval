# Paper

## ICASSP 2027 submission

The current five-page conference version is [icassp2027/paper.pdf](icassp2027/paper.pdf).
Its source, official Paper Kit, build commands, and submission metadata are in
[icassp2027/README.md](icassp2027/README.md). The files below remain the long-form version.

`paper.tex` is the source for *ChartGenEval: Corruption-Tested
Multi-Dimensional Feedback for Rhythm-Game Chart Generation*. `paper.pdf` is
the release build.
`paper.zh-TW.tex` is the Traditional Chinese companion version.

Build from this directory with:

```bash
latexmk -pdf paper.tex
latexmk -xelatex paper.zh-TW.tex
```

arXiv submission metadata:

- Author: Jhen-Ke Lin
- ORCID: <https://orcid.org/0009-0004-0311-5409>
- Primary category: `cs.SD` (Sound)
- Cross-list: `eess.AS` (Audio and Speech Processing)
- Submission license: arXiv.org perpetual, non-exclusive license 1.0

Plain-text abstract for the arXiv metadata field, identical to the PDF
abstract:

> A generated rhythm-game chart need not reproduce one official note
> sequence: many note choices can fit the same song and difficulty.
> Reference-note agreement therefore measures reconstruction, not the full
> design problem. We introduce ChartGenEval, a six-question evaluation
> framework with an automatic, corruption-tested core. It leaves note choice
> open while anchoring timing to the song: the matched official chart supplies
> only its authored timing map, never target notes.
>
> We test each core output with dose-controlled failures rather than assume
> that a familiar statistic measures chart quality. Across 80 held-out song
> groups, seven output axes satisfy prespecified sensitivity and invariance
> criteria in nine nonredundant tests. Complementary stress tests on the
> 40-song development panel expose two broader lessons. A chart-wide phase
> estimate recovers injected shifts of 15, 30, and 60 ms while chart-only
> outputs remain essentially unchanged. Common-pattern rewriting lowers
> mean language-model perplexity by 37%, and loop collapse raises mean
> self-similarity by 62%.
> ChartGenEval therefore reports separate, role-specific signals instead of one
> proxy or total score. The profile compares measured properties of generator
> outputs; using these readings as optimization targets or constraints requires
> additional task-specific validation.

The public code and records artifact is
<https://github.com/JacobLinCool/ChartGenEval>. Paper licensing is described in
the repository's `ARTIFACT_LICENSE.md`.
