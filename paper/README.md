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
abstract; 1,878 characters against the 1,920-character field limit):

> A rhythm-game chart must fit its music and play well, yet the same song
> admits many valid charts. Charting is design, so agreement with the one
> official chart penalizes valid alternatives along with real flaws. Freedom
> of note choice still leaves one requirement that no alternative escapes:
> whatever notes a chart chooses must stand on the song's musical time.
> Timing is the floor of musical fit. We introduce ChartGenEval, a metric
> suite that measures this floor without treating any note sequence as the
> answer: it takes from the official chart only its authored timing map,
> never its note placements. Its whole-chart grid-phase offset recovers
> injected 15-60 ms shifts with zero median error -- shifts that leave every
> primary chart-only output within 0.03 standard deviations of the human
> charts.
>
> Above the floor, quality admits many answers, so ChartGenEval replaces the
> single ranking with a score profile: six dimensions -- timing, note
> sequence, repetition and form, response to music, distance from
> same-difficulty human charts, and playability limits -- are scored
> separately, with no overall total. Controlled corruptions show why:
> rewriting toward common patterns lowers language-model perplexity by 37%,
> and loop collapse raises self-similarity by 62%. Ten frozen
> corruption-measurement tests, selected on a 40-song development panel, all
> survived held-out confirmation on 80 untouched song groups, with all 32
> scoped controls passing; one algebraically coupled pair leaves nine
> nonredundant tests. Other proposed components carry development evidence;
> criterion validity against player judgments remains open. Applied to five
> public generators, the suite reveals timing and difficulty-specific
> departures that a single ranking would hide. We release the implementation,
> corruptions, calibration, and per-chart records.

The public code and records artifact is
<https://github.com/JacobLinCool/ChartGenEval>. Paper licensing is described in
the repository's `ARTIFACT_LICENSE.md`.
