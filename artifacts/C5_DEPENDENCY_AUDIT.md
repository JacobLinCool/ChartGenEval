# C5 dependency audit

Date: 2026-07-12  
Status: post-freeze interpretive addendum; the frozen run, formal verdicts, and
sealed hash chain are unchanged.

## Algebraic result

Let a chart contain $M$ contiguous 4-grams and $U$ distinct 4-grams. The
released features are

$$
\text{unique\_4gram\_rate}=U/M
$$

and

$$
\text{repeat\_4gram\_rate}
=\sum_g (c_g-1)/M
=(M-U)/M
=1-U/M.
$$

The repeated-rate course band is the mirror of the unique-rate band. For a
unique-rate band $[l,u]$, the corresponding repeated-rate band is
$[1-u,1-l]$. Because the released score uses the same half-width on both
sides of a band, $x$ and $1-x$ have exactly the same distance from these
mirrored bands. Consequently, `repetition_adequacy_score` and
`surface_variety_adequacy_score` are the same measurement up to floating-point
roundoff.

## Numerical audit

- Across all 4,760 development records, the maximum raw complement residual
  `abs(repeat_4gram_rate + unique_4gram_rate - 1)` is 0, and the maximum score
  difference is $8.88\times10^{-16}$. Within the 510 C5 development rows,
  the maximum score difference is $6.66\times10^{-16}$.
- The two released C5 primary point estimates are identical within
  $10^{-12}$. Their slightly different
  bootstrap interval endpoints arise from separate bootstrap draw streams,
  not independent observations or different metric information.

## Interpretation boundary

The preregistered reducer mechanically reports `SURVIVES` for both named C5
rows, so the frozen all-pairs rule formally passes. Those ten frozen rows
represent only nine nonredundant probe--measurement tests. C5 supports the
sensitivity of one effective 4-gram repetition/uniqueness test. It also shows
cross-axis coverage because that test detects a rewrite that pattern-IC does
not reliably label as damage and that the LM assigns greater likelihood. It
does not support independent repetition-versus-variety corroboration.

The frozen `surface_structure_proxy_score` is the geometric mean of surface
variety, repetition, and density variation. Because its first two inputs are
the same effective 4-gram test, it double-weights that axis. The field is
retained to preserve the frozen record but is excluded from inferential claims
and the revised paper figures.

The sealed `confirmatory_holdout_v1/co_primary_results.json` and
`confirmatory_holdout_v1/claim_evidence.md` record the preregistered contract
and deterministic reducer output. This addendum supplies the dependency audit
needed to interpret those files correctly. A future confirmation of
structure-metric complementarity must preregister a genuinely non-equivalent
second measurement.
