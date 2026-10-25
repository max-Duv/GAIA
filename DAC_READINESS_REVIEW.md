# GAIA review: bibliography, proposal coverage, and IEEE figures

Reviewed 2026-10-24 against checkout 2c0119b and results_20261024_195326.
This is an audit, not an assertion of publication readiness. No experimental or
manuscript sources were changed. Figure regeneration used tmp/dac_audit/figures.

## Bibliography attachment

The attachment has 15 BibTeX entries. The earlier First Look correction is
consistent with the [Caltech record](https://authors.library.caltech.edu/records/1t9q5-vns23).
Plundervolt's corrected venue, authors, and pages 1466--1482 agree with the
[author institutional record](https://research.birmingham.ac.uk/en/publications/plundervolt-software-based-fault-injection-attacks-against-intel-/).
However, the list still contains substantive errors:

| Key / paper | Required correction | Evidence |
| --- | --- | --- |
| yandrofski2022mpe | Pages 383--395, not 210--222; corrected author list matches the authors' paper. DOI 10.1109/RTSS55097.2022.00040. | [Author-group publication citing the paper](https://www.cs.unc.edu/~anderson/papers/ecrts24b.pdf) |
| islam2017thermal | Pages 1079--1094, not 1021--1034. | [Caltech publication record](https://authors.library.caltech.edu/records/207qs-6wk44) |
| wang2022hertzbleed | Authors, in order: Yingchen Wang; Riccardo Paccagnella; Elizabeth Tang He; Hovav Shacham; Christopher W. Fletcher; David Kohlbrenner. Title ends with "on x86". The supplied list contains incorrect additional authors. Pages 679--697 are correct. | [USENIX BibTeX](https://www.usenix.org/conference/usenixsecurity22/presentation/wang-yingchen) |
| kogler2023collide | Pages 7285--7302. Authors: Andreas Kogler; Jonas Juffinger; Lukas Giner; Lukas Gerlach; Martin Schwarzl; Michael Schwarz; Daniel Gruss; Stefan Mangard. The supplied Moritz Lipp attribution is incorrect. | [USENIX BibTeX](https://www.usenix.org/conference/usenixsecurity23/presentation/kogler) |
| lipp2021platypus | Pages 355--371, not 3551--3568. IEEE S&P is the correct venue. | [University of Birmingham](https://research.birmingham.ac.uk/en/publications/platypus-software-based-power-side-channel-attacks-on-x86/) |
| chen2021voltpillager | Pages 699--716, not 2139--2156. This is a hardware-access attack, not the same paper or threat model as VoltJockey. | [USENIX BibTeX](https://www.usenix.org/conference/usenixsecurity21/presentation/chen-zitai) |
| boutros2022neighbors | The identified paper is ICFPT 2020, pp. 103--111, not the supplied TDSC 2022 article. Reconcile the final published author list with the publisher: the author-uploaded preprint includes Mathew Hall, whereas the author's CV lists three authors. Do not silently mix versions. | [Author CV](https://www.eecg.toronto.edu/~vaughn/vaughn_betz_cv.pdf), [author preprint](https://arxiv.org/abs/2012.07242) |
| yarom2014flush | Restore the full title: "FLUSH+RELOAD: A High Resolution, Low Noise, L3 Cache Side-Channel Attack". | [USENIX](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/yarom) |
| shusterman2021prime | Pages 2863--2880, not 2683--2700; corrected title/authors match the publisher. | [USENIX BibTeX](https://www.usenix.org/conference/usenixsecurity21/presentation/shusterman) |
| kogler2022minefield | Pages 4147--4164, not 2853--2870. | [USENIX BibTeX](https://www.usenix.org/conference/usenixsecurity22/presentation/kogler-minefield) |

Replace every `S\&amp;P` with `S\&P`. Remove Markdown fences before saving as .bib.
Protect acronyms and project names with braces. Prefer publisher-exported entries
over manually reconstructing author lists. Not every field of the remaining
entries has been independently certified in this review.

The manuscript currently uses paper/references.bib, a different bibliography
with different keys; the pasted corrections are not automatically applied to it.

## Proposal progress and remaining evidence

The proposal's primary tasks remain GPU workload characterization, defensive
monitoring, and an end-to-end monitoring/response prototype. Implementation
coverage has improved, but implementation and experimental validation are separate.

### Temperature-triggered stress

T1/controlled_3phase_runner_v2.py now waits for a temperature increase greater
than 2 C, with a five-second timeout before launching anyway. This is an initial
temperature-triggered experiment, not yet validated synchronization to victim
training/inference phases. The CSV does not record trigger-versus-timeout outcome,
threshold, or wait duration. Record those fields and compare against unsynchronized
launches before claiming benefit. A timed-out launch must not be counted as a
successful temperature trigger.

### Utilization and performance counters

T1/gpu_fault_assessor_v2.py now queries NVML GPU and memory utilization. This is
useful utilization telemetry, but it does not implement cache-line profiling,
L2 hit/miss measurements, occupancy counters, or a DCGM-based detector. Failure
still becomes zero, conflating unavailable measurements with genuine inactivity.

### Termination response

T3/closed_loop_mitigator_v2.py now sends terminate(), waits, and escalates to
kill(). This is process termination, not demonstrated cooperative CUDA cleanup.
No CUDA signal handler was found. The activation flag can become true when the
process has already exited; after kill(), the code does not wait again before
logging success. Record PID, action, exit status, and completion time separately.
Evaluate collateral victim effects and power recovery.

### Monitoring exporter

T3/prometheus_exporter.py exposes metrics and JSON. The shell runner launches it
after experiments and captures endpoint snapshots after three seconds. This is
an exporter smoke test, not a demonstrated Prometheus scrape, Grafana dashboard,
or integrated response pipeline. At a nominal 0.5-second interval, the detector
requires at least 16 samples; a three-second snapshot cannot establish its
post-warm-up detection behavior. Add a real scrape configuration, dashboard and
end-to-end alert/action trace. DCGM integration remains absent.

### Spectral reporting regression

analyze_results.py's new Welch section filters by phase but concatenates all
trials. Timestamps reset per trial, so mean(diff(timestamp)) includes resets and
phase gaps. The fallback sampling rate does not fix that discontinuity. Compute
spectra per trial/phase with an explicit sampling/resampling policy, then summarize
across trials. Integrating a PSD also requires frequency-bin widths; sum(psd)
alone is not integrated signal variance.

### Data and manuscript scope

The latest supplied main CSV still has five conditions; the T3 CSV has four.
Neither contains temperature-sync or termination trials, and utilization columns
are absent. Those additions must be described as implemented/pending evaluation,
not measured results from this dataset. The abstract currently merges these
statuses, and states clock boosting as a proven cause rather than a supported
hypothesis. It also labels a limited baseline result as zero false positives
without clearly limiting the benign workload population.

README.md's coverage table still says the new items are not implemented. Update
it with separate implementation and validation status rather than marking all
proposal items complete.

## IEEE / DAC figure audit

The official [DAC 2027 research rules](https://dac.com/2027/program/research-manuscript-submissions)
specify the IEEE template, double columns, 9- or 10-point manuscript type, six
content pages and one references-only page. The final submission is anonymous.
The rules also include disclosure requirements for AI-generated content.

[IEEE conference graphics guidance](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/)
recommends approximately 3.5-inch one-column or 7.16-inch two-column graphics,
embedded fonts, and approximately 9--10-point type at final size. Recommended
fonts include Times New Roman, Helvetica, Arial, Cambria, and Symbol.

### Checks performed

- Regenerated all six figure pairs with publication_figures.py through the CLI.
- Parsed each PDF: vector graphics (no embedded raster images); embedded subset
  DejaVu Sans TrueType/CIDFontType2 fonts; no Type 3 fonts found.
- All PDFs are approximately 7.277 inches wide. `bbox_inches='tight'` plus padding
  changes the nominal 7.16-inch canvas dimensions.
- Font sizes: Fig. 1 9/12 pt; Fig. 2 9/10/12; Fig. 3 7/9/10/12;
  Fig. 4 7/9/10/12; Fig. 5 9/10/12; Fig. 6 8/9/10/12.
- Rendered representative PDFs (Figs. 1 and 4); reviewed Fig. 1 in grayscale.
  No clipping was apparent. Thin translucent traces lose contrast in grayscale.
- Python syntax checks passed for seven relevant source files.

### Assessment

The PDFs are technically usable in LaTeX, but need a final IEEE paper layout pass.
At 7.16-inch placement, 9 pt becomes about 8.86 pt and 7 pt about 6.89 pt.
At 3.5-inch placement, those sizes become about 4.33 pt and 3.37 pt. Thus the
current composite figures must not simply be shrunk into a single column.

Recommended renderer changes:

1. Explicit one-column/two-column export profiles with fixed physical dimensions.
2. Use an IEEE-recommended embedded font; DejaVu Sans embedding is technically
   sound but does not match the recommended list.
3. Raise seven-point legends/annotations and remove oversized overall titles;
   let the manuscript caption carry the main explanation.
4. Strengthen contrast and use marker shapes where categories share axes.
5. Add sample-size/interval definitions to the final captions. A degenerate
   empirical CI from five zero-alarm trials does not prove population FPR is zero.
6. For new termination runs, add termination to Fig. 6 event-delay comparisons
   and show its response timeline. Currently only no-cap/capping phases appear
   in the delay panel, even though termination is in the summary mapping.
7. Keep the current measured-kernel column priority; the fallback total-latency
   column cannot universally be called kernel time for arbitrary future datasets.

The separation of trials, equal weighting of trial means, honest voltage labels,
and deterministic illustrative trial selection are improvements worth retaining.

### Manuscript integration blocker

The built-in LaTeX compilation attempt failed at paper/gaia_paper.tex:294 because
`table_t1_results.tex` cannot be found. The second table uses the same bare-input
pattern. `\graphicspath` does not resolve `\input` paths. Reference the actual
table paths (or copy the tables into the paper project).

The manuscript declares IEEEtran[conference], but currently has no
`\includegraphics` commands, so the figures are not actually incorporated.
`subcaption` also emitted a caption-class warning. Full-page figure placement,
caption compatibility, margins, and page count remain unverified. The author
block must be anonymized for DAC submission, and the draft needs the required
AI-content disclosure appropriate to actual use.

For these wide composites, use an IEEE two-column float, for example:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{fig4_t3_mitigation.pdf}
  \caption{Power/performance tradeoff. Dots show trial means; diamonds and
  error bars show the mean and 95\% Student-t confidence interval across
  trials. Whole-phase averages include pre-activation samples. The lower
  panels illustrate trial 1; the dashed line is the logged activation flag.}
  \label{fig:mitigation}
\end{figure*}
```

This is a placement example, not evidence that the current draft compiles or
fits the page limit. Complete the final check on the compiled manuscript PDF.
