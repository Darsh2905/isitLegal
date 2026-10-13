# Is It Legal? — RAG with deterministic, fail-closed citation verification for Indian law

The paper is `is_it_legal_final.tex` (build: `tectonic is_it_legal_final.tex`;
its round-2/support/latency figures are generated into `round2_figures.tex` by
`python3 eval/make_paper_figures.py`). `paper.tex` is the superseded
pre-revision draft and is no longer checked by the test suite.

## Status

| Component | State |
|---|---|
| Citation normaliser | done — 48/48 surface forms, 0 false positives; now also detects sections cited to acts **outside the corpus** (real or invented) and no longer reads an act's own year ("IPC 1860") as a section |
| Metadata DB + IPC↔BNS mapping | done — real corpus: the full UP Police correspondence tables (1,086 verified section-level pairs); 1,084 of 1,264 repealed provisions now have a recorded successor (was 11). Sub-section pinpoints ("Art. 19(1)(a)") resolve to their section only if the labels occur, as labels, in its text |
| Fail-closed revision (peer review, 2026-10) | done — see "Peer-review revision" below: exact quotations, bare references, case citations, a residual scan that strips any reference the parser cannot read; 22/22 new faults stripped, 24/24 correct forms kept (pre-revision code: 1/22, 23/24) |
| Second live round (3 models, AILA queries) | done — 181 answers; resolution 0.984→1.000 grounded, 0.705→1.000 ungrounded (independent audit); coverage 0.922 / 0.738 |
| Support vs resolution (labelled sample) | done — 90 sentences: 0.889 of grounded sentences fully supported, 0.356 of ungrounded; V5 flag precision 0.500 on real output |
| AILA 2019 statute retrieval (official qrels) | done — best arm MAP 0.0782 vs track best 0.1566: mid-table, reported as a negative result |
| Provenance (Gazette) + mapping cross-check | done — 1,040/1,054 BNS/BNSS/BSA sections match the Gazette text; 399/402 IPC→BNS pairs agree with an independent CAPT Bhopal table |
| Deterministic verifier (V1–V4) | done — toy corpus: 13/13 planted faults, 0 false alarms. **Real corpus (new adversarial probe set): 17/17 faults stripped, 14/14 correct citations kept, independent audit 31/31** — the original code scored 12/17 and 8/14 on the same probes. See "Verifier on the real corpus" below |
| Parser evasion resistance | done — 20/20 concealed faults caught (11 defects, 3 audit rounds — the 3rd found live via the demo, not adversarial construction), 0 false positives |
| Support screen V5 (advisory) | done — flags 9/11 wrong attributions, 0 false positives |
| V5 × citation graph (support-via-reference) | done — removes 3 of 5 false alarms on non-self-contained provisions |
| Evidence-bounded repair (R1, R2, R3) | done — +16.7pp coverage on faulty answers, 0 invariant violations (toy probes). New R3 repoints a marker that contradicts the sentence's own citation; R2 now rewrites only to a *unique* successor |
| Cross-reference resolver + citation graph | done — 7 edges, +0.200 Recall@1 on non-self-contained provisions |
| BM25 baseline | done — toy corpus saturated; real corpus Recall@1 0.375 unstemmed, **0.500 with Snowball stemming** (deployed for retrieval; the confidence gate stays unstemmed because stemming weakened abstention) |
| In-force preference (recodification at retrieval time) | done — real corpus: current law ranked above its repealed predecessor in **8/9** gold pairs (BM25 alone: 4/9) |
| Confidence gate / calibrated abstention (C3) | done — toy corpus: 8/8 held-out abstentions, 8/8 held-out answers; recalibrated on the real corpus (τ=1.077, balanced accuracy 0.938); **out-of-sample check on 8 real, externally-sourced queries: 6/8 correct (0.750)** — see below |
| Real statutory corpus (3,275 verbatim provisions, all 10 Acts) | done — replaces saturation (BM25 Recall@1 1.000→0.375 unstemmed). State/UT amendment tables no longer pollute central-law text; 12 inserted Evidence Act sections recovered (incl. §113-B, §65B's siblings, the long-standing §67A gap); IEA §90A corrected (was Uttar Pradesh's §90-A) |
| Dense retrieval (InLegalBERT vs general embedding) | done — general model wins; passage windows help both (general Recall@1 0.438→0.562, InLegalBERT 0.250→0.312); see caveat below |
| RRF fusion (BM25 + dense) | done — at the deployed depth (20 per arm): MRR@10 0.676, Recall@10 0.938, beats both single arms. (Earlier 0.690/1.000 used depth 50 and disagreed with the rerank table; both now use the deployed depth) |
| Cross-encoder reranking | done — scored over passage windows: MRR@10 0.676→0.821, nDCG@10 0.684→0.786, Recall@1 0.562→0.750. Whole-provision reranking *hurts* (0.676→0.626) — truncation, now fixed |
| End-to-end pipeline wiring (Stages 1–7) | done — now run live, end to end (gate → retrieval → real Groq generation → verifier), `eval/run_live_generation_eval.py` |
| Generation experiment, self-composed (16 answers → verifier) | done — 1.000 pre/post citation resolution; not blinded, see below |
| Generation experiment, live + independently invoked (Groq) | done — gate passes 10/16 gold queries end-to-end, 0/16 unanswerable wrongly answered; citation resolution **0.966 pre-verifier (28/29) → 1.000 post (28/28)**, mean coverage 0.972. The one pre-verifier error (a mis-named act) was invisible to earlier revisions. V4 strips 3 quotations and accepts 5 faithful elisions — still a project-authored query set, see below |
| Judgment corpus + paragraph-level parent-document retrieval | done — 110 real Supreme Court judgments (Indian Kanoon, not ILDC directly — see below), naive-vs-parent-document chunking measured for the first time: **near-tie** (Recall@1 0.900 both, MRR@10 0.942 naive vs 0.925 parent-document), not yet reflected in the paper |
| Regression harness (pytest, pinned deps, `run_all.sh`) | done — 321 tests; paper-consistency checks now target `is_it_legal_final.tex` and the generated `round2_figures.tex` |
| Live demo (`app.py`, Streamlit) | done — see "Demo" below |

## Peer-review revision (2026-10)

A reviewer judged the claims stronger than the evidence. Each point was
checked against the code before anything was changed; all held.

- **"Precision 1.0" was resolution under a support name.** The paper now
  reports citation *resolution* (by construction) and citation *support*
  (measured on a fixed random sample of 90 surviving sentences, author-labelled,
  `eval/run_support_labels_eval.py`) separately. Grounded: 0.889 fully
  supported; ungrounded: 0.356. V5's flagging precision on real output is
  0.500, against 1.000 on its own probes.
- **Case law was claimed but not checked.** `ingest/case_citations.py`
  extracts case names and SC reporter citations and the verifier resolves them
  against the 110-judgment corpus (or the retrieved text, where the statute
  itself cites a case). Closed world: a real precedent outside the corpus is
  stripped. Title changed to "...for Indian Law".
- **theta = 0.95 was length-dependent.** One substituted word at either end
  of a 20+ token quote scored (n-1)/n and passed. V4 is now exact
  (`eval/run_quote_eval.py` length sweep). Also found: the segmenter split
  quotations at internal full stops, letting a quote's tail escape V4.
- **Fail-open parser.** `verify/residual.py` strips any reference-shaped span
  no extractor claimed. Behind the *pre-revision* parser it alone catches all
  17 citation-form faults that parser missed
  (`BEFORE_ROOT=<old checkout> python3 eval/run_failclosed_eval.py`). On real
  output it fires on 70 of 6,433 sentences; every instance is listed by
  `eval/run_live_round2_eval.py`.
- **Evaluation too thin.** Second live round (`eval/capture_live_round2.py`,
  frozen in `eval/live_round2_answers.json`): 3 models, AILA fact situations,
  grounded and ungrounded, 181 answers, with Wilson CIs (`eval/stats.py`). Its
  independent audit initially found 30 unresolved citations surviving
  verification -- three verifier defects (context from stripped sentences,
  placement past an unknown act name, case names split by the segmenter), all
  fixed; post-verification resolution is now 1.000 in every cell.
- **C4 undelivered.** AILA 2019 Task 2 with official qrels
  (`eval/run_aila_eval.py`, data in `data/aila/`): mid-table, stated as such.
  RAGAS and IL-TUR still not run; the paper says so.
- **Tables XIII/XIV disagreed** (fusion depth 50 vs 20, MRR over 20 vs 10).
  Both now measure the deployed configuration.
- **Provenance.** `eval/run_provenance_eval.py` checks the corpus against the
  Gazette of India text of the BNS, BNSS and BSA (`data/official/`);
  `eval/run_mapping_audit.py` cross-checks the UP Police mapping against a
  CAPT Bhopal table. The paper no longer calls the UP Police tables official.
- **Also found and fixed:** R2 corrupting enumerations ("...IPC)ons 302 and
  420") and gluing words after act-first citations; lenticular markers
  invisible to V1; uncheckable quotations passed instead of stripped;
  `extract_citations` memoised (verification median ~8 ms per answer, cold,
  `eval/run_latency_eval.py`).
- **Restructure.** Real-corpus results lead Section VII; the 35-provision
  development-corpus experiments are in an appendix. None of their numbers
  changed in this revision -- those probe sets could not contain any construct
  it fixed.

## Earlier revision: fixes and what they measured

An architecture scan (2026-09-23) found bugs that were costing the
verifier both precision and coverage on real text, plus retrieval
improvements. Every number below comes from `./run_all.sh --evals`; "before"
is the same scripts on the code as it stood before the scan.

**Verifier on real text.** The verifier's existing probes all run on the
35-provision toy corpus, where no sub-section, act name, year or elided
quotation ever appears. `eval/run_verifier_real_eval.py` probes it on the
real corpus in matched pairs: each fault it must strip next to the correct
use of the same construct it must keep. Several constructs come from the
independently generated Groq answers rather than being invented.

| Real-corpus probe set (31 probes + 3 marker cases) | before | after |
|---|---|---|
| Faults stripped | 12/17 | **17/17** |
| Correct citations kept | 8/14 (6 wrongly stripped) | **14/14** |
| Marker/citation contradictions repointed or flagged | 1/3 | **3/3** |
| Independent post-hoc audit on emitted answers | — | 31/31 |

What was wrong, in the original code: every sub-section pinpoint ("Article
19(1)(a)", "Section 41(1)(b) CrPC") was stripped as a *nonexistent*
provision (the corpus indexes whole sections); a section cited to an
invented act ("Section 5 of the Indian Cyber Safety Act, 2019") produced no
citation at all and passed; "IPC 1860" was read as section 1860 and
stripped; a marker placed after the full stop became its own "sentence",
leaving the claim uncited; and every quotation that marked an omission
with "…" failed V4, even when every word was verbatim. Probes R24–R31
were added after the first version of the fixes passed R01–R23: attacking
the fixes found five more gaps (a lower-case fake act, fake acts named
"Law"/"Bill", a sub-clause that exists only as a cross-reference, a
year-shaped fake section), all closed.

**Live, independently generated answers (Groq gpt-oss-20b).**

| | before | after |
|---|---|---|
| Pre-verifier citation resolution | 1.000 (27/27) | **0.966 (28/29)** |
| Post-verifier citation resolution | 1.000 (21/21) | **1.000 (28/28)** |
| Mean sentence coverage | 0.930 | **0.972** |
| Quotations stripped by V4 | 8 | 3 |
| Repealed citations with no recorded successor | 2 | 0 |

The lower pre-verifier figure is a *better measurement*, not a worse
model: the model always wrote "section 118 of the Bharatiya **Samvidhan**
Adhiniyam" (the act is the Bharatiya **Sakshya** Adhiniyam), and earlier
revisions could not see citations to unrecognised act names. Of the 8
quotations previously reported as "real quotation slips", 7 were verbatim
quotations with a marked omission; 5 omit nothing meaningful and are now
accepted, and 2 omit a legal exception (CrPC s.57's "in the absence of a
special order of a Magistrate") and are still stripped, along with 1
genuine misquotation.

**Retrieval (real corpus, 16 gold queries).**

| Stage | before | after |
|---|---|---|
| BM25 Recall@1 | 0.375 | 0.500 (stemmed) |
| Dense (general) Recall@1 | 0.500 | 0.562 (passage windows; 0.438 → 0.562 on the same corrected corpus) |
| Fusion MRR@10 | 0.580 | 0.690 |
| Full stack (fusion → rerank → in-force pref.) MRR@10 | 0.631 | **0.821** |
| Full stack Recall@1 | 0.500 | **0.750** |
| Current law ranked above repealed predecessor | 4/9 | **8/9** |

Two results went the other way and are kept as findings: stemming
**weakened** the confidence gate (held-out correct abstention 1.000 → 0.625),
so the gate stays on unstemmed BM25; and whole-provision cross-encoder
reranking **hurt** once first-stage retrieval improved (MRR 0.676 → 0.626),
which is what led to passage-window reranking.

**Corpus.** State/UT amendment tables (60, confirmed one by one) were
presented as central law; they are now removed from provision text, which
also corrected IEA §90A (it held Uttar Pradesh's §90-A). 12 inserted
Evidence Act sections that were merged into their parents are now their
own citable provisions, including §113-B (presumption as to dowry death),
which a live answer cited correctly and the old verifier stripped as
nonexistent. The full correspondence tables (1,086 verified pairs) replace
the 11 hand-tracked mappings. Corpus: 3,263 → 3,275 provisions.

## How this compares to the literature — what can and cannot be claimed

What the comparison **can** say:

- **The guarantee is a different kind of thing.** Every surveyed system
  controls citation errors statistically — an adversarial evaluator
  (Bindra et al.), confidence weighting, or partly manual review
  (Nyaya-Sathi). Here, a citation that does not resolve to the corpus, or a
  quotation that is not verbatim, cannot survive: post-verifier resolution
  is 1.000 by construction, confirmed on real independent output (28/28)
  and by an audit that re-derives it from the emitted text alone (31/31 on
  the real-corpus probes).
- **It removes errors a strong grounded pipeline still makes.** The
  independently generated answers came from a retrieval-grounded pipeline
  and still contained a mis-named act and a genuine misquotation. Both were
  removed, at a cost of 2.8% of sentences.
- **Recodification is handled at every stage.** No surveyed system
  distinguishes repealed IPC/CrPC/IEA law from its BNS/BNSS/BSA successor.
  Here retrieval prefers current law (8/9), the verifier annotates every
  repealed citation with its verified successor (9/9 on the gold pairs,
  1,084 of 1,264 repealed provisions mapped), and repair rewrites it.
- **Retrieval clears the reference range this project cited.** Rustamov et
  al.'s 0.49–0.60 Recall@1 range for comparable statutory corpora was the
  expectation; the deployed stack reaches 0.750 on this project's 16 gold
  queries (plain BM25: 0.375).

What it **cannot** say:

- It cannot claim a lower hallucination *rate* than Bindra et al.'s 6.3% or
  a higher citation accuracy than their 96.1%. Those were measured on their
  benchmark, with their definitions (which include whether a citation is the
  *right* one for the claim — this verifier checks that only through the
  advisory V5 screen, recall 0.818). No surveyed system was run on this
  project's queries.
- The query sets are small (16 answerable, 16 + 8 unanswerable) and, except
  the 8 external ones, written by this project. Differences of one or two
  queries move Recall@1 by 0.06–0.13.
- Precision 1.000 is a property of what the verifier *checks*: existence,
  in-force status, marker resolution and verbatim quotation. A real, correctly
  quoted provision attached to the wrong proposition is still only flagged
  (V5), never stripped — the boundary documented in the paper.

## Numbers the paper still states

`paper.tex` has not been edited in this revision (a final version is being
prepared separately). `tests/test_paper_consistency.py` therefore fails on
exactly four checks, each because the paper states a superseded figure:

| Paper says | Measured now | Source |
|---|---|---|
| 3,263-provision corpus | 3,275 | `eval/run_real_corpus_eval.py` |
| τ_r = 1.088 | 1.077 (held-out balanced accuracy unchanged, 0.938) | `eval/run_confidence_real_eval.py` |
| live mean coverage 0.930; "V4 fires eight times" | 0.972; V4 strips 3, accepts 5 elisions | `eval/run_live_generation_eval.py` |
| live pre-verifier resolution 1.000 | 0.966 (28/29) | same |

Figures the paper states that were already stale **before** this revision
and are not caught by those four checks (they should be updated too): RRF
fusion MRR@10 0.680 (now 0.690), reranking "+0.038 nDCG / +0.005 MRR" (the
reranking result is now +0.102 nDCG / +0.145 MRR with passage windows, and
negative without them), general dense Recall@1 0.562 (0.438 whole-provision,
0.562 with passages), grounded self-composed coverage 0.969 (0.948),
cross-reference graph 2,076 edges / 257 provisions (1,936 / 246 after
removing state-amendment text and wrong-act edges), and the "seven/eleven
parser defects" count (this revision adds the defects listed above).

## Demo

`app.py` is a Streamlit companion for presenting this project alongside the
paper — not part of the research artifact. Every result it shows is either a
**live call into the real, tested code**, an **already-recorded real
result**, or (Tab 6 only) a **live call to a real hosted LLM**, labelled as
such in each case.

    pip install -r requirements.txt
    streamlit run app.py

Six tabs:

1. **Verify Anything** — the centerpiece. Type or paste any sentence with a
   citation (real or invented) and press Verify. This is the deterministic
   layer running live, on whatever you type, with no model involved.
2. **Live Retrieval** — a real question against the real 3,275-provision
   corpus: BM25 instantly, and an opt-in side-by-side of InLegalBERT vs the
   general embedding model reproducing the paper's negative result live.
3. **Real Generated Answers** — the 16 recorded generation-experiment answers
   (see the paper's methodology note, repeated on this tab), run through the
   live verifier.
4. **Recodification Lookup** — pick an old-law citation, see its current-law
   successor and whether that mapping is verified against a real source.
5. **The Numbers** — the headline measured results as charts.
6. **Live Generation** — type any question, get a real answer from a real
   model (Groq, `openai/gpt-oss-20b`), watch the verifier check it live. Needs
   a `GROQ_API_KEY` in `.streamlit/secrets.toml` (see below); without one the
   tab explains that plainly instead of faking output. Retrieval is BM25 +
   dense fusion + reranking, gated by a confidence threshold recalibrated
   specifically for the real corpus (τ=1.077, `eval/run_confidence_real_eval.py`)
   rather than reusing the toy-corpus deployment default (τ=0.305), which was
   calibrated against 35 provisions only and reproducibly answers a query
   the real corpus should refuse (confirmed as a standing regression test in
   `tests/test_confidence_real.py`). On a held-out split the recalibrated
   gate answers 7/8 held-out answerable queries and abstains on all 8
   held-out unanswerable ones (balanced accuracy 0.938); type an
   out-of-scope question (e.g. "How is agricultural income taxed?") to see
   it abstain live instead of guessing.

   `openai/gpt-oss-20b` was chosen empirically, not assumed: the initially
   recommended `llama-3.3-70b-versatile` turned out not to exist in Groq's
   current catalog, and the larger `openai/gpt-oss-120b`, while callable,
   substituted its own native citation-annotation format for the instructed
   `[n]` markers on a live test call — output this project's regex-based
   marker extraction would silently see as zero citations, passing every
   check for the wrong reason. See `generation/generator.py`'s `GroqClient`
   docstring for the full comparison.

**Before presenting:** the dense-retrieval and InLegalBERT-comparison paths
load real models on first use (general ~30–60s, InLegalBERT ~2–4 min).
Streamlit caches them for the life of the running process, not per browser
tab — launch the app and click through Tab 2's checkboxes, Tab 3's
"Grounded" option, and Tab 6's example query once, a few minutes before you
present, so those loads (and Groq's first response) are already warm when
your teacher is watching.

## Run

    pip install -r requirements.txt
    brew install tectonic     # LaTeX toolchain, for the paper build

    ./run_all.sh              # assertions, every eval table, then the paper
    ./run_all.sh --tests      # assertions only (fast)
    ./run_all.sh --evals      # eval tables only (regenerating paper numbers)
    ./run_all.sh --paper      # rebuild paper.pdf and check the build is clean

`run_all.sh` exits nonzero if any assertion fails, any script errors, or the
paper build is not clean. A clean build is zero `Overfull \hbox` and zero
undefined references or citations **in the final pass** — tectonic reruns TeX
until `paper.aux` converges, so first-pass undefined warnings are normal and
only the last pass is inspected. Underfull boxes are loose spacing rather than
content past the margin and are not treated as failures; 82 of them come from
pre-existing wide tables in Sections II and VI. The paper step skips cleanly if
`tectonic` or `paper.tex` is absent.
Individual tables:

    python3 eval/run_normalizer_eval.py    # normaliser accuracy
    python3 eval/run_verifier_eval.py      # fault detection + worked example
    python3 eval/run_quote_eval.py         # quotation check, theta sweep
    python3 eval/run_retrieval_eval.py     # BM25 Recall@k, MRR
    python3 eval/run_inforce_eval.py       # recodification diagnostic
    python3 eval/run_repair_eval.py        # repair ablation + independent audit
    python3 eval/run_xref_eval.py          # cross-reference recovery
    python3 eval/run_xref_k_sweep.py       # graph expansion, K sweep
    python3 eval/run_support_eval.py       # support screen calibration
    python3 eval/run_support_hard.py       # support screen failure boundary
    python3 eval/run_support_xref.py       # V5 x citation graph ablation
    python3 eval/run_confidence_eval.py    # calibrated abstention, dev/test split
    python3 eval/run_real_corpus_eval.py   # real corpus: scale, sourcing, retrieval desaturation
    python3 eval/run_dense_eval.py         # dense retrieval ablation (InLegalBERT vs general)
    python3 eval/run_fusion_eval.py        # RRF fusion, BM25 + dense
    python3 eval/run_rerank_eval.py        # cross-encoder reranking (MRR + nDCG)
    python3 eval/run_generation_eval.py    # real generated answers -> verifier (self-composed)
    python3 eval/run_confidence_real_eval.py  # confidence gate, recalibrated on the real corpus
    python3 eval/run_confidence_external_eval.py  # out-of-sample check, real externally-sourced queries
    python3 eval/run_live_generation_eval.py  # real generated answers -> verifier (live, Groq)
    python3 eval/run_evasion_eval.py       # parser-evasion resistance

Each `eval/run_*.py` exposes a `report()` that both prints its table and
returns the metrics as a dict. `tests/` asserts on those dicts, so the
assertions cannot drift from the published tables — both read one computation.

## Regression harness

`python3 -m pytest` asserts explicit numeric thresholds on every number quoted
in this README and in the paper, plus the CLAUDE.md Section 2 invariants
directly: no model or network import anywhere under `verify/`, verification is
deterministic, emitted answers contain no unresolvable citation under any
verifier configuration, V1/V2/V4 strip rather than annotate, and V3/V5 annotate
without stripping.

A further group of assertions ties `paper.tex` to the harness: each takes a
figure from an eval script, formats it the way the paper writes it, and requires
it to appear in the paper source. A measurement that changes fails the suite
until the paper is updated in the same edit. (These skip if `paper.tex` is not
in the checkout.)

The suite was validated by mutation: seven deliberate defects were introduced
(V2 made annotate-only, V5 made to strip, an LLM/network import added under
`verify/`, θ relaxed to the character-era 0.90, R1 disabled, V1 removed,
abbreviation masking disabled) and **all seven were caught**. For contrast, with
V2 stripping disabled — fabricated provisions surviving into emitted output,
which breaks the project's central claim — all eleven eval scripts still exited
zero and the old `run_X.py >/dev/null && echo PASS` check reported PASS for
every one of them.

## Important caveats

* `data/seed_corpus.py` section texts are **paraphrases**, not statutory
  language, flagged `verbatim=False`. The quotation check is disabled for such
  chunks rather than silently passing them. Article 21 is the one verbatim
  entry, so V4 has a real source to test against.
* IPC↔BNS mappings are hand-entered and flagged `verified=False`. **Check them
  against the official India Code correspondence tables before relying on
  them.** The verifier surfaces this provisional status in its annotations.
* No result here measures legal accuracy. They measure pipeline behaviour.
* **V5 (support screening) is advisory and never strips.** Its scores for
  correct vs. topically-adjacent-incorrect citations overlap (margin −0.09), so
  no threshold separates them. It catches gross mismatch only. Do not present it
  as entailment checking. It is deployed at τ=0.20, flagging 9/11 wrong
  attributions with 0 correct claims flagged; `run_support_hard.py` now prints
  that operating point directly rather than leaving it to be re-derived by hand.

## Support via cross-reference

V5 alone flags **every** correct citation to a non-self-contained provision —
one whose text merely refers elsewhere ("cognizance of any offence punishable
under section 138"). Its flagging precision on that class is 0.500, not the
1.000 measured on the probes written for it.

The specified repair — suppress the flag when a one-hop reference supports the
claim — was measured before being built, and **rejected as specified**: the edge
that rescues a correct citation also rescues a topically-adjacent wrong one, so
suppression converts a correctly flagged misattribution into a silent pass. It
is implemented instead as a third verdict that annotates without suppressing.
Nothing is silently passed; nothing correct is labelled unsupported. Pass the
graph explicitly to enable it:

    CitationVerifier(db, support=True, graph=build_graph(db))

## Calibrated abstention

The verifier is silent about a query the corpus cannot answer: the retriever
still returns its top-k and every citation over them may verify perfectly while
the answer is worthless. The confidence gate declines to answer instead.

It is **not** part of the deterministic guarantee — it is a tuned threshold on a
retrieval score, which is why it lives in `retrieval/` and not `verify/`. The
signal is BM25 top-1 normalised by the summed IDF of the query's substantive
terms, because raw BM25 is not comparable across queries.

Thresholds are tuned on a dev split and reported on a held-out split. Choosing
the *signal* by dev performance would have picked raw BM25 top-1, which is
perfect on dev and makes two errors on test; the deployed signal is chosen on
the comparability argument, not the bake-off. With 8 queries per class per split
none of these differences is significant — the split is the contribution, not
the number.

## Real corpus, dense retrieval, and the pipeline

The 35-provision toy corpus is kept (it is what the fault-injection probes are
keyed to) but is no longer the only corpus. `ingest/india_code.py` assembles a
second one from real, scraped, verbatim statutory text: 3,275 provisions
across all ten target Acts (IPC, BNS, CRPC, BNSS, IEA, BSA, NIA, ITA, ICA, the
Constitution).

India Code, the intended source, blocked every automated request tried from
this environment (403/404 on handle pages and PDF bitstreams alike, while the
bare domain returned 200) — confirmed across four independently-sourced URLs,
not a one-off. Indian Kanoon was substituted; see `ingest/scrape_indiankanoon.py`
for eleven markup-variant parser defects found and fixed across **five**
rounds of adversarial audit — the first two (zero BNS sections, a wrong IT Act
section body, duplicated Constitution text) reported previously; a **third**
round, triggered by actually reading generated output against real source HTML
while building the experiment below, found two more and **corrected a
previous conclusion**: BNS §318 ("Cheating") was reported as a confirmed
*source* gap in Indian Kanoon's transcription. It wasn't — Indian Kanoon
renders it with a markup tag class the parser didn't collect, so its content
was silently merged into §317 ("Stolen property"). A related pattern absorbed
five IEA sub-sections (including §65B, one of this project's most-cited
example provisions) into their parent sections. A **fourth** round then swept
the whole corpus systematically for structural anomalies (length outliers,
duplication) rather than probing reactively, finding two more defects across
twelve provisions (an orphaned Constitution sub-section, and section-level
text double-counted against its own nested sub-sections). A **fifth** round
— the same sweep, simply re-run rather than assumed sufficient after one
pass — found four more, none caught by the length/duplication checks that
found round four's: glued-together adjacent clause text (a missing
whitespace separator), amendment history polluting marginal notes, whole
sections silently dropped when their entire rule lived only in the heading
tag (ICA §184, plus five repealed IPC sections), and pre-amendment text
leaking into a current section's body from unmarked table markup. That the
fifth round found anything at all — same method, same corpus, no new
heuristic — is itself the finding: one clean pass does not certify a sweep.
One small residual (IEA §67A, whose own source heading is malformed) and one
confirmed-but-deliberately-unfixed construct (state/UT-specific amendment
tables, which share markup with genuinely current schedule content and have
no single reliable marker separating the two) remain stated gaps.

The recodification mapping now has a matched set of three real sources:
`ingest/scrape_correspondence.py` parses UP Police's published
BNS↔IPC/BNSS↔CRPC/BSA↔IEA correspondence tables and verifies all 11 tracked
pairs against them. One (IPC§379→BNS§303) needed a real fix, not just a
lookup: the source specifies the more precise BNS§303(2), but this corpus
indexes one chunk per section, not per sub-section, so there is no
`BNS§303(2)` to point at — `ingest/india_code._pair_confirmed` now accepts a
sub-section-qualified source entry as confirming our own section-level
mapping (never the reverse).

BM25 Recall@1 on the ten-Act corpus is 0.375 — *below* the 0.49–0.60 range
this project had only been able to *cite* from other papers before (it was
0.500, inside that range, before BNSS's 531 real procedural provisions were
added and started lexically competing with CRPC's on arrest/FIR queries).
Dense retrieval, RRF fusion, and cross-encoder reranking (`retrieval/dense.py`,
`fusion.py`, `rerank.py`) are all measured on this real corpus. One result is
a genuine negative finding: **InLegalBERT underperforms both a general-purpose
embedding model and plain BM25** (whole-provision Recall@1 0.250 vs 0.438 vs
0.375; with passage windows 0.312 vs 0.562, and stemmed BM25 0.500). This is
not a bug — InLegalBERT is an MLM-pretrained checkpoint never fine-tuned for
retrieval/similarity, and raw mean-pooled embeddings from such a model are
known to be poorly suited to cosine-similarity ranking without that
fine-tuning step. RRF fusion beats both arms (MRR@10 0.690 vs 0.604 stemmed
BM25 / 0.638 passage-window dense). Reranking is the more instructive
result: scored over each provision's first 512 tokens, the cross-encoder
*lowers* MRR@10 (0.676 → 0.626) once first-stage retrieval is this good;
scored over passage windows it raises it to 0.821 (nDCG@10 0.684 → 0.786,
Recall@1 0.562 → 0.750). An earlier revision reported a small gain from
whole-provision reranking (+0.038 nDCG) -- that gain existed only because
first-stage retrieval was weaker.

`pipeline.py` wires every stage together — confidence gate, fused/reranked
retrieval, generation, verification — into one `answer_query()` call. **No LLM
API credential is available in this environment**, so `generation/generator.py`'s
real client (`AnthropicClient`) has never actually been invoked; it raises
immediately rather than fabricating output when unconfigured. `tests/test_pipeline.py`
uses a scripted fake client to prove the wiring, including that a fabricated
citation is stripped regardless of which component produced it.

## Judgment corpus and the parent-document retrieval question

`ingest/scrape_judgments.py` ingests 110 real Supreme Court judgments from
Indian Kanoon — **not ILDC directly**. The official ILDC/IL-TUR release
(Malik et al., 2021) is gated behind manual HuggingFace approval, unreachable
in this environment; the one public, ungated mirror found
(`jayadityagandham9/ILDC_35k_COMPLETE`) carries a systematic text-corruption
artifact inherited from ILDC's own original preprocessing — the substring
"con" inside words was mangled into "company" throughout ("connected" →
"companynected", "Contai" → "companytai") — and no case-name/citation/bench/
date metadata at all, just `(id, text, label)`. Ingesting corrupted judgment
text into a project whose entire thesis is quotation precision would work
against that thesis, so Indian Kanoon was substituted instead, the same
substitution and for the same reason already made for the statutory corpus
(India Code blocked automated access; Indian Kanoon is the acknowledged
underlying source for ILDC and IL-TUR themselves). This is a real, sourced,
110-judgment sample spanning several years and subject areas (bail, murder,
contract, constitutional, evidence, property, negotiable-instrument
matters) — smaller than ILDC's ~35,000, not differently sourced within
itself. **This sourcing substitution is not yet reflected in the paper.**

Each judgment is parsed into individually addressable paragraphs (Indian
Kanoon's own stable `p_N` ids), with case name, date, bench, author,
equivalent citations, and statutes referred — the last derived by running
this project's own citation normaliser (`ingest/normalize.py`) over the
judgment text, reusing tested extraction rather than a second implementation.
Rhetorical-role labelling (facts / issues / reasoning / ruling,
`segment_rhetorical_roles`) is a keyword-and-position heuristic, stated as
approximate, not a claim of IL-TUR-grade NLP accuracy, and has never itself
been checked against a labelled ground truth (IL-TUR has one; this project's
own sample doesn't use it) — a stated, open weakness of this module, not
just of the statutory scraper.

**Three real bugs, found and fixed the same day this module was built** —
an adversarial audit of a brand-new scraper the way CLAUDE.md's working
culture asks for, applied to itself rather than just to the older statutory
one:

1. **Silent content loss.** The first version of the paragraph walk
   matched only `<p id="p_N">` tags. Quoted material — a precedent's
   holding, statutory text quoted in the judgment — is rendered by Indian
   Kanoon as a *separate* tag family, `<blockquote id="blockquote_N">`,
   sitting between numbered paragraphs in the true reading order but
   invisible to that walk. Direct measurement: 8,187 blockquotes across
   105 of 110 sampled judgments (95%), 4.1M characters of content silently
   missing — the same failure shape as the statutory corpus's
   silent-section-loss bug found earlier this session, except recurring on
   nearly every judgment rather than a handful of sections, and
   disproportionately removing exactly the quoted-holding content a
   citation-precision project should care about most. Fixed by walking
   `<p>` and `<blockquote>` together in document order (confirmed neither
   nests inside the other anywhere in this corpus).
2. **Active corruption, found checking the fix for #1.** `<span
   class="hidden_text">` — CSS-hidden PDF pagination and e-filing
   digital-signature furniture — is not hidden from `get_text()`, and was
   found spliced mid-*sentence* into real prose: "...complied with all the
   Signature Not Verified Digitally signed by DEEPAK SINGH conditions. The
   technical bids..." 1,491 bare page numbers and 48 signature blocks
   across the sample. Worse than #1: not a gap, a sentence that reads as
   genuine judgment text but isn't. Fixed by stripping these spans before
   extracting each paragraph's text.
3. **A rarer third variant.** One paragraph in the sample is a
   free-standing `<span class="citetext">`, not nested in any `<p>` —
   would have been silently dropped the same way as #1 by an id-based
   allowlist that only knew about `<blockquote>`. Fixed with an inverted
   rule (include any free-floating `<span>` unless it's the confirmed
   decorative `hidden_text` class, rather than allowlisting only the one
   class already seen) — a deliberately more general fix than chasing each
   new tag name as it turns up, since the whole point of #1 and #2 is that
   "assume there are more" was the right instinct both times.

**This changed the measured result below, not just the corpus size** — see
`tests/test_judgment_corpus.py` for the regression coverage on all three.

**The RAPTOR-style design decision, measured for the first time** (CLAUDE.md
Tier 1 item 6 — previously "justified only by citation to other papers, not
by our own measurement"): `ingest/judgment_chunks.py` builds two chunkings —
NAIVE (fixed 4-paragraph windows, 4,879 chunks) and PARENT-DOCUMENT (one
paragraph per chunk, 19,377 chunks, each expandable back to its surrounding
context via `parent_context()`). `eval/run_judgment_retrieval_eval.py`
compares them on a 20-query gold set
(`eval/judgment_queries.py` — script-generated from this project's own
corpus with a fixed seed, not an externally sourced benchmark, since none
exists for this specific sample; each query is 6 distinctive terms drawn
from one real paragraph, the target being that exact paragraph). Result:
**a near-tie** — Recall@1 0.900 for both strategies, MRR@10 0.942 (naive)
vs 0.925 (parent-document). (Before the three fixes above, on the
incomplete/corrupted corpus, this same measurement had shown naive chunking
clearly winning, Recall@1 0.950 vs 0.900 — that apparent result did not
survive fixing the underlying data and is superseded, not merely refined.)
Read as a null result rather than a
confirmation either way: on this query methodology (terms drawn directly
from the target paragraph's own vocabulary), BM25 finds the right paragraph
about equally well whether it's isolated or embedded in a 4-paragraph
window, which is a weaker test than a natural-language question would be —
a real, stated limitation of the gold set, not of the chunking strategies
themselves. **Not yet folded into the paper**, which still states the
parent-document design as an unmeasured architectural choice.

## The generation experiment

`eval/run_generation_eval.py` does what a scripted fake client can't: it runs
**real, unscripted generated answers** through the unmodified verifier. With
no API credential available, the answers in `eval/generation_answers.py` were
composed directly by Claude Opus 5 — the model driving this project's build —
to all 16 gold queries, ungrounded (no retrieved context) and grounded
(strictly the real pipeline's actual top-5 chunks, under the exact prompt
contract `generation/generator.py` defines). **This is not a blinded
evaluation** — the generating model has full visibility into this project's
corpus, verifier, and gold answers — and the result should be read as a
demonstration of correct wiring under genuine text, not as an estimate of
real-world hallucination rates. Report card:

- Citation resolution (the V1/V2/V4 core guarantee) holds at **1.000
  pre- and post-verification in both conditions** — the generating model,
  answering carefully, didn't fabricate a nonexistent provision or an
  unresolvable marker either way.
- **Repair fired on genuine prose for the first time**: 9 of 16 ungrounded
  answers cited now-superseded Acts and were automatically repointed to
  current law with a transparent annotation, unprompted.
- **V3 correctly declined to strip an unmapped repealed citation**: the
  ungrounded defamation answer cited "Sections 499–502 of the IPC" as a
  range; only §499 has a recorded BNS mapping, so 500–502 were annotated as
  superseded-with-no-mapping rather than silently dropped or wrongly resolved
  — a real, honest gap in this project's mapping coverage, surfaced by real
  text.
- **V4 caught a real quotation error**: the grounded defamation answer
  quoted a source at 91.7% token match (below the 0.95 threshold) — an actual
  mistake, not a planted fault — and was correctly stripped, dropping that
  query's coverage to 0.50 while every other grounded query held at 1.00.

## The live generation experiment (independently invoked)

`eval/run_live_generation_eval.py` is a genuine methodological step up from
the round above: every query is answered by **openai/gpt-oss-20b via Groq**
— a model this project did not build, seeing nothing but the query and
whatever the real pipeline actually retrieves — not composed by this
project's own model. Only the raw completion text is frozen
(`eval/live_generation_answers.py`); the gate decision, retrieval, and
verification are all recomputed fresh on every run. The query set is still
project-authored (`eval/gold_qa.py`'s 16 queries, written before retrieval
was tuned to bound but not remove that bias) — a real advance in *generator*
independence, not yet in *query-set* independence.

Run end to end through the deployed system exactly as a real user would
experience it:

- **The real-corpus confidence gate, applied to all 32 queries**: all 16
  deliberately unanswerable queries were correctly refused (0 wrongly
  answered). Applied to the full 16-query gold set rather than only the
  held-out test split, the gate passed 10 and refused 6 — materially lower
  than the held-out split's 0.875 coverage, because several genuinely
  answerable but short, colloquially phrased questions ("What happens if a
  cheque bounces...?") score close to τ. Reported as a real coverage cost,
  not smoothed away.
- **Citation resolution: 0.966 before verification (28/29), 1.000 after
  (28/28).** The one error is a real section attributed to a non-existent
  act ("the Bharatiya Samvidhan Adhiniyam"). Earlier revisions reported
  1.000 before verification too, because citations to unrecognised act
  names were invisible to the extractor; the error was always there.
- **V4 (quotation matching)**: earlier revisions reported 8 "real
  quotation slips" here. Checking each one: 7 were verbatim quotations whose
  omissions were marked with "…" — standard legal quotation practice that
  the contiguous-match score could not accept. With ellipsis-aware checking
  (every segment verbatim, in order, within one provision, and no omitted
  stretch containing a negation, exception, condition or alternative), 5
  are accepted and 3 are still stripped: one genuine misquotation, and two
  elisions that silently drop a legal exception (CrPC s.57's "in the
  absence of a special order of a Magistrate under section 167", Art. 22's
  "excluding the time necessary for the journey"). Mean coverage is 0.972
  (was 0.930).
- Found live, from this experiment itself, not from adversarial probing: a
  12th parser-evasion defect (`ingest/normalize.py`'s D12) — "the Indian
  Penal Code (Section 500)" extracted zero citations because the act-name-
  then-section pattern required a comma or "of the," and a literal "("
  satisfied neither. Fixed; see "Known bounds" below.

The full promised ablation — a large, externally-sourced, adversarial query
set, run against an independently invoked generator — remains the single
largest piece of future work; this experiment advances generator
independence but not query-set independence.

## Out-of-sample check: real, externally-sourced unanswerable queries

`eval/run_confidence_external_eval.py` addresses the other query-set
independence gap — the confidence gate's own — with eight genuinely
externally-sourced unanswerable queries: the verbatim titles of real
question-and-answer pages on law4u.in, a public Indian legal Q&A platform,
kept alongside their source URLs in `eval/external_unanswerable_qa.py` so
"this project did not write these" is checkable, not asserted. The deployed
gate (τ=1.077) is applied exactly as calibrated — no threshold is fit to
these eight queries.

**6/8 correctly abstain (0.750)**, a real degradation from the perfect 1.000
measured on the author-written test split, reported honestly rather than
smoothed over — a suspiciously perfect score on a genuinely external set
would be less credible, not more. Both failures are explainable:

- *"What is the limitation period for filing a recovery suit in India?"*
  scores high because BNSS§514 genuinely discusses limitation periods — for
  criminal cognizance, not the civil Limitation Act the question means — a
  topically-adjacent collision, the same class of hard case the V5 support
  screen already documents, showing up here in retrieval instead.
- *"How Do the Provisions of Section 56(2) of the Income Tax Act Apply to
  Gifts?"* fails differently: this corpus's own alias table resolves "ITA"
  to the Information Technology Act, not the Income Tax Act, and generic
  procedural vocabulary ("provisions", "apply") plus that act-name
  collision scores just high enough to slip through.

Bounds stated as plainly as the result: eight queries from one platform is
a small, single-source sample, not a large adversarial corpus, and a Q&A
site plausibly shapes its own titles for search visibility — a different
kind of authorship bias than this project's own, not an absence of one.

## Known bounds

The guarantee holds only over citation forms the parser extracts. Twelve
parser defects have been found across four rounds of adversarial audit (the
first two by deliberately probing the parser; the third and fourth by
inspecting genuinely generated live text — the demo build, then the live
generation experiment itself) and repaired — see `paper.tex`'s "Parser
Evasion" subsection for the full list. Fixing the first round changed no
previously reported number, because the probe sets had avoided every broken
construct — treat that as a warning about the probes, not reassurance about
the code. The fact that live generation keeps finding constructs deliberate
probing missed is itself the strongest evidence more such defects exist;
assume they do.

## Next

1. India Code scraper → replace the seed corpus (biggest single win: makes V4
   meaningful across the corpus and de-saturates retrieval).
2. Verify the recodification mapping table against official sources.
3. Dense retrieval + RRF; then the InLegalBERT vs general-embedding ablation.
4. Wire `retrieval/xref.py` into `verify/support.py` so a low V5 score on a
   non-self-contained provision is checked for support-via-reference before
   flagging (CLAUDE.md Tier 3, item 11).
