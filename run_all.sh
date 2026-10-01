#!/usr/bin/env bash
#
# Full regression + evaluation run.
#
#   ./run_all.sh          pytest assertions, every eval table, then the paper
#   ./run_all.sh --tests  pytest assertions only (fast; use in a pre-commit hook)
#   ./run_all.sh --evals  eval tables only (regenerating numbers for the paper)
#   ./run_all.sh --paper  rebuild paper.pdf and check the build is clean
#
# Exits nonzero if any assertion fails or any eval script errors.
#
# NOTE ON THE ASSERTIONS. Until this harness existed, the project's regression
# check was `python3 eval/run_X.py >/dev/null && echo PASS`, which only
# confirmed the script did not crash: a script silently computing 0.00 for
# every metric still printed PASS. That is the same class of failure -- checking
# well-formedness rather than correctness -- that this project's whole argument
# is about. The pytest suite asserts explicit numeric thresholds instead.

set -uo pipefail
cd "$(dirname "$0")"

# Stale bytecode can survive an edit when the file's size is unchanged and the
# edit lands in the same second as the previous compile (Python validates .pyc
# by mtime+size). That is a real hazard here, because the adversarial audit
# workflow in CLAUDE.md Section 6 involves scripted mutate/restore cycles.
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

MODE="${1:-all}"
FAILED=0

run_tests() {
    echo "=============================================================="
    echo " REGRESSION SUITE (assertions)"
    echo "=============================================================="
    if python3 -m pytest tests/; then
        echo "  regression suite: PASS"
    else
        echo "  regression suite: FAIL"
        FAILED=1
    fi
}

run_evals() {
    echo
    echo "=============================================================="
    echo " EVALUATION TABLES"
    echo "=============================================================="
    for f in eval/run_*.py; do
        echo
        echo "##############################################################"
        echo "# $f"
        echo "##############################################################"
        if ! python3 "$f"; then
            echo "  ERROR: $f exited nonzero"
            FAILED=1
        fi
    done
}

# A clean rebuild is defined as zero `Overfull \hbox` and zero undefined
# references or citations in the FINAL pass. Tectonic reruns TeX until the aux
# file converges, so first-pass undefined warnings are expected and are not
# failures; only the last pass is inspected. Underfull boxes are loose spacing
# rather than content spilling past the margin, and are not treated as failures.
run_paper() {
    echo
    echo "=============================================================="
    echo " PAPER BUILD"
    echo "=============================================================="
    if ! command -v tectonic >/dev/null 2>&1; then
        echo "  tectonic not installed -- skipping (brew install tectonic)"
        return
    fi
    if [ ! -f paper.tex ]; then
        echo "  paper.tex not present -- skipping"
        return
    fi

    local log; log="$(mktemp)"
    if ! tectonic --print --keep-logs paper.tex >"$log" 2>&1; then
        echo "  BUILD FAILED"
        tail -30 "$log"
        FAILED=1
        rm -f "$log"
        return
    fi

    # Isolate the final TeX pass: everything after the last rerun marker.
    local start final
    start="$(grep -n 'note: Rerunning TeX' "$log" | tail -1 | cut -d: -f1)"
    [ -z "$start" ] && start=1
    final="$(mktemp)"
    tail -n "+$start" "$log" >"$final"

    local overfull undefined
    overfull="$(grep -c 'Overfull' "$final" || true)"
    undefined="$(grep -cE '(Reference|Citation) `[^'\'']*'\'' .*undefined' "$final" || true)"

    echo "  final pass: Overfull \\hbox = $overfull, undefined refs/citations = $undefined"
    if [ "$overfull" -ne 0 ] || [ "$undefined" -ne 0 ]; then
        echo "  BUILD NOT CLEAN"
        grep -E 'Overfull|(Reference|Citation) `.*undefined' "$final" | head -20
        FAILED=1
    else
        echo "  paper.pdf: clean build"
    fi
    rm -f "$log" "$final"
}

case "$MODE" in
    --tests) run_tests ;;
    --evals) run_evals ;;
    --paper) run_paper ;;
    all)     run_tests; run_evals; run_paper ;;
    *) echo "usage: $0 [--tests|--evals|--paper]" >&2; exit 2 ;;
esac

echo
echo "=============================================================="
if [ "$FAILED" -eq 0 ]; then
    echo " ALL CHECKS PASSED"
else
    echo " FAILURES PRESENT -- do not update paper.tex from this run"
fi
echo "=============================================================="
exit "$FAILED"
