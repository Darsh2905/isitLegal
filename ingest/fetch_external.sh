#!/usr/bin/env bash
# Downloads the external data the 2026-10 evaluations use (not committed):
#   data/aila/      AILA 2019 (FIRE) -- eval/run_aila_eval.py, capture_live_round2.py
#   data/official/  Gazette of India texts of the BNS, BNSS, BSA -- eval/run_provenance_eval.py
#                   CAPT Bhopal BNS->IPC correspondence table -- eval/run_mapping_audit.py
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/aila data/official
if [ ! -f data/aila/Query_doc.txt ]; then
    curl -sL -C - -o data/aila/AILA_2019_dataset.zip \
        "https://zenodo.org/api/records/4063986/files/AILA_2019_dataset.zip/content"
    (cd data/aila && unzip -q -o AILA_2019_dataset.zip)
fi
get() { [ -f "data/official/$1" ] || curl -sL -A "Mozilla/5.0" -o "data/official/$1" "$2"; }
get The_Bharatiya_Nyaya_Sanhita__2023.pdf \
    "https://prsindia.org/files/bills_acts/bills_parliament/2023/The%20Bharatiya%20Nyaya%20Sanhita,%202023.pdf"
get The_Bharatiya_Nagarik_Suraksha_Sanhita__2023.pdf \
    "https://prsindia.org/files/bills_acts/bills_parliament/2023/The%20Bharatiya%20Nagarik%20Suraksha%20Sanhita,%202023.pdf"
get The_Bharatiya_Sakshya_Adhiniyam__2023.pdf \
    "https://prsindia.org/files/bills_acts/acts_parliament/2023/The%20Bharatiya%20Sakshya%20Bill,%202023.pdf"
get capt_bhopal_bns_ipc.pdf \
    "https://www.keralaprisons.gov.in/userfiles/act-and-rules/comparison_summary_BNS_to_IPC.pdf"
echo "external data in place"
