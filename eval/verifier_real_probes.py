"""
Verifier probes on the REAL 3,263-provision corpus (eval/run_verifier_real_eval.py).

Every existing verifier probe set (eval/verifier_probes.py) runs on the
35-provision hand-typed toy corpus, where the one verbatim provision is Article
21 and no act name, sub-section, year or elision ever appears. These probes
exercise the checks against real statutory text, and -- following CLAUDE.md
Section 6 Class C -- are built in matched pairs: for each construct, a FAULT
the verifier must strip AND a legitimate, correct use of the same construct it
must keep. A check that catches the fault by also stripping the legitimate
form has not been shown to work.

Quotations are copied from the corpus text itself (IPC 302, BNS 103, Arts.
20-22, CrPC 57). Several constructs are taken from the independently
generated gpt-oss-20b answers (eval/live_generation_answers.py) rather than
invented: the "Bharatiya Samvidhan Adhiniyam" act name, the markers after a
full stop, and the elided Art. 22 / CrPC 57 quotations.

Each probe: (probe_id, construct, answer_sentence, chunk_keys, expect)
  expect = "strip" -- a fault; the sentence must be removed.
  expect = "keep"  -- correct law, correctly cited; it must survive (an
                      advisory annotation is fine, stripping is not).
The chunk_keys become retrieved chunks [1], [2], ... in order.
"""

PROBES = [
    # --- existence of the cited section (V2) ---
    ("R01", "fabricated section, real act",
     "Section 7777 of the Bharatiya Nyaya Sanhita punishes online fraud [1].",
     ["BNS§318"], "strip"),
    ("R02", "real section, real act",
     "Section 318 of the Bharatiya Nyaya Sanhita defines cheating [1].",
     ["BNS§318"], "keep"),

    # --- sub-section pinpoints (V2 via MetadataDB.resolve) ---
    ("R03", "real sub-section pinpoint",
     "Article 19(1)(a) of the Constitution guarantees freedom of speech and expression [1].",
     ["CONST§19"], "keep"),
    ("R04", "real sub-section pinpoint, in-force act",
     "Section 103(1) of the BNS prescribes death or imprisonment for life for murder [1].",
     ["BNS§103"], "keep"),
    ("R05", "fabricated sub-section of a real section",
     "Section 302(7) of the IPC permits a sentence of fine alone for murder [1].",
     ["IPC§302"], "strip"),
    ("R06", "fabricated sub-section of a real article",
     "Article 21(3) of the Constitution guarantees a right to bail [1].",
     ["CONST§21"], "strip"),

    # --- act names the corpus does not contain (V2 via unresolved-act refs) ---
    ("R07", "fabricated act name",
     "Section 5 of the Indian Cyber Safety Act, 2019 makes phishing punishable with seven years [1].",
     ["ITA§66"], "strip"),
    ("R08", "mis-named real act (live gpt-oss-20b output)",
     "Under section 118 of the Bharatiya Samvidhan Adhiniyam, a court may presume a dowry death [1].",
     ["BSA§118"], "strip"),
    ("R09", "correctly named act",
     "Under section 118 of the Bharatiya Sakshya Adhiniyam, a court may presume a dowry death [1].",
     ["BSA§118"], "keep"),

    # --- enactment years next to act names ---
    ("R10", "act name followed by its year",
     "The IPC 1860 was replaced by the BNS 2023 [1].",
     ["BNS§103"], "keep"),
    ("R11", "act, year, then section",
     "Section 66 of the IT Act 2000 punishes computer-related offences [1].",
     ["ITA§66"], "keep"),

    # --- verbatim quotation (V4) ---
    ("R12", "exact quotation",
     'Article 21 provides that "No person shall be deprived of his life or personal liberty except according to procedure established by law" [1].',
     ["CONST§21"], "keep"),
    ("R13", "single-word substitution inside a quotation",
     'Article 21 provides that "No person shall be deprived of his life or personal liberty except according to procedure established by statute" [1].',
     ["CONST§21"], "strip"),
    ("R14", "faithful elision, harmless words omitted (live output)",
     'Article 22 does not protect "any person who…is an enemy alien" [1].',
     ["CONST§22"], "keep"),
    ("R15", "faithful elision, harmless words omitted (live output)",
     'The authority must "communicate…the grounds on which the order has been made" [1].',
     ["CONST§22"], "keep"),
    ("R16", "elision that drops an exception (live output)",
     "“No police officer shall detain in custody a person arrested without warrant for a longer period than under all the circumstances of the case is reasonable, and such period shall not…exceed twenty-four hours” [1].",
     ["CRPC§57"], "strip"),
    ("R17", "elision that drops an exception and a negation",
     'Article 20 states that "No person shall be convicted of any offence … be subjected to a penalty greater than that which might have been inflicted" [1].',
     ["CONST§20"], "strip"),
    ("R18", "silent omission, no ellipsis marker",
     'Section 302 of the IPC states that "Whoever commits murder shall be punished with death, and shall also be liable to fine" [1].',
     ["IPC§302"], "strip"),
    ("R19", "elision stitching distant fragments of a long provision",
     'Article 22 says "No person who is arrested shall be detained in custody … the maximum period prescribed by any law made by Parliament" [1].',
     ["CONST§22"], "strip"),

    # --- markers (V1) and sentence segmentation ---
    ("R20", "marker naming no retrieved chunk",
     "Murder is punishable with death or imprisonment for life [9].",
     ["BNS§103"], "strip"),
    ("R21", "marker placed after the full stop (live output)",
     "The punishment for murder is death or imprisonment for life, and the offender may also be liable to fine. [1]",
     ["BNS§103"], "keep"),

    # --- enumerations (existing P0 behaviour, now on real text) ---
    ("R22", "fabricated section hidden in an enumeration",
     "Sections 302 and 7777 of the IPC deal with murder [1].",
     ["IPC§302"], "strip"),
    ("R23", "enumeration of real sections",
     "Sections 302 and 304 of the IPC deal with homicide [1].",
     ["IPC§302", "IPC§304"], "keep"),

    # --- second round: attacks on the checks above, found by trying to break
    # them after R01-R23 all passed (each of R24-R27 and R29 got through the
    # first implementation; see ingest/normalize.py, ingest/metadata_db.py) ---
    ("R24", "fabricated act name in lower case",
     "Section 5 of the indian cyber safety act makes phishing punishable [1].",
     ["ITA§66"], "strip"),
    ("R25", "fabricated statute called a 'Law'",
     "Section 12 of the Cyber Security Law makes hacking punishable [1].",
     ["ITA§66"], "strip"),
    ("R26", "statute outside the corpus called a 'Bill'",
     "Section 3 of the Digital Personal Data Protection Bill requires consent [1].",
     ["ITA§66"], "strip"),
    ("R27", "sub-clause that exists only as a cross-reference",
     "Article 19(2)(a) permits restrictions on speech [1].",
     ["CONST§19"], "strip"),
    ("R28", "real sub-clause, last item of a list",
     "Article 19(1)(g) protects the right to practise any profession [1].",
     ["CONST§19"], "keep"),
    ("R29", "fabricated section written like a year",
     "IPC 1999 punishes cyber stalking [1].",
     ["IPC§302"], "strip"),
    ("R30", "back-reference to the act in context",
     "Section 3 of this Sanhita defines general exceptions [1].",
     ["BNS§3"], "keep"),
    ("R31", "generic 'the new Act'",
     "Section 103 of the new Act prescribes death for murder [1].",
     ["BNS§103"], "keep"),
]

# Marker/citation consistency (advisory + repair R3): each must survive, and
# the wrong pointer must be either repointed or flagged, never silently kept.
MARKER_PROBES = [
    ("M01", "marker points at a different retrieved article",
     "Article 21 protects life and personal liberty [1].",
     ["CONST§19", "CONST§21"], "repointed"),
    ("M02", "marker points at the cited provision's successor (coherent)",
     "Section 302 of the IPC punishes murder with death or imprisonment for life [1].",
     ["BNS§103"], "untouched"),
    ("M03", "marker points at an unrelated provision; cited one not retrieved",
     "Article 21 protects life and personal liberty [1].",
     ["CONST§19"], "flagged"),
]
