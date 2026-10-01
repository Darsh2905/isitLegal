"""
Hand-authored test set for the citation normaliser.

Each entry: (surface form as it might appear in generated text, expected key).
Written to cover the forms actually observed in Indian legal prose: abbreviated
and expanded act names, pre- and post-posed section words, sub-section and
letter suffixes, and the "u/s" shorthand common in charge sheets.

`NEGATIVES` are strings that must yield NO citation — they exist to measure
false-positive rate, which matters because a verifier that invents references
is as dangerous as a generator that does.
"""

CASES = [
    # --- IPC, varied surface forms ---
    ("Section 420 IPC", "IPC§420"),
    ("Sec. 420 IPC", "IPC§420"),
    ("sec 420 IPC", "IPC§420"),
    ("s.420 of the Indian Penal Code", "IPC§420"),
    ("section 420, IPC", "IPC§420"),
    ("Section 420 of the Indian Penal Code, 1860", "IPC§420"),
    ("u/s 302 IPC", "IPC§302"),
    ("under section 302 of the Indian Penal Code", "IPC§302"),
    ("IPC Section 302", "IPC§302"),
    ("IPC 302", "IPC§302"),
    ("IPC §302", "IPC§302"),
    ("I.P.C. Section 376", "IPC§376"),
    ("Section 376(2) IPC", "IPC§376(2)"),
    ("section 304B of the Penal Code", "IPC§304B"),
    ("Sec 304b IPC", "IPC§304B"),

    # --- BNS ---
    ("Section 103 BNS", "BNS§103"),
    ("BNS Section 103", "BNS§103"),
    ("section 103(1) of the Bharatiya Nyaya Sanhita", "BNS§103(1)"),
    ("BNS §318", "BNS§318"),
    ("Section 318 of the Bharatiya Nyaya Sanhita, 2023", "BNS§318"),
    ("s. 63 BNS", "BNS§63"),

    # --- CrPC / BNSS ---
    ("Section 41 CrPC", "CRPC§41"),
    ("section 41 of the Code of Criminal Procedure", "CRPC§41"),
    ("Sec. 154 Cr.P.C.", "CRPC§154"),
    ("u/s 161 CrPC", "CRPC§161"),
    ("Section 35 BNSS", "BNSS§35"),
    ("section 173 of the Bharatiya Nagarik Suraksha Sanhita", "BNSS§173"),

    # --- Evidence Act / BSA ---
    ("Section 65B of the Indian Evidence Act", "IEA§65B"),
    ("Sec 65B Evidence Act", "IEA§65B"),
    ("Section 63 BSA", "BSA§63"),
    ("section 61 of the Bharatiya Sakshya Adhiniyam", "BSA§61"),

    # --- Other central acts ---
    ("Section 138 of the NI Act", "NIA§138"),
    ("Section 138 NI Act", "NIA§138"),
    ("section 138 of the Negotiable Instruments Act, 1881", "NIA§138"),
    ("Section 66A of the IT Act", "ITA§66A"),
    ("Sec. 66A Information Technology Act", "ITA§66A"),
    ("Section 43A of the Information Technology Act, 2000", "ITA§43A"),
    ("Section 73 of the Indian Contract Act", "ICA§73"),
    ("section 10 of the Contract Act", "ICA§10"),

    # --- Constitution: articles ---
    ("Article 21", "CONST§21"),
    ("Art. 21", "CONST§21"),
    ("Article 19(1)(a)", "CONST§19(1)(a)"),
    ("article 14 of the Constitution", "CONST§14"),
    ("Articles 21", "CONST§21"),

    # --- inline in running prose ---
    ("The accused was charged under Section 302 IPC for the offence.",
     "IPC§302"),
    ("This is governed by section 138 of the Negotiable Instruments Act.",
     "NIA§138"),
    ("Protection flows from Article 21 of the Constitution of India.",
     "CONST§21"),
    ("Cheating is punishable under Sec. 420 of the Indian Penal Code.",
     "IPC§420"),
]

# Strings that must produce zero citations.
NEGATIVES = [
    "The court considered the matter carefully.",
    "There were 420 pages in the record.",
    "The appeal was allowed with costs.",
    "In 2023 the legislature enacted sweeping reforms.",
    "The witness deposed that he saw nothing.",
    "Section headings were reorganised in the new code.",
    "The bench comprised three judges.",
    "A sum of Rs. 302 was recovered.",
]
