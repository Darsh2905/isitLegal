"""
Seed statutory corpus (development scaffold).

IMPORTANT PROVENANCE NOTE
-------------------------
This is a small hand-entered corpus used to develop and test the pipeline
end-to-end before the full India Code ingestion is available. Section texts
here are PARAPHRASED SUMMARIES of the provision's effect, not verbatim
statutory language, and are marked `verbatim=False` accordingly.

Consequences, enforced elsewhere in the code:
  * The citation verifier's quotation check is DISABLED for chunks with
    verbatim=False, because you cannot string-match a quote against a
    paraphrase. Quotation verification is only meaningful once real statutory
    text is ingested.
  * No evaluation result computed on this corpus should be reported as a
    measure of legal accuracy. It measures PIPELINE behaviour only.

The `mapping` table below records IPC->BNS correspondences. These are
widely-reported correspondences entered by hand and flagged `verified=False`;
they MUST be checked against the official India Code correspondence tables
before any published claim rests on them. Correspondence across the 2023
recodification is not always one-to-one.
"""

# (act, section, marginal_note, text, repealed_by, verbatim)
SECTIONS = [
    # ---- IPC (repealed by BNS w.e.f. 1 July 2024) ----
    ("IPC", "302", "Punishment for murder",
     "Prescribes the punishment for murder, namely death or imprisonment for "
     "life, and liability to fine.", "BNS", False),
    ("IPC", "420", "Cheating and dishonestly inducing delivery of property",
     "Deals with cheating where the deceived person is dishonestly induced to "
     "deliver property or to alter or destroy a valuable security.", "BNS", False),
    ("IPC", "378", "Theft",
     "Defines theft as dishonestly taking movable property out of another "
     "person's possession without that person's consent.", "BNS", False),
    ("IPC", "379", "Punishment for theft",
     "Prescribes the punishment for theft.", "BNS", False),
    ("IPC", "375", "Rape",
     "Defines the offence of rape.", "BNS", False),
    ("IPC", "376", "Punishment for rape",
     "Prescribes punishment for the offence of rape.", "BNS", False),
    ("IPC", "304B", "Dowry death",
     "Concerns death of a woman caused by burns or bodily injury in abnormal "
     "circumstances within seven years of marriage where cruelty for dowry is "
     "shown.", "BNS", False),
    ("IPC", "499", "Defamation",
     "Defines defamation by words, signs or visible representations intended "
     "to harm reputation.", "BNS", False),

    # ---- BNS (in force) ----
    ("BNS", "103", "Punishment for murder",
     "Prescribes the punishment for murder under the Bharatiya Nyaya Sanhita.",
     None, False),
    ("BNS", "318", "Cheating",
     "Deals with cheating and dishonestly inducing delivery of property under "
     "the Bharatiya Nyaya Sanhita.", None, False),
    ("BNS", "303", "Theft",
     "Defines theft and prescribes punishment under the Bharatiya Nyaya "
     "Sanhita.", None, False),
    ("BNS", "63", "Rape",
     "Defines the offence of rape under the Bharatiya Nyaya Sanhita.",
     None, False),
    ("BNS", "64", "Punishment for rape",
     "Prescribes punishment for rape under the Bharatiya Nyaya Sanhita.",
     None, False),
    ("BNS", "80", "Dowry death",
     "Concerns dowry death under the Bharatiya Nyaya Sanhita.", None, False),
    ("BNS", "356", "Defamation",
     "Defines defamation under the Bharatiya Nyaya Sanhita.", None, False),

    # ---- CrPC / BNSS ----
    ("CRPC", "41", "When police may arrest without warrant",
     "Sets out the circumstances in which a police officer may arrest a person "
     "without a warrant and without an order from a Magistrate.", "BNSS", False),
    ("CRPC", "154", "Information in cognizable cases",
     "Provides for recording of information relating to the commission of a "
     "cognizable offence, commonly the First Information Report.", "BNSS", False),
    ("BNSS", "35", "When police may arrest without warrant",
     "Sets out circumstances for arrest without warrant under the Bharatiya "
     "Nagarik Suraksha Sanhita.", None, False),
    ("BNSS", "173", "Information in cognizable cases",
     "Provides for recording information about cognizable offences under the "
     "Bharatiya Nagarik Suraksha Sanhita.", None, False),

    # ---- Evidence Act / BSA ----
    ("IEA", "65B", "Admissibility of electronic records",
     "Governs admissibility of electronic records as evidence, subject to "
     "conditions and a certificate requirement.", "BSA", False),
    ("BSA", "63", "Admissibility of electronic records",
     "Governs admissibility of electronic records under the Bharatiya Sakshya "
     "Adhiniyam.", None, False),

    # ---- Other central acts (in force) ----
    ("NIA", "138", "Dishonour of cheque for insufficiency of funds",
     "Creates the offence of dishonour of a cheque returned unpaid for "
     "insufficiency of funds, subject to notice and time conditions.",
     None, False),
    ("ITA", "66A", "Punishment for sending offensive messages",
     "Provided punishment for sending offensive messages through communication "
     "services. STRUCK DOWN as unconstitutional in Shreya Singhal v. Union of "
     "India (2015).", None, False),
    ("ITA", "43A", "Compensation for failure to protect data",
     "Concerned compensation for failure to protect sensitive personal data by "
     "a body corporate.", None, False),
    ("ITA", "69", "Power to intercept, monitor or decrypt information",
     "Confers power to issue directions for interception, monitoring or "
     "decryption of information through any computer resource.", None, False),
    ("ICA", "73", "Compensation for loss caused by breach of contract",
     "Provides for compensation for loss or damage caused by breach of "
     "contract.", None, False),

    # ---- Provisions that are NOT semantically self-contained. ----
    # These exist to exercise cross-reference resolution: their text carries
    # almost no topical signal of its own and refers out to another provision.
    # This is the failure mode Rustamov et al. diagnose and defer to future
    # work: a dense or lexical retriever cannot reach these from a natural
    # query, because the words that describe the offence live elsewhere.
    ("IPC", "511", "Attempting to commit offences",
     "Whoever attempts to commit an offence punishable under this Code shall "
     "be punished with imprisonment of a term which may extend to one-half of "
     "the longest term provided for that offence, as provided in section 302 "
     "and section 420.", "BNS", False),
    ("BNS", "62", "Punishment for attempting to commit offences",
     "Punishment for attempting to commit offences, determined by reference "
     "to the punishment provided in section 103 of this Sanhita.", None, False),
    ("NIA", "142", "Cognizance of offences",
     "No court shall take cognizance of any offence punishable under section "
     "138 except upon a complaint made in writing within the period specified "
     "therein.", None, False),
    ("ITA", "72", "Penalty for breach of confidentiality and privacy",
     "Save as otherwise provided in this Act or any other law for the time "
     "being in force, any person who has secured access to any electronic "
     "record contrary to section 69 shall be liable to penalty.", None, False),
    ("BNSS", "36", "Procedure on arrest",
     "The procedure following an arrest made under section 35 shall be as "
     "specified in this section and in section 173.", None, False),

    # ---- Constitution ----
    # verbatim=True: this is the actual text of Article 21, short enough to
    # enter accurately. It exists so the quotation check (V4) has at least one
    # real source to match against during development.
    ("CONST", "21", "Protection of life and personal liberty",
     "No person shall be deprived of his life or personal liberty except "
     "according to procedure established by law.", None, True),
    ("CONST", "14", "Equality before law",
     "Guarantees equality before the law and equal protection of the laws "
     "within the territory of India.", None, False),
    ("CONST", "19", "Protection of certain rights regarding freedom of speech",
     "Protects freedoms including speech and expression, subject to reasonable "
     "restrictions.", None, False),
    ("CONST", "22", "Protection against arrest and detention",
     "Provides safeguards to persons arrested or detained, including the right "
     "to be informed of grounds of arrest and to consult a legal practitioner.",
     None, False),
]

# Special status annotations beyond simple repeal.
STATUS_NOTES = {
    "ITA§66A": "Struck down as unconstitutional by the Supreme Court in "
               "Shreya Singhal v. Union of India (2015). No longer valid law.",
}

# IPC -> BNS style correspondences across the 2023 recodification.
# verified=False: entered by hand, PENDING confirmation against the official
# India Code correspondence tables.
MAPPING = [
    # (old_key, new_key, verified)
    ("IPC§302", "BNS§103", False),
    ("IPC§420", "BNS§318", False),
    ("IPC§378", "BNS§303", False),
    ("IPC§379", "BNS§303", False),
    ("IPC§375", "BNS§63",  False),
    ("IPC§376", "BNS§64",  False),
    ("IPC§304B", "BNS§80", False),
    ("IPC§499", "BNS§356", False),
    ("CRPC§41", "BNSS§35", False),
    ("CRPC§154", "BNSS§173", False),
    ("IEA§65B", "BSA§63",  False),
]
