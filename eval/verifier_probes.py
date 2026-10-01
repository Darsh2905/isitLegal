"""
Probe set for the deterministic verifier.

Each probe is a synthetic generated answer with a KNOWN ground-truth label:
does it contain a citation fault the verifier is specified to catch?

This measures the verifier as a detector -- its precision and recall against
planted faults. It does NOT measure end-to-end system hallucination, which
requires a real generator over the real corpus.

Fault types planted:
  F1 unresolved [n] marker      (cites a chunk that was never retrieved)
  F2 non-existent provision     (fabricated section number)
  F3 repealed provision         (cites IPC post-recodification)
  F4 unsupported quotation      (quoted text absent from source)
"""

# Retrieved context available to the generator in every probe.
# (index, provision key)
CONTEXT = [
    (1, "IPC§420"),
    (2, "BNS§318"),
    (3, "NIA§138"),
    (4, "CONST§21"),
    (5, "BNSS§35"),
]

# (id, answer_text, has_fault, fault_types, note)
PROBES = [
    # ---------- CLEAN: should pass untouched ----------
    ("clean-1",
     "Cheating is addressed by the Bharatiya Nyaya Sanhita [2]. "
     "The provision covers dishonest inducement to deliver property [2].",
     False, [], "valid markers, in-force provision"),

    ("clean-2",
     "Dishonour of a cheque for insufficiency of funds is an offence under "
     "Section 138 of the Negotiable Instruments Act [3].",
     False, [], "in-force provision, valid marker"),

    ("clean-3",
     "Article 21 protects life and personal liberty [4]. "
     "Arrest powers are governed by Section 35 BNSS [5].",
     False, [], "two in-force provisions"),

    ("clean-4",
     "The answer depends on the facts of the case [1].",
     False, [], "valid marker, no legal reference"),

    # ---------- F1: unresolved marker ----------
    ("f1-a",
     "Cheating is punishable under the Bharatiya Nyaya Sanhita [9].",
     True, ["F1"], "marker [9] never retrieved"),

    ("f1-b",
     "Section 138 of the NI Act creates the offence [3]. "
     "A further condition applies [7].",
     True, ["F1"], "second sentence cites unretrieved [7]"),

    ("f1-c",
     "The provision is clear [12] and admits no exception [13].",
     True, ["F1"], "two unresolved markers"),

    # ---------- F2: non-existent provision ----------
    ("f2-a",
     "This conduct is punishable under Section 999 IPC [1].",
     True, ["F2"], "fabricated section number"),

    ("f2-b",
     "Liability arises under Section 4321 of the Bharatiya Nyaya Sanhita [2].",
     True, ["F2"], "fabricated BNS section"),

    ("f2-c",
     "The matter falls under Article 999 of the Constitution [4].",
     True, ["F2"], "fabricated article"),

    ("f2-d",
     "Section 138 NI Act applies [3], as does Section 8888 IPC [1].",
     True, ["F2"], "one valid, one fabricated"),

    # ---------- F3: repealed provision (annotate) ----------
    ("f3-a",
     "Cheating is punishable under Section 420 IPC [1].",
     True, ["F3"], "IPC superseded by BNS"),

    ("f3-b",
     "Arrest without warrant is governed by Section 41 CrPC [5].",
     True, ["F3"], "CrPC superseded by BNSS"),

    ("f3-c",
     "Electronic records are admissible under Section 65B of the Indian "
     "Evidence Act [1].",
     True, ["F3"], "IEA superseded by BSA"),

    ("f3-d",
     "Offensive messages are punishable under Section 66A of the IT Act [1].",
     True, ["F3"], "struck down in Shreya Singhal"),

    # ---------- mixed ----------
    ("mix-a",
     "Section 420 IPC covers cheating [1]. "
     "A related offence appears in Section 7777 IPC [9].",
     True, ["F2", "F3", "F1"], "repealed + fabricated + unresolved marker"),

    ("mix-b",
     "Section 138 NI Act applies [3]. "
     "Article 21 is also engaged [4]. "
     "Section 5555 BNS is relevant [2].",
     True, ["F2"], "two clean sentences, one fabricated"),
]

# ---------- F4: quotation checks (require a verbatim source) ----------
QUOTE_PROBES = [
    ("q-clean-1",
     'Article 21 provides that "No person shall be deprived of his life or '
     'personal liberty except according to procedure established by law" [4].',
     False, [], "exact verbatim quotation"),

    ("q-clean-2",
     'The Constitution states that "no person shall be deprived of his life '
     'or personal liberty" [4].',
     False, [], "exact partial quotation, different case"),

    ("q-f4-a",
     'Article 21 provides that "No person shall be deprived of property or '
     'wealth except by due compensation of law" [4].',
     True, ["F4"], "fabricated quotation attributed to Art 21"),

    ("q-f4-b",
     'Article 21 guarantees that "every citizen shall have an absolute and '
     'unqualified right to personal liberty at all times" [4].',
     True, ["F4"], "invented expansive quotation"),

    ("q-f4-c",
     'The provision reads "No person shall be deprived of his life or personal '
     'liberty except according to procedure established by statute" [4].',
     True, ["F4"], "near-miss: 'statute' substituted for 'law'"),
]

# ---------- R1 probes: correct claim, misattributed marker ----------
# These isolate the case repair R1 exists for and which the original probe set
# failed to cover: the generator names a provision that WAS retrieved, states
# something true about it, but attaches the wrong marker. Strip-only deletes a
# recoverable true claim; repair repoints the marker.
REPAIR_PROBES = [
    ("r1-a",
     "Section 138 of the NI Act creates the offence of cheque dishonour [7].",
     True, ["F1"], "correct claim about retrieved NIA138, wrong marker"),

    ("r1-b",
     "Article 21 protects life and personal liberty [8].",
     True, ["F1"], "correct claim about retrieved CONST21, wrong marker"),

    ("r1-c",
     "Cheating is dealt with by Section 318 BNS [11]. "
     "Arrest powers appear in Section 35 BNSS [12].",
     True, ["F1"], "two correct claims, both markers wrong"),

    ("r1-d",
     "Section 138 NI Act applies here [3]. "
     "Article 21 is also engaged [9].",
     True, ["F1"], "one correct marker, one repairable"),

    # Not repairable: no provision named, so nothing to repoint at.
    ("r1-neg",
     "The position is well settled in law [9].",
     True, ["F1"], "unresolved marker, no provision named -- must still strip"),
]

# ---------- V5 probes: citation resolves but does not support the claim ----------
# Every citation below is REAL, in force, and correctly retrieved. Checks V1-V4
# pass on all of them. Only topical support distinguishes the two groups.
SUPPORT_PROBES = [
    # -- supported: claim matches the provision's subject matter --
    ("s-ok-1", "Dishonour of a cheque for insufficiency of funds is an offence [1].",
     "NIA§138", False, "claim matches s.138 subject matter"),
    ("s-ok-2", "No person shall be deprived of life or personal liberty except by "
     "procedure established by law [2].", "CONST§21", False, "matches Art 21"),
    ("s-ok-3", "A cheque returned unpaid can attract criminal liability if notice "
     "conditions are satisfied [1].", "NIA§138", False, "paraphrased but on-topic"),
    ("s-ok-4", "Police may arrest a person without a warrant in specified "
     "circumstances [3].", "BNSS§35", False, "matches arrest provision"),
    ("s-ok-5", "Equality before the law is guaranteed within the territory of "
     "India [4].", "CONST§14", False, "matches Art 14"),

    # -- unsupported: real citation, unrelated claim --
    ("s-bad-1", "Murder is punishable with death or imprisonment for life [1].",
     "NIA§138", True, "murder claim cited to cheque provision"),
    ("s-bad-2", "The right to own property is guaranteed [2].",
     "CONST§21", True, "property claim cited to Art 21"),
    ("s-bad-3", "Dowry death within seven years of marriage is covered [1].",
     "NIA§138", True, "dowry claim cited to cheque provision"),
    ("s-bad-4", "Electronic records are admissible in evidence subject to a "
     "certificate [3].", "BNSS§35", True, "evidence claim cited to arrest provision"),
    ("s-bad-5", "Theft of movable property without consent is an offence [4].",
     "CONST§14", True, "theft claim cited to equality article"),
    ("s-bad-6", "Defamation by spoken words harming reputation is punishable [1].",
     "NIA§138", True, "defamation claim cited to cheque provision"),
]

# ---------- V5 hard probes: topically ADJACENT but incorrect citations ----------
# The realistic failure mode. Each wrong citation is drawn from the same subject
# area as the claim, so lexical overlap is high and the screen is under genuine
# pressure. These probe the boundary of what a non-semantic method can do.
SUPPORT_HARD = [
    # correct attributions among closely-related provisions
    ("h-ok-1", "Theft is dishonestly taking movable property without consent [1].",
     "IPC§378", False, "definition claim -> definition section"),
    ("h-ok-2", "The punishment for rape is prescribed by statute [1].",
     "BNS§64", False, "punishment claim -> punishment section"),
    ("h-ok-3", "The procedure to be followed after an arrest is laid down [1].",
     "BNSS§36", False, "procedure claim -> procedure section"),
    ("h-ok-4", "Attempting to commit an offence is itself punishable [1].",
     "IPC§511", False, "attempt claim -> attempt section"),

    # adjacent but wrong: same subject area, wrong provision
    ("h-bad-1", "Theft is dishonestly taking movable property without consent [1].",
     "IPC§379", True, "definition claim -> punishment section"),
    ("h-bad-2", "Rape is defined by reference to absence of consent [1].",
     "BNS§64", True, "definition claim -> punishment section"),
    ("h-bad-3", "The circumstances permitting arrest without warrant are set out [1].",
     "BNSS§36", True, "when-arrest claim -> procedure section"),
    ("h-bad-4", "Murder is punishable with imprisonment for life [1].",
     "IPC§511", True, "murder claim -> attempt section"),
    ("h-bad-5", "Cognizance of a cheque dishonour offence requires a written "
     "complaint [1].", "NIA§138", True, "cognizance claim -> offence section"),
]

# ---------- Evasion probes: constructs that defeated earlier implementations ----------
# Each conceals a fabricated or superseded reference inside a syntactic form the
# verifier previously mis-parsed. These are adversarial in the specific sense
# that they target the PARSER rather than the checks: if a citation is never
# extracted, no check can ever fire on it, and the guarantee is void without any
# error being raised. All were undetected before the parser fixes.
EVASION_PROBES = [
    ("ev-list-1",
     "The accused was charged under sections 302 and 7777 of the IPC [1].",
     True, ["fabricated section concealed in an enumeration"]),
    ("ev-list-2",
     "Liability arises under Ss. 420 and 9999 IPC [1].",
     True, ["fabricated section in an abbreviated enumeration"]),
    ("ev-list-3",
     "Articles 14, 21 and 999 are engaged [4].",
     True, ["fabricated article in an enumerated list"]),
    ("ev-abbrev-1",
     "He was charged under Sec. 8888 IPC. The trial proceeded [1].",
     True, ["fabricated section before an abbreviation-induced split"]),
    ("ev-abbrev-2",
     "See Ram v. State. Liability arises under s. 5555 IPC [1].",
     True, ["fabricated section after a case-citation abbreviation"]),
    ("ev-quote-1",
     'Article 21 provides that "no person shall be deprived of property rights '
     'under any circumstances whatsoever [4].',
     True, ["fabricated quotation with no closing mark"]),
    ("ev-repealed-1",
     "The charges fall under sections 302 and 420 of the IPC [1].",
     True, ["two superseded provisions inside an enumeration"]),

    # controls: same syntactic forms, nothing fabricated
    ("ev-ok-1",
     "Charges were framed under sections 318 and 103 of the BNS [2].",
     False, ["valid enumeration, in-force provisions"]),
    ("ev-ok-2",
     "Refer to Sec. 138 NI Act. Notice is required [3].",
     False, ["valid citation across an abbreviation"]),

    # ---- second adversarial audit of the parser -------------------------
    # Twelve further evasion constructs, each concealing a fabricated section
    # in a form the parser did not extract at all. Found by asking what
    # syntax a generator could emit that no pattern covers, rather than by
    # extending the probes above -- which is why none of them appeared there.
    ("ev-actlist-1",
     "See BNS sections 318 and 7777 for the applicable offence [2].",
     True, ["fabricated section in an ACT-FIRST enumeration"]),
    ("ev-actlist-2",
     "Liability arises under IPC ss. 420 and 9999 [1].",
     True, ["fabricated section in an abbreviated act-first enumeration"]),
    ("ev-slash-1",
     "Punishable under sections 318/7777 BNS [2].",
     True, ["fabricated section joined by a slash"]),
    ("ev-slash-2",
     "The charge is under BNS 318/7777 [2].",
     True, ["fabricated section in a bare act-first slash enumeration"]),
    ("ev-semi-1",
     "Charges lie under sections 318; 7777 of the BNS [2].",
     True, ["fabricated section joined by a semicolon"]),
    ("ev-rw-1",
     "The offence falls under sections 318 r/w 7777 BNS [2].",
     True, ["fabricated section joined by the r/w conjunction"]),
    ("ev-md-1",
     "Cheating is punishable under Section **7777** BNS [2].",
     True, ["fabricated section wrapped in markdown emphasis"]),
    ("ev-md-2",
     "Cheating is punishable under Section *7777* BNS [2].",
     True, ["fabricated section wrapped in single-asterisk emphasis"]),
    ("ev-fn-1",
     "Cheating is punishable under Section 7777[2] BNS [2].",
     True, ["fabricated section separated from its act by a footnote marker"]),
    ("ev-paren-1",
     "Cheating is punishable under Section (7777) BNS [2].",
     True, ["fabricated section wrapped in parentheses"]),
    ("ev-zwsp-1",
     "Cheating is punishable under Section 7777\u200bBNS [2].",
     True, ["fabricated section hidden by a zero-width space"]),
    ("ev-tick-1",
     "Cheating is punishable under `Section 7777 BNS` [2].",
     True, ["fabricated section inside a backtick code span"]),

    # controls for the constructs above: same syntax, all provisions real
    # and in force, so any flag here is a false positive.
    ("ev-ok-3",
     "Charges were framed under BNS sections 318 and 103 [2].",
     False, ["valid act-first enumeration"]),
    ("ev-ok-4",
     "Charges were framed under sections 318/103 BNS [2].",
     False, ["valid slash enumeration"]),
    ("ev-ok-5",
     "Charges were framed under sections 318 r/w 103 BNS [2].",
     False, ["valid r/w enumeration"]),
    ("ev-ok-6",
     "Cheating is punishable under Section **318** BNS [2].",
     False, ["valid citation wrapped in markdown emphasis"]),

    # ---- third adversarial audit: found live, not by extending probes -----
    # Discovered inspecting a genuinely live-generated app.py Tab 6 answer
    # during the demo build, not by adversarially auditing the parser --
    # the first defects in this project found through real usage rather than
    # deliberate construction, which is itself evidence that a
    # twice-audited parser can still hide a gap only real generation
    # surfaces. See tests/test_real_corpus.py for the more severe half of
    # this finding (a false VALIDATION against the wrong real provision,
    # only reproducible against the real corpus -- the 35-provision toy
    # corpus used here has no same-numeral collision pair to demonstrate it).
    ("ev-actlist-repeat-1",
     "The offence falls under the Bharatiya Nyaya Sanhita §318, §7777 [2].",
     True, ["fabricated section, act-first list with the symbol repeated "
            "per item instead of dropped after the first"]),
    ("ev-hyphensuffix-ok-1",
     "Failure to protect sensitive data attracts compensation under "
     "Section 43-A of the Information Technology Act [1].",
     False, ["valid citation, hyphenated compound suffix -- the canonical "
             "key must drop the hyphen (\"ITA§43A\") to match how every key "
             "in the corpus is actually stored, or this flags a real, "
             "in-force citation as fabricated"]),

    # ---- fourth adversarial audit: found live, from the actual live-
    # generation experiment (eval/run_live_generation_eval.py), not from
    # extending these probes -- the second round in this project found via
    # real usage rather than deliberate construction. Groq's real answer to
    # "Is defamation a criminal offence in India?" wrote "the Indian Penal
    # Code (Section 500)" and extracted zero citations: P3 (act name, then
    # a section reference) required a comma or "of the" between the act and
    # the section word, and a literal "(" is neither.
    ("ev-actparen-1",
     "The offence falls under the Bharatiya Nyaya Sanhita (Section 7777) [2].",
     True, ["fabricated section, act name followed by a parenthesised "
            "section reference"]),
    ("ev-actparen-ok-1",
     "Cheating is punishable under the Bharatiya Nyaya Sanhita (Section 318) [2].",
     False, ["valid citation, act name followed by a parenthesised section "
             "reference"]),
]


# ---------------------------------------------------------------------------
# SUPPORT_XREF -- correct and incorrect citations to NON-SELF-CONTAINED
# provisions, the class retrieval/xref.py exists to handle.
#
# A provision whose text merely refers elsewhere ("cognizance of any offence
# punishable under section 138") carries almost no topical vocabulary of its
# own. A claim about the SUBSTANCE of the offence, correctly cited to it,
# therefore scores near zero on V5 and is flagged as unsupported -- a false
# alarm produced by the citation being correct in exactly the way the graph
# module was built to handle.
#
# This set was constructed BEFORE the fix, following the procedure used for
# SUPPORT_HARD: build the cases that should break the mechanism, measure, then
# change the mechanism. The negatives matter as much as the positives -- they
# are wrong citations to the SAME provisions, and a repair that rescues the
# positives by rescuing these too has bought recall with precision.
#
# Format: (id, claim, cited_key, is_wrong_attribution, note)
# ---------------------------------------------------------------------------
SUPPORT_XREF = [
    # Correct citations whose support lives one hop away.
    ("x-ok-1",
     "A cheque returned unpaid for insufficiency of funds can only be pursued "
     "on a written complaint.",
     "NIA§142", False, "cognizance section; offence defined in s.138"),
    ("x-ok-2",
     "Improperly obtaining decrypted information gathered through lawful "
     "monitoring attracts a penalty.",
     "ITA§72", False, "penalty section; power defined in s.69"),
    ("x-ok-3",
     "After a warrantless arrest the police must follow the prescribed steps.",
     "BNSS§36", False, "procedure section; arrest power in s.35"),
    ("x-ok-4",
     "An unsuccessful murder attempt draws a penalty derived from the "
     "sentence for killing.",
     "BNS§62", False, "attempt section; punishment fixed by s.103"),
    ("x-ok-5",
     "Trying but failing to dishonestly induce someone to hand over property "
     "is itself punishable.",
     "IPC§511", False, "attempt section; substance in s.420"),

    # Wrong citations to the same provisions. The graph must not rescue these.
    ("x-bad-1",
     "Dowry death occurring within seven years of marriage is covered.",
     "NIA§142", True, "dowry claim cited to the cheque cognizance section"),
    ("x-bad-2",
     "Equality before the law is guaranteed within the territory of India.",
     "BNS§62", True, "equality claim cited to the attempt section"),
    ("x-bad-3",
     "Defamation by spoken words harming reputation is an offence.",
     "ITA§72", True, "defamation claim cited to the confidentiality penalty"),
    ("x-bad-4",
     "Electronic records are admissible subject to a certificate.",
     "BNSS§36", True, "evidence claim cited to the arrest-procedure section"),
    ("x-bad-5",
     "Dishonestly taking movable property without consent is theft.",
     "IPC§511", True, "theft claim cited to the attempt section"),
    ("x-bad-6",
     "Murder is punished with death or imprisonment for life.",
     "IPC§511", True, "murder claim cited to the attempt section that "
                      "expressly refers to s.302 -- the adversarial case"),
]
