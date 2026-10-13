"""
Third adversarial round on the REAL corpus: the fail-closed revision
(2026-10), prompted by a peer review that called the parser-based design
fail-open.

Every construct below was found in this round -- by attacking the parser
again, by the quotation length sweep, or in the live gpt-oss-20b answers --
and each one passed verification in the code as it stood before the round
(eval/run_failclosed_eval.py runs that code too, for the "before" column).
Built in matched pairs as eval/verifier_real_probes.py is: a FAULT the
verifier must strip, and a correct use of the same construct it must keep.
A construct caught only by stripping its correct form as well has not been
shown to work.

Each probe: (probe_id, construct, answer, chunk_keys, expect)
  expect = "strip" -- a fault; the sentence must be removed.
  expect = "keep"  -- correct; it must survive (annotation is fine).
Chunk keys become retrieved chunks [1], [2], ... in order; an empty list means
ungrounded generation (no retrieved context at all).
"""

_P65B = ("(1) Notwithstanding anything contained in this Act, any records. "
         "information contained in an electronic record which is printed on a "
         "paper, stored, recorded or copied in optical or magnetic media")

PROBES = [
    # --- bare sections: no act named (previously never extracted) ---
    ("F01", "bare fabricated section, act only via the marker",
     "Section 7777 permits arrest without a warrant [1].", ["CRPC§41"], "strip"),
    ("K01", "bare real section, act via the marker",
     "Section 41 permits arrest without a warrant [1].", ["CRPC§41"], "keep"),
    ("F02", "bare fabricated section after an act named without a section",
     "Under the BNSS, section 7777 governs arrest [1].", ["BNSS§35"], "strip"),
    ("K02", "bare real section after an act named without a section (live output)",
     "Under the Criminal Procedure Code (CRPC) a police officer may arrest "
     "without a warrant in the circumstances set out in Section 41.",
     ["CRPC§41"], "keep"),
    ("F03", "bare fabricated sub-section, placed on the marker's provision",
     "Under sub-section (9), the police may arrest without a warrant [1].",
     ["CRPC§41"], "strip"),
    ("K03", "bare real sub-section, placed on the marker's provision",
     "Under sub-section (2), a police officer may arrest a person [1].",
     ["CRPC§41"], "keep"),

    # --- surface forms the patterns did not cover ---
    ("F04", "'s' with no full stop",
     "s 7777 IPC punishes theft [1].", ["IPC§378"], "strip"),
    ("K04", "'s' with no full stop, real section",
     "s 302 IPC punishes murder [1].", ["IPC§302"], "keep"),
    ("F05", "possessive act name",
     "IPC's section 7777 punishes theft [1].", ["IPC§378"], "strip"),
    ("K05", "possessive act name, real section",
     "IPC's section 302 punishes murder [1].", ["IPC§302"], "keep"),
    ("F06", "act name in parentheses after the section",
     "Section 7777 (IPC) punishes theft [1].", ["IPC§378"], "strip"),
    ("K06", "act name in parentheses after the section, real section",
     "Section 302 (IPC) punishes murder [1].", ["IPC§302"], "keep"),
    ("F07", "double section symbol over a range",
     "§§ 302-7777 IPC deal with homicide [1].", ["IPC§302"], "strip"),
    ("K07", "double section symbol over a range, real sections",
     "§§ 302-304 IPC deal with homicide [1].", ["IPC§302", "IPC§304"], "keep"),
    ("F08", "act abbreviation the alias table lacked ('Const') (live output)",
     "Const §7777 guarantees a right to privacy [1].", ["CONST§21"], "strip"),
    ("K08", "act abbreviation the alias table lacked, real article (live output)",
     "Const §22 protects persons who are arrested [1].", ["CONST§22"], "keep"),

    # --- pinpoint written before the citation ---
    ("F09", "fabricated clause written before the article",
     "Clause (9) of Article 22 protects persons who are arrested [1].",
     ["CONST§22"], "strip"),
    ("K09", "real clause written before the article",
     "Clause (1) of Article 22 gives an arrested person the right to consult "
     "a legal practitioner [1].", ["CONST§22"], "keep"),

    # --- forms no pattern can sensibly parse: the residual scan's job ---
    ("F10", "section number in words",
     "Section seven thousand of the IPC punishes theft [1].", ["IPC§378"], "strip"),
    ("F11", "ordinal section",
     "The 7777th section of the IPC punishes theft [1].", ["IPC§378"], "strip"),
    ("K10", "act names with enactment years (no provision cited)",
     "The BNS 2023 replaced the IPC 1860 [1].", ["BNS§103"], "keep"),
    ("K11", "two codes compared with 'vs' (not a case name)",
     "IPC vs BNS: the numbering of murder changed [1].", ["BNS§103"], "keep"),
    ("K12", "group reference with no provision number",
     "Part III of the Constitution guarantees fundamental rights [1].",
     ["CONST§21"], "keep"),

    # --- case law (previously not extracted at all) ---
    ("F12", "fabricated case name",
     "In Ramesh Kumar v. State of Haryana the Court held that every arrest "
     "needs written reasons [1].", ["CRPC§41"], "strip"),
    ("K13", "real case in the judgment corpus",
     "In Hitesh Verma v. State of Uttarakhand the Court considered insults "
     "made in a public place [1].", ["CRPC§41"], "keep"),
    ("F13", "fabricated reporter citation",
     "The point was settled in (2019) 4 SCC 101 [1].", ["CRPC§41"], "strip"),
    ("K14", "real case name with its own reporter citation",
     "Dr. Subhash Kashinath Mahajan v. State of Maharashtra, (2018) 6 SCC 454, "
     "considered anticipatory bail [1].", ["CRPC§438"], "keep"),
    ("F14", "real case name with another real case's citation",
     "Hitesh Verma v. State of Uttarakhand, AIR 2018 SC 1498, considered "
     "insults made in a public place [1].", ["CRPC§41"], "strip"),
    ("F15", "real precedent outside the 110-judgment corpus (closed world)",
     "In Arnesh Kumar v. State of Bihar the Court restricted arrests for "
     "offences punishable with up to seven years [1].", ["CRPC§41"], "strip"),

    # --- markers ---
    ("F16", "lenticular marker naming no retrieved chunk (gpt-oss style)",
     "Murder is punishable with death or imprisonment for life 【9】.",
     ["BNS§103"], "strip"),
    ("K15", "lenticular marker naming a retrieved chunk (live output)",
     "Murder is punishable with death or imprisonment for life 【1】.",
     ["BNS§103"], "keep"),

    # --- quotations ---
    ("F17", "substitution after a full stop inside a multi-sentence quotation",
     'Section 65B provides that "' + _P65B.replace("information contained",
                                                    "statute contained")
     + '" [1].', ["IEA§65B"], "strip"),
    ("K16", "verbatim multi-sentence quotation",
     'Section 65B provides that "' + _P65B + '" [1].', ["IEA§65B"], "keep"),
    ("F18", "first-word substitution in a 22-token quotation (inverts it)",
     'Article 22 provides that "Every person who is arrested shall be detained '
     'in custody without being informed, as soon as may be, of the grounds for '
     'such arrest" [1].', ["CONST§22"], "strip"),
    ("K19", "the same 22-token quotation, verbatim",
     'Article 22 provides that "No person who is arrested shall be detained '
     'in custody without being informed, as soon as may be, of the grounds for '
     'such arrest" [1].', ["CONST§22"], "keep"),
    ("F19", "ungrounded quotation with no source to check against",
     'The law says "every arrest requires a warrant from a Magistrate".',
     [], "strip"),
    ("K17", "ungrounded verbatim quotation of the provision it cites",
     'Article 21 says "No person shall be deprived of his life or personal '
     'liberty except according to procedure established by law".', [], "keep"),

    # --- found in the second live round (three models, AILA queries): the
    # residual scan stripped each of these, correctly for the faults and at a
    # coverage cost for the correct forms, until the parser was extended ---
    ("F20", "fabricated section in a list joined by 'or' (live output)",
     "A complaint lies under sections 494 or 7777 of the IPC [1].",
     ["IPC§494"], "strip"),
    ("K20", "list joined by 'or' (live output)",
     "A complaint lies under sections 494 or 495 of the IPC [1].",
     ["IPC§494", "IPC§495"], "keep"),
    ("F21", "fabricated second member of a mixed-act list (live output)",
     "Sections 198 CrPC and 7777 BNSS govern the complaint [1].",
     ["CRPC§198"], "strip"),
    ("K21", "mixed-act list, number then act (live output)",
     "Sections 198 CrPC and 219 BNSS govern the complaint [1][2].",
     ["CRPC§198", "BNSS§219"], "keep"),
    ("F22", "fabricated clause written before a bare section symbol (live output)",
     "Clauses (9) of \u00a722 exclude these safeguards for enemy aliens [1].",
     ["CONST§22"], "strip"),
    ("K22", "real clause written before a bare section symbol (live output)",
     "Clauses (3) of \u00a722 exclude these safeguards for enemy aliens [1].",
     ["CONST§22"], "keep"),
    ("K23", "verbatim quotation containing an act number (live output)",
     'Section 198 says "No Court shall take cognizance of an offence punishable '
     'under Chapter XX of the Indian Penal Code (45 of 1860) except upon a '
     'complaint made by some person aggrieved by the offence" [1].',
     ["CRPC§198"], "keep"),
    ("K24", "a date beside an act name (live output)",
     "Since 1 July 2024 (BNS), murder is punishable with death [1].",
     ["BNS§103"], "keep"),

    # --- repair rule R2 on an enumeration (garbled output, now fixed) ---
    ("K18", "superseded enumeration: annotated, not corrupted by R2",
     "The accused was charged under sections 302 and 420 of the IPC [1].",
     ["BNS§103", "BNS§318"], "keep"),
]
