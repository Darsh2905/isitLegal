"""
Recorded generation-experiment answers (Tier 2, item 10 -- the paper's
promised headline ablation, run for the first time).

WHAT THIS IS. No LLM API credential is available in the development
environment (see generation/generator.py). Rather than leave the pre/post-
verifier hallucination comparison entirely unmeasured, this module records
REAL generated answers -- composed by Claude Opus 5, the same model driving
this research session, invoked directly rather than through the
generation/generator.py API interface -- to the 16 gold_qa.py queries, under
two conditions:

  UNGROUNDED -- answered from the model's own parametric knowledge, with no
               retrieved context supplied. Analogous to the paper's "ungrounded
               LLM as the zero baseline" (Section on Ablation Design).
  GROUNDED   -- answered strictly from the actual top-5 chunks the real
               pipeline (BM25 + dense, RRF-fused, cross-encoder reranked; see
               retrieval/rerank.py) retrieved for that query against the real
               3,251-provision corpus, under the exact system prompt
               generation/generator.py defines (cite every claim with [n],
               quote verbatim in quotation marks, no outside knowledge, say so
               if the chunks don't answer the question).

METHODOLOGICAL LIMITATION, STATED PLAINLY. This is NOT a blinded, independent
evaluation. The generating model is the same session that built the corpus,
the verifier, and the gold answer key, and has full knowledge of all three
while composing these answers. This is a real and serious limitation on how
much these specific numbers should be trusted as an unbiased measurement of
"how an LLM behaves" in general, as opposed to "how this pipeline's wiring
behaves when fed genuine, unscripted natural-language output" -- which is what
it does demonstrate, and which is more than the scripted-fake-client tests in
tests/test_pipeline.py could. A true replication requires an independently
invoked model (via generation/generator.py's AnthropicClient, once credentials
exist) with no visibility into this project's gold answers or verifier
internals. Until then, read every number derived from this module as a
proof-of-concept demonstration, not a robust estimate.

WHY RECORD RATHER THAN GENERATE LIVE, EVERY RUN. These answers were composed
once, deliberately, as an act of genuine (if not blinded) generation -- not
templated, not created to hit a target pass/fail outcome. Recording them as
fixed strings makes every downstream number reproducible by construction
(CLAUDE.md invariant 6: every number must be reproducible by running a script
in eval/), the same guarantee every other probe set in this project already
has, and prevents a re-run from silently drifting because a different
"generation" happened to be composed differently.

The UNGROUNDED condition draws on real prior knowledge and is a FAVOURABLE
case for that arm, not a neutral one: the generating model has spent this
entire session immersed in the exact 2023 recodification the corpus concerns,
and the gold query set skews toward well-known, heavily-discussed landmark
provisions (murder, Article 21, cheque dishonour, FIR registration). A naive
ungrounded call from a model without that specific immersion, or on a query
distribution skewed toward obscure provisions, would be expected to hallucinate
considerably more. Report the ungrounded hallucination rate below as a LOWER
BOUND, not a representative estimate.
"""

# Each entry: (qid, condition, chunks_used_or_None, answer_text)
#   condition in {"ungrounded", "grounded"}
#   chunks_used: list of Chunk(index, key, text, verbatim) for the grounded
#                condition (needed by CitationVerifier.verify), None for
#                ungrounded (verified against retrieved=[]).

UNGROUNDED_ANSWERS = {
    "q01": (
        "Yes. Cheating that dishonestly induces a person to deliver property "
        "is an offence under Section 318 of the Bharatiya Nyaya Sanhita, 2023 "
        "(the successor to Section 420 of the Indian Penal Code, 1860), "
        "punishable with imprisonment of up to seven years and a fine."
    ),
    "q02": (
        "Murder is punishable under Section 103 of the Bharatiya Nyaya "
        "Sanhita, 2023 (previously Section 302 of the Indian Penal Code) with "
        "death or imprisonment for life, along with a fine."
    ),
    "q03": (
        "Section 138 of the Negotiable Instruments Act, 1881 makes dishonour "
        "of a cheque for insufficiency of funds a criminal offence, "
        "punishable with imprisonment of up to two years, a fine of up to "
        "twice the cheque amount, or both, provided the cheque was presented "
        "within its validity period, a demand notice was issued within 30 "
        "days of the dishonour, and the drawer failed to pay within 15 days "
        "of receiving that notice."
    ),
    "q04": (
        "Yes, in specified circumstances. Section 35 of the Bharatiya Nagarik "
        "Suraksha Sanhita, 2023 (previously Section 41 of the Code of "
        "Criminal Procedure, 1973) allows a police officer to arrest without "
        "a warrant for a cognizable offence and in other listed situations, "
        "such as being in possession of suspected stolen property or being a "
        "proclaimed offender."
    ),
    "q05": (
        "Article 21 of the Constitution of India provides that no person "
        "shall be deprived of life or personal liberty except according to "
        "procedure established by law."
    ),
    "q06": (
        "Yes. Section 63 of the Bharatiya Sakshya Adhiniyam, 2023 (previously "
        "Section 65B of the Indian Evidence Act, 1872) makes electronic "
        "records admissible without further proof of the original, provided "
        "the statutory conditions about the computer's operation are "
        "satisfied and, ordinarily, a certificate under the section is "
        "produced."
    ),
    "q07": (
        "Yes, this can amount to theft. Section 303 of the Bharatiya Nyaya "
        "Sanhita, 2023 (previously Sections 378 and 379 of the Indian Penal "
        "Code) defines theft as dishonestly taking movable property out of a "
        "person's possession without their consent, and punishes it with "
        "imprisonment."
    ),
    "q08": (
        "Under Section 73 of the Indian Contract Act, 1872, the party who "
        "suffers a breach is entitled to compensation for loss or damage "
        "that naturally arose from the breach, or that the parties knew when "
        "contracting was likely to result from it. Compensation is not "
        "given for remote or indirect loss. If the contract specifies a sum "
        "payable on breach, Section 74 caps recoverable compensation at that "
        "stipulated amount."
    ),
    "q09": (
        "Section 66A of the Information Technology Act, 2000, which "
        "criminalised sending offensive messages by computer resource, was "
        "struck down as unconstitutional by the Supreme Court in Shreya "
        "Singhal v. Union of India (2015) for violating freedom of speech. "
        "It is not a valid basis for prosecution, though other provisions "
        "(for example those addressing obscenity, defamation, or criminal "
        "intimidation) may still apply depending on the content of the "
        "message."
    ),
    "q10": (
        "Article 14 of the Constitution provides that the State shall not "
        "deny to any person equality before the law or the equal protection "
        "of the laws within the territory of India."
    ),
    "q11": (
        "Article 22 of the Constitution provides safeguards for a person who "
        "is arrested: the right to be informed of the grounds of arrest, the "
        "right to consult and be defended by a legal practitioner of their "
        "choice, and the right to be produced before a magistrate within "
        "twenty-four hours of arrest, excluding travel time. Separate, "
        "narrower safeguards apply to preventive detention under the later "
        "clauses of the same Article."
    ),
    "q12": (
        "Yes. Defamation is a criminal offence under Section 356 of the "
        "Bharatiya Nyaya Sanhita, 2023 (previously Sections 499-502 of the "
        "Indian Penal Code), punishable with simple imprisonment of up to "
        "two years, a fine, or both."
    ),
    "q13": (
        "Under Section 173 of the Bharatiya Nagarik Suraksha Sanhita, 2023 "
        "(previously Section 154 of the Code of Criminal Procedure, 1973), "
        "information about a cognizable offence given orally to an officer "
        "in charge of a police station must be reduced to writing, read over "
        "to the informant, and signed by them, and a copy must be given to "
        "the informant free of cost."
    ),
    "q14": (
        "Section 80 of the Bharatiya Nyaya Sanhita, 2023 (previously Section "
        "304B of the Indian Penal Code) defines dowry death: where a "
        "married woman's death occurs within seven years of marriage under "
        "abnormal circumstances, and it is shown she was subjected to "
        "cruelty or harassment for dowry shortly before her death, her "
        "husband or his relatives are deemed to have caused the death. It is "
        "punishable with imprisonment of not less than seven years, "
        "extendable to imprisonment for life."
    ),
    "q15": (
        "Yes, under specified conditions. Section 69 of the Information "
        "Technology Act, 2000 empowers the Central or State Government to "
        "direct interception, monitoring, or decryption of information "
        "through a computer resource where necessary for reasons such as the "
        "sovereignty or security of the State or public order, subject to "
        "procedures and safeguards that are separately prescribed."
    ),
    "q16": (
        "Under Section 64 of the Bharatiya Nyaya Sanhita, 2023 (previously "
        "Section 376 of the Indian Penal Code), rape is punishable with "
        "rigorous imprisonment of not less than ten years, extendable to "
        "imprisonment for life, along with a fine. Enhanced minimum "
        "sentences apply for specified aggravating circumstances, such as "
        "the age of the victim or repeat offences."
    ),
}


# ---------------------------------------------------------------------------
# GROUNDED condition. Each answer is composed strictly from the chunks in
# GROUNDED_CHUNKS[qid] -- the ACTUAL top-5 output of the real pipeline
# (BM25 + dense, RRF-fused, cross-encoder reranked; retrieval/rerank.py)
# against the real 3,251-provision corpus, under the exact prompt contract
# generation/generator.py defines: cite every claim with [n], quote verbatim
# in quotation marks, no outside knowledge, say so if the chunks don't
# answer the question. Retrieval quality varies genuinely across these 16
# queries -- some got the exact on-point provision at rank 1, several got
# only adjacent or procedural provisions, and one (q13) got nothing relevant
# at all. That variation was not curated or filtered; it is what the pipeline
# actually returned, and the answers below were composed to reflect it
# honestly rather than to reproduce the gold key regardless of what was
# retrieved.
# ---------------------------------------------------------------------------

GROUNDED_ANSWERS = {
    "q06": (
        'The retrieved material confirms that electronic records can be '
        'proved as evidence but does not itself contain the rule for how: '
        'Section 62 of the Bharatiya Sakshya Adhiniyam, 2023 states that '
        '"The contents of electronic records may be proved in accordance '
        'with the provisions of section 63" [1], and Section 65A of the '
        'Indian Evidence Act, 1872 states, in materially the same terms, '
        'that they "may be proved in accordance with the provisions of '
        'section 65B" [2] -- both point to a further section for the actual '
        'admissibility rule, and neither that section\'s text nor the '
        'certificate requirement it carries is among the retrieved material.'
    ),
    "q01": (
        'Yes. Section 420 of the Indian Penal Code, 1860 punishes cheating '
        'that dishonestly induces a person to deliver property, with '
        'imprisonment of up to seven years and a fine [1]. The current-law '
        'equivalent is Section 318 of the Bharatiya Nyaya Sanhita, 2023, '
        'which defines cheating in the same terms and separately punishes '
        'cheating that dishonestly induces delivery of property with '
        'imprisonment of up to seven years and a fine [4].'
    ),
    "q02": (
        'Murder is punishable with death or imprisonment for life, and a '
        'fine, under Section 103 of the Bharatiya Nyaya Sanhita, 2023 [1], '
        'which corresponds to Section 302 of the Indian Penal Code, 1860 [2].'
    ),
    "q03": (
        'A cheque returned unpaid for insufficiency of funds is an offence '
        'under Section 138 of the Negotiable Instruments Act, 1881, '
        'punishable "with imprisonment for a term which may be extended to '
        'two years, or with fine which may extend to twice the amount of '
        'the cheque, or with both" [2]. The offence applies only if the '
        'cheque was presented within six months of being drawn or within '
        'its validity period, whichever is earlier; the payee gave written '
        'notice demanding payment within thirty days of learning of the '
        'dishonour; and the drawer failed to pay within fifteen days of '
        'that notice [2].'
    ),
    "q04": (
        'Yes, in specified circumstances. Section 41 of the Code of '
        'Criminal Procedure, 1973 allows a police officer to arrest without '
        'a warrant or a Magistrate\'s order in a list of situations, '
        'including where the person commits a cognizable offence in the '
        'officer\'s presence, or where a reasonable complaint, credible '
        'information, or reasonable suspicion exists that the person has '
        'committed a cognizable offence and specified conditions are '
        'satisfied [5]. The retrieved material does not include the '
        'current-law provision restating this power under the Bharatiya '
        'Nagarik Suraksha Sanhita, 2023.'
    ),
    "q05": (
        'Article 21 of the Constitution of India provides: "No person shall '
        'be deprived of his life or personal liberty except according to '
        'procedure established by law." [1]'
    ),
    "q07": (
        'Yes, this can amount to theft. Section 378 of the Indian Penal '
        'Code, 1860 provides that "Whoever, intending to take dishonestly '
        'any moveable property out of the possession of any person without '
        'that person’s consent, moves that property in order to such '
        'taking, is said to commit theft." [2] The retrieved material does '
        'not include the current-law consolidated theft provision or the '
        'punishment prescribed for it.'
    ),
    "q08": (
        'Under Section 73 of the Indian Contract Act, 1872, a party injured '
        'by a breach "is entitled to receive, from the party who has broken '
        'the contract, compensation for any loss or damage caused to him '
        'thereby, which naturally arose in the usual course of things from '
        'such breach, or which the parties knew, when they made the '
        'contract, to be likely to result from the breach of it," but not '
        '"for any remote and indirect loss or damage" [2]. If the contract '
        'itself names a sum payable on breach or otherwise stipulates a '
        'penalty, Section 74 instead entitles the aggrieved party to '
        '"reasonable compensation not exceeding the amount so named" [1].'
    ),
    "q09": (
        'Section 66A of the Information Technology Act, 2000 punished '
        'sending, by computer resource or communication device, information '
        'that is grossly offensive, known to be false and sent to cause '
        'annoyance or similar harm, or email intended to deceive or mislead '
        'about its origin, with imprisonment of up to three years and a '
        'fine [1]. However, the retrieved material also records that this '
        'section was "Held unconstitutional by the Supreme Court in Shreya '
        'Singhal v Union of India, AIR 2015 SC 1523" [1], so it does not '
        'currently support a valid prosecution.'
    ),
    "q10": (
        'Article 14 of the Constitution provides: "The State shall not deny '
        'to any person equality before the law or the equal protection of '
        'the laws within the territory of India." [1]'
    ),
    "q11": (
        'Article 22 of the Constitution provides that a person who is '
        'arrested must be informed of the grounds for the arrest and must '
        'not be denied the right to consult and be defended by a legal '
        'practitioner of their choice, and must be produced before the '
        'nearest magistrate within twenty-four hours of the arrest, '
        'excluding travel time, and not be detained beyond that period '
        'without a magistrate\'s authority [1]. Separately, both Section 57 '
        'of the Code of Criminal Procedure, 1973 [4] and Section 58 of the '
        'Bharatiya Nagarik Suraksha Sanhita, 2023 [5] provide that a person '
        'arrested without a warrant may not be detained in custody beyond '
        'twenty-four hours without a special order of a Magistrate.'
    ),
    "q12": (
        'The retrieved material indicates that defamation is a criminal '
        'offence -- it refers to prosecution "for an offence punishable '
        'under Chapter XXI of the Indian Penal Code" [1] and, under current '
        'law, "an offence punishable under section 356 of the Bharatiya '
        'Nyaya Sanhita, 2023" [2] -- and that prosecution ordinarily '
        'requires a complaint by the aggrieved person [1][2]. The retrieved '
        'material does not include the provision defining defamation or '
        'stating its punishment, so those cannot be stated from it.'
    ),
    "q13": (
        'The retrieved material does not address how a First Information '
        'Report is registered; the retrieved provisions instead concern the '
        'medical examination of rape victims and of arrested or accused '
        'persons [1][2][3][5] and the Information Technology Act\'s short '
        'title [4], none of which supports an answer to this question.'
    ),
    "q14": (
        'Dowry death is defined in Section 304B of the Indian Penal Code, '
        '1860: where a woman\'s death is caused by burns, bodily injury, or '
        'occurs otherwise than under normal circumstances within seven '
        'years of her marriage, and it is shown that she was subjected '
        'shortly before her death to cruelty or harassment by her husband '
        'or his relatives for or in connection with a demand for dowry, '
        '"such death shall be called ‘dowry death’, and such husband or '
        'relative shall be deemed to have caused her death," punishable '
        'with imprisonment of not less than seven years, extendable to '
        'imprisonment for life [1]. Section 80 of the Bharatiya Nyaya '
        'Sanhita, 2023 defines and punishes dowry death in materially the '
        'same terms [2], and Section 118 of the Bharatiya Sakshya '
        'Adhiniyam, 2023 provides that where these circumstances are shown, '
        '"the Court shall presume that such person had caused the dowry '
        'death" [3].'
    ),
    "q15": (
        'Yes, under specified conditions. Section 69 of the Information '
        'Technology Act, 2000 empowers the Central or State Government, '
        'where satisfied it is necessary in the interest of matters such as '
        'the sovereignty or security of the State or public order, to '
        '"direct any agency of the appropriate Government to intercept, '
        'monitor or decrypt or cause to be intercepted or monitored or '
        'decrypted any information generated, transmitted, received or '
        'stored in any computer resource," for reasons recorded in writing, '
        'subject to procedures and safeguards separately prescribed [1].'
    ),
    "q16": (
        'Section 64 of the Bharatiya Nyaya Sanhita, 2023 provides that, '
        'except in the cases addressed by its sub-section (2), rape "shall '
        'be punished with rigorous imprisonment of either description for a '
        'term which shall not be less than ten years, but which may extend '
        'to imprisonment for life, and shall also be liable to fine" [1]. '
        'The retrieved text of sub-section (2), which addresses rape by a '
        'police officer and other aggravating circumstances, is cut off '
        'after its opening words and cannot be stated from what was '
        'retrieved. Separately, enhanced minimum sentences of twenty years '
        'to life apply where the offence is committed against a woman '
        'under sixteen [3] or under twelve years of age [3], including when '
        'committed by a group [4][5].'
    ),
}

# The chunks these answers were composed against are NOT stored here: they
# are re-derived fresh by eval/run_generation_eval.py from the real pipeline
# (retrieval is deterministic, so this reproduces the same chunks exactly)
# rather than duplicated as a second, driftable copy of the corpus.
