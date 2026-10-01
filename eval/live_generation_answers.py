"""
Frozen snapshot of a real, live run of the deployed pipeline --
confidence gate -> BM25+dense+RRF+reranked retrieval -> Groq generation --
against every query in eval/gold_qa.py (16 answerable) and
eval/unanswerable_qa.py (16 unanswerable, dev+test combined).

WHAT MAKES THIS DIFFERENT FROM eval/generation_answers.py. That earlier
round's answers were composed by Claude Opus 5 -- the same model that built
this corpus, this verifier, and the gold answer key -- invoked directly,
not through an API call, and explicitly disclosed as non-blinded. This
round is a real API call to an INDEPENDENTLY invoked model (Groq,
openai/gpt-oss-20b) with no visibility into this project's internals: it
receives only the query and whatever the real retrieval pipeline actually
ranked, under the same prompt contract (generation/generator.py's
SYSTEM_PROMPT) as the Streamlit demo's Live Generation tab.

WHAT THIS ROUND STILL DOES NOT ESTABLISH. The QUERY SET is still project-
authored (eval/gold_qa.py's own docstring: authored before retrieval was
tuned, to avoid bias toward what the system happens to retrieve -- which
bounds the bias but does not remove project authorship). This is not a
large, adversarially-sourced, externally-authored query set, and sixteen
queries is not a statistically powerful sample. Treat this as a genuine
methodological upgrade over the earlier round -- independent generation,
not independent query authorship -- not as the paper's full promised
ablation.

CAPTURED. 2026-08-31, model openai/gpt-oss-20b via Groq, real-corpus
confidence gate (tau=1.088). Only the raw completion text and the exact
retrieved chunk keys are frozen here -- the gate score and the verification
itself are recomputed fresh by eval/run_live_generation_eval.py, exactly as
eval/run_generation_eval.py already re-derives retrieval fresh for the
earlier round, per CLAUDE.md invariant 6.
"""

# Each entry: qid -> {"query", "outcome", "chunk_keys", "raw_text"}.
# outcome is one of:
#   "generated"       -- gate passed, Groq produced an answer (frozen below)
#   "gate_abstained"  -- the real-corpus gate refused this KNOWN-answerable
#                        query; no generation call was made, matching what a
#                        real user of the deployed system would experience
LIVE_ANSWERABLE = {
    'q01': {
        "query": 'Is cheating someone into handing over property a crime in India?',
        "outcome": 'gate_abstained',
        "gate_score": 0.8436136282350096,
    },
    'q02': {
        "query": 'What is the punishment for murder?',
        "outcome": 'generated',
        "gate_score": 2.2690055544376606,
        "chunk_keys": ['BNS§103', 'IPC§302', 'BNS§104', 'IPC§396', 'BNS§109'],
        "raw_text": 'The punishment for murder is death or imprisonment for life, and the offender may also be liable to fine. [1][2]',
    },
    'q03': {
        "query": 'What happens if a cheque bounces due to insufficient funds?',
        "outcome": 'gate_abstained',
        "gate_score": 0.6610077505903031,
    },
    'q04': {
        "query": 'Can the police arrest me without a warrant?',
        "outcome": 'generated',
        "gate_score": 1.7522540979089571,
        "chunk_keys": ['CRPC§55', 'BNSS§55', 'CRPC§56', 'CRPC§57', 'CRPC§41'],
        "raw_text": 'Yes.  Under the Criminal Procedure Code (CRPC) a police officer may arrest a person without a warrant if the arrest falls under the circumstances set out in **Section\u202f41**.  These circumstances include, for example, arresting a person who commits a cognisable offence in the officer’s presence, a person on whom a reasonable complaint, credible information or reasonable suspicion exists that he has committed a cognisable offence punishable by imprisonment of up to seven years, a person who has been proclaimed as an offender, or a person who is suspected of having committed a serious offence punishable by more than seven years, among other specific conditions\u202f[3].  If none of these conditions are met, the officer must obtain a warrant before arresting the individual.',
    },
    'q05': {
        "query": 'What is my right to life and personal liberty?',
        "outcome": 'generated',
        "gate_score": 1.5069933286800976,
        "chunk_keys": ['CONST§21', 'CONST§26', 'IPC§336', 'IPC§97', 'BNS§35'],
        "raw_text": 'You have the right to life and personal liberty as guaranteed by Constitution Section\u202f21: “No person shall be deprived of his life or personal liberty except according to procedure established by law.” [1]',
    },
    'q06': {
        "query": 'Are electronic records admissible as evidence in court?',
        "outcome": 'generated',
        "gate_score": 1.1751402107446713,
        "chunk_keys": ['BSA§62', 'IEA§65A', 'IEA§45A', 'BSA§81', 'BSA§33'],
        "raw_text": 'Yes.  \n- BSA\u202f§62 states that “the contents of electronic records may be proved in accordance with the provisions of section\u202f63”\xa0[1].  \n- IEA\u202f§65A likewise allows the contents of electronic records to be proved “in accordance with the provisions of section\u202f65B”\xa0[2].  \n- BSA\u202f§81 presumes the genuineness of an electronic or digital record that is kept substantially in the form required by law and produced from proper custody\xa0[4].  \n- For matters involving information transmitted or stored electronically, the court may consider the opinion of an Examiner of Electronic Evidence as a relevant fact\xa0[3].  \n\nThese provisions collectively establish that electronic records are admissible as evidence in court.',
    },
    'q07': {
        "query": "Is it illegal to take someone's belongings without permission?",
        "outcome": 'gate_abstained',
        "gate_score": 0.6178840672612743,
    },
    'q08': {
        "query": 'What compensation can I claim if a contract is breached?',
        "outcome": 'gate_abstained',
        "gate_score": 1.0285034077348976,
    },
    'q09': {
        "query": 'Can I be punished for sending offensive messages online?',
        "outcome": 'gate_abstained',
        "gate_score": 0.8483726539265692,
    },
    'q10': {
        "query": 'What does the Constitution say about equality before the law?',
        "outcome": 'generated',
        "gate_score": 1.3273612614316495,
        "chunk_keys": ['CONST§14', 'CONST§16', 'CONST§305', 'CONST§372A', 'CONST§225'],
        "raw_text": 'The Constitution guarantees equality before the law in Article\u202f14: “The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.”[1]',
    },
    'q11': {
        "query": 'What are the safeguards when a person is arrested and detained?',
        "outcome": 'generated',
        "gate_score": 1.5500428418147452,
        "chunk_keys": ['CONST§22', 'BNSS§376', 'CRPC§51', 'CRPC§57', 'BNSS§58'],
        "raw_text": "Safeguards that must be observed when a person is arrested and detained are set out mainly in the Constitution and the Criminal Procedure Code (and related statutes).  The key safeguards are:\n\n1. **Right to be informed of grounds of arrest**  \n   – “No person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest”【1】.  \n\n2. **Right to legal representation**  \n   – “…nor shall he be denied the right to consult, and to be defended by, a legal practitioner of his choice”【1】.  \n\n3. **Prompt judicial review**  \n   – “Every person who is arrested and detained in custody shall be produced before the nearest magistrate within a period of twenty‑four hours of such arrest…and no such person shall be detained…beyond the said period without the authority of a magistrate”【1】.  \n\n4. **Restrictions on detention beyond 24\u202fhrs**  \n   – “No police officer shall detain in custody a person arrested without warrant for a longer period than under all the circumstances of the case is reasonable, and such period shall not…exceed twenty‑four hours…exclusive of the time necessary for the journey from the place of arrest to the Magistrate's Court” (CRPC §57)【4】.  \n   – The same rule is repeated in BNSS §58 with reference to section\xa0187【5】.  \n\n5. **Special provisions for enemy aliens and preventive detention**  \n   – The safeguards in clauses (1) and (2) of Const §22 do not apply to “any person who…is an enemy alien” or to “any person who is arrested or detained under any law providing for preventive detention”【1】.  \n   – If detention is under a preventive‑detention law, the authority must “communicate…the grounds on which the order has been made” and “afford him the earliest opportunity of making a representation against the order”【1】.  \n   – Preventive detention may last longer than three months only if an Advisory Board reports “sufficient cause for such detention” before the expiration of the three‑month period, and even then the maximum period is limited by Parliament’s law【1】.  \n\n6. **Special safeguards for detainees in prison or mental‑health establishments**  \n   – If a person is detained in a jail, the Inspector‑General of Prisons (or, in the case of a mental‑health establishment, the Mental Health Review Board) must certify that the person is “capable of making his defence”; the person is then taken before a magistrate or court for further proceedings under section\xa0371【2】.  \n\n7. **Search and custody of personal effects**  \n   – The officer who arrests the person may search the person and place in safe custody all articles (except necessary wearing‑apparel) found on him; a receipt is given to the arrested person【3】.  \n   – If a female is to be searched, the search must be carried out by another female with strict regard to decency【3】.  \n\nThese provisions together provide the procedural safeguards that protect a person’s rights when he or she is arrested and detained.",
    },
    'q12': {
        "query": 'Is defamation a criminal offence in India?',
        "outcome": 'generated',
        "gate_score": 1.5155230069922292,
        "chunk_keys": ['CRPC§199', 'BNSS§222', 'IPC§108A', 'BNSS§208', 'CRPC§320'],
        "raw_text": 'Yes – defamation is a criminal offence in India.  It is punishable under the Indian Penal Code (Section\u202f500) and is listed among the offences that may be compounded under the Code of Criminal Procedure [5].',
    },
    'q13': {
        "query": 'How is a First Information Report registered?',
        "outcome": 'generated',
        "gate_score": 1.10833643730881,
        "chunk_keys": ['CRPC§164A', 'BNSS§184', 'CRPC§54', 'ITA§1', 'CRPC§53A'],
        "raw_text": 'The supplied source chunks do not contain information about how a First Information Report is registered.',
    },
    'q14': {
        "query": 'What is dowry death under Indian law?',
        "outcome": 'generated',
        "gate_score": 1.9529938843890935,
        "chunk_keys": ['IPC§304B', 'BNS§80', 'BSA§118', 'IEA§113', 'CRPC§354'],
        "raw_text": '**Dowry death under Indian law**\n\nA “dowry death” is defined as the death of a woman who:\n\n* has been married for seven years or less;  \n* dies because of burns or bodily injury, or in any other way that does not reflect normal circumstances;  \n* and for whom it is shown that shortly before her death she was subjected to cruelty or harassment by her husband or a relative of her husband **for, or in connection with, a demand for dowry**.  \n  The death is therefore called a “dowry death,” and the husband or relative is deemed to have caused it.[1]  \n\nThe same definition is found in the Bharatiya Nyaya Sanhita (BNS) section\u202f80.[2]  \n\nUnder section\u202f118 of the Bharatiya Samvidhan Adhiniyam (BSA), if it is shown that a woman had been subjected to such cruelty or harassment for a dowry before her death, the court **presumes** that the person who caused the cruelty has caused the dowry death.[3]  \n\nThe Indian Evidence Act, 1947, section\u202f113‑B also provides a presumption that the person responsible for the cruelty or harassment for a dowry has caused the dowry death.[4]  \n\n**Punishment**  \nA person who commits a dowry death shall be punished with imprisonment for a term not less than',
    },
    'q15': {
        "query": 'Can the government intercept or monitor my communications?',
        "outcome": 'gate_abstained',
        "gate_score": 1.0329936020520425,
    },
    'q16': {
        "query": 'What is the punishment for rape?',
        "outcome": 'generated',
        "gate_score": 1.9672691882098385,
        "chunk_keys": ['BNS§64', 'IPC§376AB', 'BNS§65', 'IPC§376DB', 'IPC§376DA'],
        "raw_text": 'The punishment for rape depends on the age of the victim and whether the offence was committed by a single person or a group.\n\n- **General rape (no age restriction on the victim)**  \n  “Whoever, except in the cases provided for in sub‑section (2), commits rape, shall be punished with rigorous imprisonment of either description for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine.”\u202f[1]\n\n- **Rape of a woman under twelve years of age**  \n  “Whoever, commits rape on a woman under twelve years of age shall be punished with rigorous imprisonment for a term which shall not be less than twenty years, but which may extend to imprisonment for life, … and with fine or with death….”\u202f[2]\n\n- **Rape of a woman under sixteen years of age**  \n  “Whoever, commits rape on a woman under sixteen years of age shall be punished with rigorous imprisonment for a term which shall not be less than twenty years, but which may extend to imprisonment for life, and shall also be liable to fine.”\u202f[3]\n\n- **Group or common‑intention offences involving a woman under twelve**  \n  “Each of those persons shall be deemed to have committed the offence of rape and shall be punished with imprisonment for life … and with fine, or with death.”\u202f[4]\n\n- **Group or common‑intention offences involving a woman under sixteen**  \n  “Each of those persons shall be deemed to have committed the offence of rape and shall be punished with imprisonment for life … and with fine.”\u202f[5]',
    },
}

# Every one of the 16 unanswerable queries (8 dev + 8 test) was correctly
# abstained by the real-corpus gate -- zero generation calls were made, so
# there is no raw text to freeze here. This dict exists so the analysis
# script has one place to confirm that against, the same way it confirms
# the answerable side.
LIVE_UNANSWERABLE_ALL_ABSTAINED = 16  # count, both splits
