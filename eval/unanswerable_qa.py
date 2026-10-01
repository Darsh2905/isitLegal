"""
Deliberately unanswerable queries for the confidence gate (contribution C3).

Each is a real question about Indian law whose answer is NOT in the development
corpus: the corpus covers ten central enactments and parts of the Constitution,
so tax, intellectual property, family law, company law, labour law and
limitation are all outside it. A system without an abstention gate answers these
anyway, from whatever the retriever happened to rank first.

These are HARD negatives on purpose: several share vocabulary with the corpus
("property", "bail", "penalty", "notice", "records"), so a gate cannot succeed
merely by noticing an unfamiliar word.

Split into dev and test. The threshold is tuned on dev ONLY and reported on
test, because a threshold tuned and reported on the same queries measures
nothing -- the criticism this paper makes of other work applies to it too.
"""

UNANSWERABLE_DEV = [
    "What is the GST rate applicable to textile exports?",
    "How do I register a trademark in India?",
    "What are the grounds for divorce under Hindu marriage law?",
    "What is the limitation period for filing a civil suit for recovery?",
    "How is stamp duty calculated on a property sale deed?",
    "What are the rules for foreign direct investment in insurance?",
    "How do I apply for anticipatory bail in a corruption case?",
    "What maternity leave is a woman employee entitled to?",
]

UNANSWERABLE_TEST = [
    "What licence is needed to operate a drone commercially?",
    "How is agricultural income taxed?",
    "What notice must an employer give before retrenchment?",
    "How are company directors disqualified under the Companies Act?",
    "What records must a charitable trust maintain annually?",
    "What is the penalty for late filing of an income tax return?",
    "How do I obtain a patent for a software invention?",
    "What are the reservation rules for promotions in public employment?",
]

UNANSWERABLE = UNANSWERABLE_DEV + UNANSWERABLE_TEST
