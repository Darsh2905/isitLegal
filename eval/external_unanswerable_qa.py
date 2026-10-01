"""
Genuinely externally-sourced unanswerable queries -- the half of Limitation
Seven's "author-written, not adversarially sourced" gap that is actually
fixable without a large new corpus. Every question below is the VERBATIM
title of a real, published Q&A page on law4u.in, a public Indian legal
question-and-answer platform -- not composed by this project, unlike
eval/unanswerable_qa.py's UNANSWERABLE_DEV/UNANSWERABLE_TEST. Sourced via
web search on 2026-08-31; the URL is kept alongside each question specifically
so the claim "this was not authored by us" is checkable, not just asserted.

WHAT THIS DOES AND DOES NOT ESTABLISH. These are real page titles from a
real platform, which is a genuine step away from author-written queries. It
is NOT a claim that law4u.in's own users generated these questions
independently of any legal-content strategy (a Q&A content site plausibly
seeds and edits its own questions for SEO), and it is a small set (eight),
not a statistically powered sample -- the same honest bound this project
places on every other adversarial probe set it builds. Treat this as a
genuine out-of-sample check on the ALREADY-DEPLOYED, already-calibrated
gate (retrieval/confidence.py's real-corpus tau), not a retuning exercise:
no threshold here is fit to these queries.
"""

EXTERNAL_UNANSWERABLE = [
    ("How can one register a trademark in India?",
     "https://law4u.in/answer/280/How-can-one-register-a-trademark-in-India"),
    ("What are the legal requirements for appointing and removing directors "
     "in India?",
     "https://law4u.in/answer/121/What-are-the-legal-requirements-for-appointing-and-removing-directors-in-India"),
    ("How Many Weeks of Maternity Leave Is a Woman Entitled to in India?",
     "https://www.law4u.in/top-answer/9410/how-many-weeks-of-maternity-leave-is-a-woman-entitled-to-in-india"),
    ("What is the limitation period for filing a recovery suit in India?",
     "https://law4u.in/answer/9605/What-is-the-limitation-period-for-filing-a-recovery-suit-in-India"),
    ("What are the grounds for divorce under the Hindu Marriage Act?",
     "https://law4u.in/answer/6670/What-are-the-grounds-for-divorce-under-the-Hindu-Marriage-Act"),
    ("Can anticipatory bail be granted in cases of economic offenses?",
     "https://law4u.in/answer/2070/Can-anticipatory-bail-be-granted-in-cases-of-economic-offenses"),
    ("How Do the Provisions of Section 56(2) of the Income Tax Act Apply to "
     "Gifts?",
     "https://law4u.in/top-answer/10330/how-do-the-provisions-of-section-562-of-the-income-tax-act-apply-to-gifts"),
    ("Can a Marriage Be Dissolved Without Mutual Consent In India?",
     "https://law4u.in/top-answer/3269/can-a-marriage-be-dissolved-without-mutual-consent-in-india"),
]
