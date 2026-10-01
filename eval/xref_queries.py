"""
Queries whose correct answer is a NON-SELF-CONTAINED provision.

Each gold provision's own text describes almost nothing -- it defers to another
section. A user asking about the substance therefore cannot reach it lexically
or semantically. This is the Rustamov failure mode, isolated.
"""
# (qid, query, gold_keys, anchor_desc)
XREF_GOLD = [
 ("x1","What is the punishment for attempting to commit murder?",
      ["IPC§511","BNS§62"], "attempt provisions defer to the murder section"),
 ("x2","Is there a time limit for filing a cheque bounce complaint?",
      ["NIA§142"], "cognizance provision defers to s.138"),
 ("x3","What is the penalty for accessing electronic records without authority?",
      ["ITA§72"], "confidentiality penalty defers to s.69"),
 ("x4","What procedure must police follow after making an arrest?",
      ["BNSS§36"], "procedure provision defers to s.35 and s.173"),
 ("x5","What happens if someone tries to commit cheating but fails?",
      ["IPC§511","BNS§62"], "attempt provisions defer to cheating section"),
]
