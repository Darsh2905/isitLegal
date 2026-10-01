"""
Gold QA set: natural-language legal questions with expected provisions.

Authored BEFORE tuning retrieval, per the methodology, to avoid biasing the
questions toward what the system happens to retrieve. Scoped to provisions
present in the seed corpus. `in_force_answer` records what the CURRENT law is,
so recodification handling can be scored separately from retrieval.
"""
# (qid, question, relevant_keys, in_force_answer)
GOLD = [
 ("q01","Is cheating someone into handing over property a crime in India?",
  ["BNS§318","IPC§420"],"BNS§318"),
 ("q02","What is the punishment for murder?",["BNS§103","IPC§302"],"BNS§103"),
 ("q03","What happens if a cheque bounces due to insufficient funds?",
  ["NIA§138"],"NIA§138"),
 ("q04","Can the police arrest me without a warrant?",
  ["BNSS§35","CRPC§41"],"BNSS§35"),
 ("q05","What is my right to life and personal liberty?",["CONST§21"],"CONST§21"),
 ("q06","Are electronic records admissible as evidence in court?",
  ["BSA§63","IEA§65B"],"BSA§63"),
 ("q07","Is it illegal to take someone's belongings without permission?",
  ["BNS§303","IPC§378","IPC§379"],"BNS§303"),
 ("q08","What compensation can I claim if a contract is breached?",["ICA§73"],"ICA§73"),
 ("q09","Can I be punished for sending offensive messages online?",["ITA§66A"],None),
 ("q10","What does the Constitution say about equality before the law?",
  ["CONST§14"],"CONST§14"),
 ("q11","What are the safeguards when a person is arrested and detained?",
  ["CONST§22"],"CONST§22"),
 ("q12","Is defamation a criminal offence in India?",
  ["BNS§356","IPC§499"],"BNS§356"),
 ("q13","How is a First Information Report registered?",
  ["BNSS§173","CRPC§154"],"BNSS§173"),
 ("q14","What is dowry death under Indian law?",["BNS§80","IPC§304B"],"BNS§80"),
 ("q15","Can the government intercept or monitor my communications?",["ITA§69"],"ITA§69"),
 ("q16","What is the punishment for rape?",["BNS§64","IPC§376"],"BNS§64"),
]
