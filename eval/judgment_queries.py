"""
Gold query set for the judgment-retrieval chunking comparison
(eval/run_judgment_retrieval_eval.py). Hand-triggered but script-generated,
same convention as this project's other probe sets being explicit about
provenance rather than claiming an external benchmark: no official
relevance judgments exist for this project's own 110-judgment Indian
Kanoon sample (see ingest/scrape_judgments.py), so a real, deterministic
alternative was used instead of authoring queries by hand.

METHOD. Fixed random seed (7) over the judgments with >=40 paragraphs (87
of 110 -- paragraph counts and text reflect the 2026-09-23 fixes to
scrape_judgments.py that recovered quoted-material blockquotes, stripped
inline e-filing/pagination artifacts, and rescued one free-floating
<span>-wrapped paragraph; see its module docstring) -- the subset where
naive fixed-window chunking and per-paragraph
parent-document chunking can actually differ, since a short judgment's
whole body fits in one or two chunks either way. For each of 20 sampled
judgments, one paragraph strictly inside the body (not the first or last
10 paragraphs, avoiding boilerplate headers/signature blocks) was chosen at
random, and its 6 longest non-stopword terms became the query. This is a
"vocabulary overlap" query, not a natural-language question -- it tests
whether the retrieval layer can find the paragraph a set of its own
distinctive terms came from, which is exactly the signal naive chunking's
dilution (concatenating that paragraph with up to 3 lexically unrelated
neighbours before indexing) or parent-document chunking's isolation (one
paragraph, one chunk) should move.

Each entry: (query, doc_id, para_index, case_name).
"""

GOLD = [
    ('distribution appreciable competition enterprises competitive association', '66135551', 223, 'Rajasthan Cylinders And Containers ... vs U.O.I And Anr'),
    ('consequence obstruction proceedings respondent directions clarified', '18303067', 20, 'Shafin Jahan vs Asokan K.M.'),
    ('adjudication petitioner objection concerned binding against', '103201262', 38, 'The Silppi Constructions Contractors vs Union Of India'),
    ('applications reiterating impleadment peremptory eventually reiterated', '161155740', 29, 'Wg. Cdr. Arifur Rahman Khan And Aleya ... vs Dlf Southern Homes Pvt. Ltd. (Now Known ...'),
    ('confessional admissible statements extracted presiding officers', '181925663', 81, 'In Re: To Issue Certain Guidelines ... vs The State Of Andhra Pradesh'),
    ('nomination improperly especially candidates materially contested', '172105947', 266, 'Arjun Panditrao Khotkar vs Kailash Kushanrao Gorantyal'),
    ('extinguished proceedings difficulty government resolution including', '30560910', 128, 'Ghanashyam Mishra And Sons Private ... vs Edelweiss Asset Reconstruction ...'),
    ('maintainable chargesheet fundamental proceedings association restriction', '31538746', 40, 'Himachal Pradesh Cricket Association vs The State Of Himachal Pradesh'),
    ('advertisements advertisement regulation programmes television prohibits', '103640961', 660, 'Kaushal Kishor vs The State Of Uttar Pradesh Govt. Of U.P. ...'),
    ('reconsideration representation superannuation proportionate consideration consultation', '139489769', 66, 'Prem Singh vs The State Of Uttar Pradesh Secretary'),
    ('fundamental bharathamma appellants conditions profitably infraction', '26303342', 41, 'Amrit Paul Singh vs Tata Aig General Insurance Co. Ltd.'),
    ('constitutional transgressed constitution legislatures legislature limitations', '90225896', 111, 'Union Of India vs The State Of Maharashtra'),
    ('significance contributory respondents resultantly negligently negligence', '153452910', 55, 'Mangla Ram vs The Oriental Insurance Company Ltd'),
    ('preponderance successfully basalingappa passionately jurisdiction undertaking', '19349803', 12, 'M/S Kalamani Tex vs P. Balasubramanian'),
    ('jurisdiction proceedings provisional attachment violation initiated', '62362537', 34, 'M/S Radha Krishan Industries vs The State Of Himachal Pradesh'),
    ('eligibility appointment application termination hereinabove declaration', '88491207', 63, 'Bharat Broadband Network Limited vs United Telecoms Limited'),
    ('honourable punishment committed religious barbarism sometimes', '92846055', 49, 'Shakti Vahini vs Union Of India'),
    ('reasonableness entertaining requirement discussion oppressive arbitrary', '80182706', 153, 'M/S Hindon Forge Pvt. Ltd. vs The State Of Uttar Pradesh Thr. District ...'),
    ('investigation contemplated allegations constitute magistrate cognizable', '14567786', 48, 'Anand Kumar Mohatta vs State (Govt. Of Nct Of Delhi) Department ...'),
    ('nominations submitting commission directions elections direction', '115643279', 14, 'West Bengal State Election Commission vs Communist Party Of India (Marxist) Its ...'),
]
