"""Generates the CivilLens datasets.

Outputs (in this folder):
  train_toxicity.csv       labelled comments for training the local toxicity classifier
  rewrite_pairs.csv        toxic comment + candidate rewrite + expected verdict (for evaluation)
  input_comments_100.csv   100 comments for the Batch Review module (demo input)

Run:  python data/generate_dataset.py
All comments are synthetic (civic complaints on public portals). Toxic language is kept
to mild insults and dismissive phrases; there are no slurs.
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).parent

SUBJECTS = {
    # subject: (have/has, are/is, issues)
    "the officials": ("have", "are", ["the potholes on our road", "the broken street lights", "garbage collection", "the blocked drainage"]),
    "the municipal corporation": ("has", "is", ["garbage collection", "the potholes on the main road", "the overflowing drains", "stray cattle on the highway"]),
    "the ward councillor": ("has", "is", ["the broken footpath", "the street lights in our lane", "the park near our school"]),
    "the electricity board": ("has", "is", ["the daily power cuts", "the faulty transformer", "the wrong electricity bills"]),
    "the water department": ("has", "is", ["the water supply", "the leaking pipeline", "the dirty tap water"]),
    "the transport authority": ("has", "is", ["the bus timings", "the overcrowded buses", "the broken bus shelter"]),
    "the hospital staff": ("have", "are", ["the long waiting time", "the shortage of beds", "the broken lift in the ward"]),
    "the college administration": ("has", "is", ["the delayed exam results", "the hostel water problem", "the fee refund"]),
    "the customer support team": ("has", "is", ["my refund", "the damaged delivery", "my complaint ticket"]),
    "the contractor": ("has", "is", ["the half-built flyover", "the dug-up road", "the metro construction mess"]),
    "the bank staff": ("have", "are", ["my loan application", "the ATM that never works", "the long queue at the counter"]),
    "the traffic police": ("have", "are", ["the signal at the junction", "the illegal parking near the school", "the traffic jam every morning"]),
    "the railway department": ("has", "is", ["the train delays", "the dirty platform", "the broken escalator"]),
    "the panchayat members": ("have", "are", ["the village road", "the drinking water tank", "the street lights"]),
}
TIMES = ["2 years", "six months", "three weeks", "over a month", "a whole year", "10 days", "four months", "eight weeks"]

COMPLAINTS = [
    "{S} {have} not fixed {I} for {T}.",
    "Nobody at {S} has looked at {I} for {T}.",
    "Why {have} {S} still not resolved {I} after {T}?",
    "{S} keep promising to fix {I}, but nothing has happened in {T}.",
    "We have complained about {I} for {T} and {S} {have} done nothing.",
    "It has been {T} and nothing has been done about {I}. {Scap} must act now.",
    "{S} {be} not responding to our complaints about {I}.",
    "For {T} we have been waiting, and {S} {have} still not repaired {I}.",
]

INSULT_PRED_PL = ["useless idiots", "a bunch of clowns", "pathetic", "brainless", "incompetent fools",
                  "total morons", "lazy losers", "clueless jokers", "worthless"]
INSULT_PRED_SG = ["useless", "a bunch of clowns", "pathetic", "brainless", "a total joke", "clueless",
                  "worthless", "incompetent", "shameless"]
INSULT_ADJ = ["Useless", "Pathetic", "Brainless", "Clueless", "Lazy", "Incompetent", "Shameless"]

TOXIC_TEMPLATES = {
    "insult": [
        "{Scap} {be} {INS}. {C}",
        "{C} What a bunch of clowns.",
        "{ADJ} people! {C}",
        "{C} {Scap} {be} {INS} and should be ashamed.",
        "{C} Only idiots would run things like this.",
        "{C} Honestly, {S} {be} {INS}.",
    ],
    "profanity": [
        "{C} This is damn ridiculous.",
        "What the hell? {C}",
        "{C} The whole service is crap.",
        "{C} This freaking system sucks.",
        "Bloody hell, {c}",
    ],
    "harassment": [
        "{C} {Scap}, just shut up and do your job.",
        "Get lost, {S}. {C}",
        "{C} Nobody cares about you, just resign and disappear.",
        "{C} Go away and never come back.",
        "{C} No one wants you here.",
    ],
    "threat": [
        "{C} You will regret this.",
        "{C} We will make sure you lose your job.",
        "{C} Wait and see what happens to you.",
        "{C} You will pay for this.",
    ],
    "sarcasm": [
        "Wow, great job as always, {S}. {C}",
        "{C} Thanks for nothing.",
        "Brilliant work by {S}, truly world class. {C}",
        "{C} What a genius plan by {S}.",
        "Oh great, {c} Slow clap.",
    ],
}

CLEAN_EXTRA = [
    "Thank you {S} for fixing {I} so quickly.",
    "{Scap} handled {I} well this time.",
    "Good work by {S} on {I}; it is much better now.",
    "When will {S} fix {I}?",
    "Can {S} share an update on {I}?",
    "Please ask {S} to look into {I}.",
]
IDENTITY = ["Muslim woman", "Hindu family", "Christian student", "Sikh shopkeeper", "gay man", "transgender person",
            "disabled person", "senior citizen", "deaf student", "migrant worker", "elderly woman", "Jain trader"]
IDENTITY_TEMPLATES = [
    "As {a} {ID}, I request {S} to fix {I} soon.",
    "As {a} {ID}, I feel unsafe because of {I}.",
    "I am {a} {ID} and {I} affects me every day. Please help.",
    "As {a} {ID}, I am thankful that {S} improved {I}.",
]

GENERIC_OVERCLEAN = ["I have some thoughts about this.", "Please look into this matter.", "Thank you.",
                     "This is my feedback.", "Okay, noted.", "Something should be checked."]
STANCE_FLIP = ["{Scap} {have} handled {I} very well.", "Everything about {I} is perfect now, thanks to {S}.",
               "Great work by {S}; {I} is no longer a problem.", "{Scap} {be} doing an excellent job with {I}."]
INVENTED = [" {Scap} stole the money meant for it.", " Everyone knows {S} took bribes for this.",
            " {Scap} {be} corrupt and lied to the public."]
STILL_TOXIC = [" {Scap} {be} lazy.", " This is damn ridiculous.", " {Scap} {be} clueless.", " What a pathetic effort."]
SARCASTIC = ["Wow, great job as always, {S}. ", "Brilliant work, truly world class. "]


def cap(s):
    return s[0].upper() + s[1:] if s else s


def fill(tpl, S, I=None, T=None, C=None, extra=None):
    have, be, _ = SUBJECTS[S]
    d = dict(S=S, Scap=cap(S), have=have, be=be, I=I or "", T=T or "", C=C or "", c=(C[0].lower() + C[1:]) if C else "")
    d.update(extra or {})
    return cap(tpl.format(**d))


def complaint(rng):
    S = rng.choice(list(SUBJECTS))
    I = rng.choice(SUBJECTS[S][2])
    T = rng.choice(TIMES)
    C = fill(rng.choice(COMPLAINTS), S, I, T)
    return S, I, T, C


def toxic_comment(rng, category=None):
    S, I, T, C = complaint(rng)
    category = category or rng.choice(list(TOXIC_TEMPLATES))
    tpl = rng.choice(TOXIC_TEMPLATES[category])
    text = fill(tpl, S, I, T, C, {"INS": rng.choice(INSULT_PRED_PL if SUBJECTS[S][1] == "are" else INSULT_PRED_SG), "ADJ": rng.choice(INSULT_ADJ)})
    return dict(text=text, category=category, subject=S, issue=I, complaint=C)


def clean_comment(rng):
    S = rng.choice(list(SUBJECTS))
    I = rng.choice(SUBJECTS[S][2])
    T = rng.choice(TIMES)
    r = rng.random()
    if r < 0.45:
        text = fill(rng.choice(COMPLAINTS), S, I, T)
    elif r < 0.75:
        text = fill(rng.choice(CLEAN_EXTRA), S, I, T)
    else:
        ident = rng.choice(IDENTITY)
        a = "an" if ident[0].lower() in "aeiou" else "a"
        text = fill(rng.choice(IDENTITY_TEMPLATES), S, I, T, extra={"ID": ident, "a": a})
    return dict(text=text, category="none", subject=S, issue=I, complaint=text)


def candidate_rewrite(rng, row, kind):
    S, I, C = row["subject"], row["issue"], row["complaint"]
    if kind == "good":
        return C
    if kind == "still_toxic":
        return C + fill(rng.choice(STILL_TOXIC), S, I)
    if kind == "meaning_lost":
        return rng.choice(GENERIC_OVERCLEAN)
    if kind == "stance_flipped":
        return fill(rng.choice(STANCE_FLIP), S, I)
    if kind == "sarcastic":
        return fill(rng.choice(SARCASTIC), S, I) + C
    if kind == "invented_fact":
        return C + fill(rng.choice(INVENTED), S, I)
    raise ValueError(kind)


EXPECTED = {"good": "Accept", "still_toxic": "Revise", "meaning_lost": "Revise", "sarcastic": "Revise",
            "stance_flipped": "Reject", "invented_fact": "Reject"}


def toxic_spans(text):
    from_lex = []
    low = text.lower()
    import re
    import sys
    sys.path.insert(0, str(HERE.parent))
    from civillens.lexicon import TOXIC_LEXICON, SARCASM_CUES
    for cat, words in TOXIC_LEXICON.items():
        for w in sorted(words, key=len, reverse=True):
            if re.search(r"\b" + re.escape(w) + r"\b", low) and not any(w in f for f in from_lex):
                from_lex.append(w)
    for cue in SARCASM_CUES:
        if cue in low and not any(cue in f for f in from_lex):
            from_lex.append(cue)
    return "; ".join(from_lex)


def build_train(rng, n_toxic=2400, n_clean=2400):
    rows, seen = [], set()
    cats = list(TOXIC_TEMPLATES)
    while sum(r["is_toxic"] for r in rows) < n_toxic:
        r = toxic_comment(rng, cats[len(rows) % len(cats)])
        if r["text"] not in seen:
            seen.add(r["text"]); rows.append(dict(text=r["text"], is_toxic=1, category=r["category"]))
    while sum(1 - r["is_toxic"] for r in rows) < n_clean:
        r = clean_comment(rng)
        if r["text"] not in seen:
            seen.add(r["text"]); rows.append(dict(text=r["text"], is_toxic=0, category="none"))
    rng.shuffle(rows)
    return rows


def build_pairs(rng, n=600, exclude=()):
    kinds = ["good"] * 7 + ["still_toxic", "meaning_lost", "stance_flipped", "sarcastic", "invented_fact"] * 2 + ["good"]
    rows, seen = [], set(exclude)
    while len(rows) < n:
        t = toxic_comment(rng)
        if t["text"] in seen:
            continue
        seen.add(t["text"])
        kind = kinds[len(rows) % len(kinds)]
        rows.append(dict(pair_id=f"P{len(rows)+1:04d}", original_comment=t["text"],
                         candidate_rewrite=candidate_rewrite(rng, t, kind), rewrite_type=kind,
                         expected_verdict=EXPECTED[kind], category=t["category"]))
    return rows


def build_input(rng, exclude=()):
    seen = set(exclude)
    plan = (["good"] * 24 + ["still_toxic"] * 8 + ["meaning_lost"] * 7 + ["stance_flipped"] * 7 +
            ["sarcastic"] * 7 + ["invented_fact"] * 7 + ["auto"] * 25 + ["clean"] * 15)
    rng.shuffle(plan)
    rows = []
    for kind in plan:
        while True:
            t = clean_comment(rng) if kind == "clean" else toxic_comment(rng)
            if t["text"] not in seen:
                seen.add(t["text"]); break
        if kind == "clean":
            rew, exp, rtype = t["text"], "Accept", "no_change_needed"
        elif kind == "auto":
            rew, exp, rtype = "", "", "auto_rewrite"
        else:
            rew, exp, rtype = candidate_rewrite(rng, t, kind), EXPECTED[kind], kind
        rows.append(dict(comment_id=f"C{len(rows)+1:03d}", original_comment=t["text"], candidate_rewrite=rew,
                         toxic_spans=toxic_spans(t["text"]) if kind != "clean" else "",
                         category=t["category"], target=t["subject"], rewrite_type=rtype, expected_verdict=exp))
    return rows


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path.name}: {len(rows)} rows")


if __name__ == "__main__":
    train = build_train(random.Random(42))
    write(HERE / "train_toxicity.csv", train)
    pairs = build_pairs(random.Random(7), exclude={r["text"] for r in train})
    write(HERE / "rewrite_pairs.csv", pairs)
    inp = build_input(random.Random(2026), exclude={r["text"] for r in train} | {r["original_comment"] for r in pairs})
    write(HERE / "input_comments_100.csv", inp)
