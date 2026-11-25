"""Word lists used by CivilLens.

The lists are deliberately limited to mild, common insults and dismissive phrases.
They are used for (1) masking toxic words before anything is shown or sent to an LLM,
(2) span-level toxicity checks and (3) the offline (no-LLM) fallbacks.
"""

# category -> phrases (longest phrases are matched first)
TOXIC_LEXICON = {
    "insult": [
        "useless idiots", "bunch of clowns", "bunch of idiots", "bunch of fools", "incompetent fools",
        "brainless", "idiots", "idiot", "morons", "moron", "fools", "fool", "stupid", "dumb",
        "useless", "pathetic", "clowns", "clown", "losers", "loser", "incompetent", "worthless",
        "lazy", "clueless", "shameless", "jokers", "joker", "donkeys", "nincompoops",
    ],
    "profanity": [
        "damn", "crap", "crappy", "bloody", "hell", "freaking", "sucks",
    ],
    "harassment": [
        "shut up", "get lost", "nobody cares about you", "go away and never come back",
        "go away", "no one wants you here", "just resign and disappear", "get out",
    ],
    "threat": [
        "you will regret this", "we will make sure you lose your job",
        "you will pay for this", "wait and see what happens to you", "we know where your office is",
    ],
}

# Sarcasm / mockery cues (implicit toxicity a classifier often misses)
SARCASM_CUES = [
    "wow, great job", "great job as always", "brilliant work", "truly world class", "thanks for nothing",
    "slow clap", "what a genius", "geniuses", "genius", "oh wonderful", "oh great", "bravo",
    "award-winning effort", "as expected from you", "nice going", "well done, really",
]

# Words showing the writer is complaining / against something
NEGATIVE_WORDS = {
    "not", "never", "no", "nothing", "ignored", "ignore", "still", "broken", "failed", "fail", "delay",
    "delayed", "worst", "poor", "bad", "late", "unresolved", "dirty", "overflowing", "missing", "unsafe",
    "problem", "problems", "complaint", "complain", "waiting", "stuck", "slow", "careless", "useless",
    "without", "again", "cannot", "dark", "flooded", "flooding", "cut", "cuts", "shortage", "rude", "wrong", "unfair", "pending",
}

# Words showing the writer is happy / in favour
POSITIVE_WORDS = {
    "well", "good", "great", "excellent", "handled", "resolved", "fixed", "thank", "thanks", "appreciate",
    "happy", "satisfied", "wonderful", "perfect", "smooth", "efficient", "quick", "helpful", "improved",
    "fine", "nice", "best", "amazing", "love",
}

# Words that would be an invented accusation if they appear only in the rewrite
ACCUSATION_WORDS = {
    "stole", "steal", "stealing", "corrupt", "corruption", "bribe", "bribes", "fraud", "scam", "lied",
    "cheat", "cheated", "looted", "embezzled", "criminal", "illegal",
}

# Identity terms used for the identity-swap fairness test (never toxic by themselves)
IDENTITY_TERMS = [
    "muslim", "hindu", "christian", "sikh", "jain", "buddhist", "woman", "man", "gay", "lesbian",
    "transgender", "disabled", "deaf", "blind", "elderly", "senior citizen", "migrant", "student",
]
IDENTITY_SWAPS = {
    "muslim": "hindu", "hindu": "christian", "christian": "sikh", "sikh": "muslim", "jain": "buddhist",
    "buddhist": "jain", "woman": "man", "man": "woman", "gay": "straight", "lesbian": "straight",
    "transgender": "cisgender", "disabled": "non-disabled", "deaf": "hearing", "blind": "sighted",
    "elderly": "young", "senior citizen": "young person", "migrant": "local resident", "student": "teacher",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
    "on", "for", "at", "by", "with", "from", "as", "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "we", "our", "us", "you", "your", "i", "me", "my", "he", "she", "his", "her", "have",
    "has", "had", "do", "does", "did", "so", "if", "then", "than", "there", "here", "what", "which", "who",
    "why", "how", "all", "any", "can", "could", "should", "would", "will", "just", "about", "into", "over",
    "after", "before", "again", "also", "very", "too", "more", "most", "such", "only", "own", "same", "please",
    "kindly", "really", "even", "every", "some", "need", "needs", "now", "up", "out", "please", "look",
}

