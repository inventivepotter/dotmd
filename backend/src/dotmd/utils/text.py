"""Text processing utilities."""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Canonical noise-filtering used by BM25, TF-IDF, and key-term extraction.
#
# ALL filtering logic lives here so callers just use ``is_noise_token()``.
# The function handles both lowercase tokens (BM25/TF-IDF) and uppercase
# tokens (acronym extraction) via case-insensitive lookup.
# ---------------------------------------------------------------------------

# Lowercase stop words: NLTK stopwords + sklearn extras + common English +
# markup/mermaid/CSS noise.  This is the single canonical list.
_STOP_WORDS: frozenset[str] = frozenset(
    # --- NLTK English stopwords (179 words) ---
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between into through during before after above below to from up down "
    "in out on off over under again further then once here there when where "
    "why how all any both each few more most other some such no nor not only "
    "own same so than too very s t can will just don should now d ll m o re "
    "ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn "
    "needn shan shouldn wasn weren won wouldn "
    # --- sklearn / spaCy extras ---
    "also already always among amount another anyway anywhere back become "
    "becomes becoming behind beside besides beyond bill bottom call came "
    "can cannot co con could couldnt cry de describe detail done due eg "
    "eight either eleven else elsewhere empty enough etc even ever every "
    "everyone everything everywhere except fifteen fifty fill find fire "
    "first five former formerly forty found four front full further get "
    "give go gone got had has hasnt hence hereafter hereby herein hereupon "
    "however hundred inc indeed interest keep last latter latterly least "
    "less ltd made many may meanwhile might mill mine moreover move much "
    "must myself name namely neither nevertheless next nine nobody none "
    "noone nothing now nowhere often one onto others otherwise part per "
    "perhaps please put rather say see seem several show side since "
    "sincere six sixty somehow someone something sometime sometimes "
    "somewhere still such system take ten thick thin third though three "
    "through throughout thru thus together top toward towards twelve "
    "twenty two un upon us via want well whatever whenever wherever "
    "whether whither whole whom whose will within without yet "
    # --- Common English verbs / adjectives (additional) ---
    "also area back best case come deep does done down each even fact "
    "find form four free good half hand hard help here high home idea "
    "keep kind knew know late less life line link list live long look "
    "made main make move near need next none note open page pass past "
    "plan play plus pull push real rest role rule safe same save sign "
    "small space start state store think title under value world write "
    "learn level guard guide human known large given great avoid check "
    # --- Mermaid / diagram tokens ---
    "subgraph direction flowchart mindmap graph classdef linkstyle click "
    "style fill color stroke width "
    # --- CSS / HTML noise ---
    "font size height margin padding left right center bold italic "
    "div span img src alt href class".split()
)

# Uppercase tokens to reject during acronym extraction.
# These are common English words or markup tokens that appear in uppercase
# but are NOT meaningful acronyms.
_SKIP_UPPER: frozenset[str] = frozenset(
    # Short function words (uppercase forms caught by acronym regex)
    "A AM AN AS AT BE BY DO GO HE IF IN IS IT ME MY NO OF OH OK ON OR "
    "OX SO TO UP US WE "
    "ADD ALL AND ANY ARE BAD BIG BIT BUT CAN DAY DID END FAR FEW FOR "
    "GET GOT HAS HAD HER HIM HIS HOW ITS JOB KEY LET LOT MAY MET MIX "
    "NEW NOR NOT NOW ODD OFF OLD ONE OUR OUT OWN PUT RAN RUN SAT SAW "
    "SAY SET SHE SIT SIX TEN THE TOO TOP TRY TWO USE VIA WAS WAY WHO "
    "WHY WON YET YOU "
    # Programming / markup tokens
    "CSS DIV DOM EOF FIG GIT HEX IMG INT LOG MAX MIN MOD MUT NaN NIL "
    "NUL OBJ OPT PNG PRE PTR RAW REF REL RES RET ROW SRC STD STR SUB "
    "SUM SVG TAB TAG TMP URL VAL VAR XML "
    # Common markdown / document structure
    "EG IE VS OK NA NB PS RE FYI TBD TBA WIP FAQ "
    # Size / unit abbreviations
    "KB MB GB TB MS NS HZ MHZ GHZ "
    # Mermaid / diagram direction tokens
    "LR RL TB TD BT BR BL TR TL "
    # Common English words that appear uppercase in markdown/headings
    "DATA ALSO AREA BACK BEEN BEST BOTH CALL CAME CASE CODE COME "
    "DEEP DOES DONE DOWN EACH EVEN FACT FILE FILL FIND FLOW FORM FOUR "
    "FREE FROM FULL GIVE GOES GONE GOOD HALF HAND HARD HAVE HEAD HELP "
    "HERE HIGH HOME IDEA INTO JUST KEEP KIND KNEW KNOW LAST LATE LEFT "
    "LESS LIFE LINE LINK LIST LIVE LONG LOOK MADE MAIN MAKE MANY MORE "
    "MOST MOVE MUCH MUST NAME NEAR NEED NEXT NONE NOTE ONCE ONLY OPEN "
    "OVER PAGE PART PASS PAST PLAN PLAY PLUS PULL PUSH REAL REST ROLE "
    "ROLES RULE SAFE SAME SAVE SHOW SIDE SIGN SIZE SOME STEP STOP SURE "
    "TAKE TELL TEXT THAN THAT THEM THEN THIS TIME TRUE TURN TYPE UNIT "
    "UPON USED VERY VIEW WANT WELL WENT WHAT WHEN WILL WISH WITH WORD "
    "WORK YEAR YOUR ZERO AVOID BELOW BUILD CHECK COULD EVERY FIRST "
    "GIVEN GREAT GUARD GUIDE HUMAN KNOWN LARGE LEARN LEVEL MIGHT NEVER "
    "OTHER POINT RIGHT SHALL SHARE SHOULD SINCE SMALL SPACE START STATE "
    "STILL STORE STYLE THINK THOSE THREE TITLE UNDER UNTIL VALUE WHICH "
    "WHILE WHOLE WORLD WOULD WRITE "
    # Common uppercase words that aren't acronyms in technical docs
    "CLOUD IDENTITY INTEGRITY CONFIDENTIALITY AVAILABILITY".split()
)

# Compiled pattern to detect hex color codes (e.g., f39c12, ff6b6b)
_HEX_COLOR_RE = re.compile(r"^[0-9a-f]{3,8}$")


def is_noise_token(token: str) -> bool:
    """Return True if *token* is a stop word, hex color, or skip-listed.

    Works for both lowercase tokens (BM25/TF-IDF) and uppercase tokens
    (acronym extraction).  This is the single source of truth for all
    noise filtering across the codebase.
    """
    if token in _STOP_WORDS:
        return True
    if token in _SKIP_UPPER:
        return True
    # Check lowercase form for mixed-case tokens
    lower = token.lower()
    if lower in _STOP_WORDS:
        return True
    # Hex color codes (e.g., f39c12, ff6b6b, fff)
    if _HEX_COLOR_RE.match(lower):
        return True
    return False


def tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer with stop-word removal for BM25."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return [t for t in tokens if not is_noise_token(t)]


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def clean_text(text: str) -> str:
    """Strip excessive whitespace while preserving paragraph breaks."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.rstrip()
        cleaned.append(stripped)
    result = "\n".join(cleaned)
    # Collapse 3+ blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex.

    Handles common abbreviations and avoids splitting on decimal points.
    """
    # Split on sentence-ending punctuation followed by whitespace and uppercase
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]
