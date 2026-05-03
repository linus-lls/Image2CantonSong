"""
cantonese_lyrics_evaluator.py  ―  Term-project edition  (v4)
=============================================================
Six metrics, each ∈ [0, 1].  Overall = weighted average.

Key change in v4  (rhyme extraction overhaul)
---------------------------------------------
ROOT-CAUSE FIX  ─  Previous versions extracted the rhyme final from
jp_lines[-1], i.e. the last token produced when the *whole line* was
passed to the jyutping library.  When the library skips non-CJK
characters (spaces, parentheses, punctuation), the last surviving token
can be an interior syllable rather than the line-final character, causing
most _pair_score calls to return 0 even on a song with near-perfect rhyme.

The fix introduces:
  _char_to_jp(char)        — queries jyutping for a SINGLE CJK character.
  _line_rhyme_info(line)   — finds the last CJK char of a text line and
                             calls _char_to_jp on it; falls back to
                             jp_lines[-1] only if the single-char query
                             returns None.

rhyme_consistency_score() and rhyme_debug_info() now accept both
`lines` (original text) and `jp_lines` (pre-converted tokens).

Other v4 improvements
---------------------
• text_to_jp_lines(): strips non-CJK characters BEFORE calling the
  library, preventing space/punctuation tokenisation drift.
• _normalize_jp(): handles slash- and space-separated alternative
  readings (e.g. 'wui6/wui2' → 'wui6').
• diagnose_jp_library(): shows raw library output for test characters
  so conversion failures are immediately visible.
• rhyme_debug_info(): shows (char, jp, final, checked) per line.
• naturalness_score() weights: 0.8 + 0.2 = 1.0 (was 0.9 in v2).

Optional dependencies
---------------------
pycantonese  OR  ToJyutping   metrics 1 & 2
jieba                         metric 3 (falls back to single chars)
numpy                         metric 5 (falls back to pure Python)
"""

import re
import math
import statistics
from collections import Counter


# ══════════════════════════════════════════════════════════════
# § 0  UTILITIES  &  JYUTPING CONVERSION
# ══════════════════════════════════════════════════════════════

def split_lines(text: str) -> list[str]:
    """
    Canonical line splitter for the entire module.

    Splits ONLY on '\\n', strips each line, discards blank lines.
    Intra-line spaces (common in Cantonese lyric formatting) are NEVER
    treated as line separators.
    """
    return [ln.strip() for ln in text.split('\n') if ln.strip()]


def _get_jp_func():
    """
    Return a unified callable  text → list[(char, jyutping | None)].
    Prefers pycantonese ≥ 3.4; falls back to ToJyutping ≥ 1.1.
    Returns None when neither library is installed.
    """
    try:
        import pycantonese
        return lambda t: pycantonese.characters_to_jyutping(t)
    except ImportError:
        pass
    try:
        import ToJyutping
        return lambda t: ToJyutping.get_jyutping_list(t)
    except ImportError:
        pass
    return None


_JP_FUNC = _get_jp_func()


def _normalize_jp(jp) -> str | None:
    """
    Normalise library output to canonical 'syllableN' (e.g. 'jyut6').
    Handles:
      • Standard 'coeng4'
      • Tone digit in the middle 'coe4ng'  (rare library variants)
      • Slash-separated alternatives 'wui6/wui2'  → takes first reading
      • Space-separated alternatives 'coeng4 zoeng6' → takes first reading
    Returns None for unrecognisable input.
    """
    if not jp:
        return None
    s = str(jp).strip().lower()
    # Take only the first alternative when multiple are offered
    if '/' in s:
        s = s.split('/')[0].strip()
    if ' ' in s:
        s = s.split()[0].strip()
    if re.match(r'^[a-z]+[1-6]$', s):
        return s
    m = re.match(r'^([a-z]+)([1-6])([a-z]*)$', s)
    if m:
        return m.group(1) + m.group(3) + m.group(2)
    return None


def _char_to_jp(char: str) -> str | None:
    """
    Query normalised jyutping for a SINGLE CJK character.

    Querying one character at a time eliminates tokenisation drift that
    occurs when a full multi-word line is passed to the library: the
    library cannot misplace the tone-bearing character when there is only
    one character to process.
    """
    if _JP_FUNC is None or not char:
        return None
    try:
        for _, jp in _JP_FUNC(char):
            n = _normalize_jp(jp)
            if n:
                return n
    except Exception:
        pass
    return None


def text_to_jp_lines(text: str) -> list[list[str]] | None:
    """
    Convert lyrics to a list-of-lists of normalised jyutping tokens.
    One inner list per non-empty source line (via split_lines).

    v4: strips non-CJK characters before calling the library so that
    spaces and punctuation cannot displace the final token position.
    """
    if _JP_FUNC is None:
        return None
    result = []
    for line in split_lines(text):
        # Pass CJK-only string to avoid space/punctuation tokenisation drift
        cjk_line = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf]', '', line)
        if not cjk_line:
            result.append([])
            continue
        try:
            pairs  = _JP_FUNC(cjk_line)
            tokens = [_normalize_jp(jp) for _, jp in pairs if jp]
        except Exception:
            tokens = []
        result.append([t for t in tokens if t])
    return result


def diagnose_jp_library() -> dict:
    """
    Test the jyutping library against known Cantonese characters.
    Run this first when debugging unexpectedly low scores.

    Expected output for a correctly working library:
      場 → coeng4 (final: oeng)
      傷 → soeng1 (final: oeng)
      想 → soeng2 (final: oeng)
      常 → soeng4 (final: oeng)
    """
    if _JP_FUNC is None:
        return {'installed': False, 'note': 'Install pycantonese or ToJyutping'}
    test_cases = {
        '場': 'coeng4',
        '傷': 'soeng1',
        '想': 'soeng2',
        '常': 'soeng4',
        '林': 'lam4',
    }
    results = {}
    for char, expected in test_cases.items():
        try:
            raw = _JP_FUNC(char)
            norm = _char_to_jp(char)
            final = _extract_final(norm)
            results[char] = {
                'raw':      raw,
                'norm':     norm,
                'final':    final,
                'expected': expected,
                'ok':       norm == expected,
            }
        except Exception as e:
            results[char] = {'error': str(e)}
    lib_name = 'pycantonese' if 'pycantonese' in str(_JP_FUNC) else 'ToJyutping'
    return {'installed': True, 'library': lib_name, 'tests': results}


# ══════════════════════════════════════════════════════════════
# § 1  TONAL AESTHETICS
# ══════════════════════════════════════════════════════════════
# Two sub-metrics averaged:
#   (a) Consecutive-same-tone penalty: 1 − max_run / N
#   (b) Ping-ze alternation at positions 2 & 4  (二四不同 rule)

def _ping_ze(tone: int) -> int:
    """Return 0 (平) for tones 1–2, 1 (仄) for tones 3–6."""
    return 0 if tone <= 2 else 1


def _max_run(seq: list) -> int:
    """Length of the longest run of identical consecutive values."""
    if not seq:
        return 0
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def tonal_aesthetics_score(jp_lines: list[list[str]]) -> float:
    """
    Score ∈ [0, 1].

    Sub-metric (a): 1 − min(1, max_run / total_syllables).
    Sub-metric (b): proportion of ≥4-syllable lines where tone-class at
        position 2 ≠ tone-class at position 4.
        Falls back to 0.5 when no qualifying line exists.
    """
    all_tones: list[int] = []
    eligible = alternating = 0

    for line in jp_lines:
        tones = [int(jp[-1]) for jp in line if jp and jp[-1].isdigit()]
        all_tones.extend(tones)
        if len(tones) >= 4:
            eligible += 1
            if _ping_ze(tones[1]) != _ping_ze(tones[3]):
                alternating += 1

    if not all_tones:
        return 0.0

    score_a = 1.0 - min(1.0, _max_run(all_tones) / len(all_tones))
    score_b = (alternating / eligible) if eligible else 0.5
    return (score_a + score_b) / 2.0


# ══════════════════════════════════════════════════════════════
# § 2  RHYME CONSISTENCY  (v4 — single-char extraction)
# ══════════════════════════════════════════════════════════════

_INITIALS: list[str] = [
    'gw', 'kw', 'ng',          # multi-char initials MUST precede single-char
    'b', 'p', 'm', 'f',
    'd', 't', 'n', 'l',
    'g', 'k', 'h',
    'w', 'j', 'z', 'c', 's',
]

_RIME_FAMILIES: list[frozenset] = [
    # Open rimes
    frozenset({'ong',  'oeng'              }),   # 場/傷/障/樑  (most common)
    frozenset({'ung',  'ong'               }),   # 中/通
    frozenset({'ing',  'eng'               }),   # 情/誠
    frozenset({'eoi',  'oi',  'ui'         }),   # 堆/雷/回/追
    frozenset({'ou',   'o'                 }),   # 好/報
    frozenset({'iu',   'eu'                }),   # 笑/秋
    frozenset({'ai',   'aai'               }),
    frozenset({'au',   'aau'               }),
    frozenset({'am',   'aam'               }),
    frozenset({'an',   'aan'               }),
    frozenset({'ang',  'aang'              }),
    frozenset({'in',   'an',  'aan'        }),   # 年/千/邊
    frozenset({'im',   'am',  'aam'        }),
    # Checked (入聲) rimes
    frozenset({'ap',   'aap'               }),
    frozenset({'at',   'aat'               }),
    frozenset({'ak',   'aak'               }),
    frozenset({'ik',   'ak',  'aak'        }),
    frozenset({'it',   'at',  'aat'        }),
    frozenset({'ip',   'ap',  'aap'        }),
]


def _extract_final(jp: str | None) -> str | None:
    """Strip tone digit and initial consonant(s); return the 韻母 (rime)."""
    if not jp:
        return None
    core = re.sub(r'[1-6]$', '', jp)
    for ini in _INITIALS:
        if core.startswith(ini):
            tail = core[len(ini):]
            return tail if tail else core   # bare-initial syllable edge case
    return core


def _is_checked(jp: str | None) -> bool:
    """True when the syllable ends in a stop coda (-p / -t / -k)."""
    return bool(jp and re.sub(r'[1-6]$', '', jp).endswith(('p', 't', 'k')))


def _vowel_nucleus(final: str) -> str:
    """Extract the first maximal vowel sequence from a rime string."""
    m = re.search(r'[aeiou]+', final) if final else None
    return m.group() if m else ''


def _pair_score(f1: str, f2: str, chk1: bool, chk2: bool) -> float:
    """
    Compare two line-final rimes.

    Returns
    -------
    1.0  exact rime match
    0.8  same Cantonese rime family  (e.g. ong / oeng)
    0.4  same vowel nucleus          (loose near-rhyme fallback)
    0.0  入聲 / 非入聲 mismatch, or phonetically unrelated
    """
    if f1 is None or f2 is None:
        return 0.0
    if chk1 != chk2:
        return 0.0
    if f1 == f2:
        return 1.0
    for family in _RIME_FAMILIES:
        if f1 in family and f2 in family:
            return 0.8
    n1, n2 = _vowel_nucleus(f1), _vowel_nucleus(f2)
    if n1 and n1 == n2:
        return 0.4
    return 0.0


def _rhyme_density(
    finals:  list[str | None],
    checked: list[bool],
    window:  int = 4,
) -> float:
    """
    Sliding-window rhyme density.

    For each line i, score against lines i+1 … i+window−1.
    Returns the mean score across all non-None pairs within the window.
    None finals are excluded from both numerator and denominator.
    """
    total = score = 0.0
    n = len(finals)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            if finals[i] is not None and finals[j] is not None:
                score += _pair_score(finals[i], finals[j], checked[i], checked[j])
                total += 1.0
    return score / total if total else 0.0


# ── v4 core: per-line last-character extraction ─────────────────

def _line_rhyme_info(
    line:    str,
    jp_line: list[str] | None = None,
) -> tuple[str | None, str | None, bool]:
    """
    Return (last_jp, final, is_checked) for the last CJK character in `line`.

    Strategy
    --------
    1. Find the last CJK character in the original text line.
    2. Query _char_to_jp() for that single character (most reliable).
    3. If that returns None AND jp_line is provided, fall back to jp_line[-1].

    Returns (jp, final, checked).  All three are None/False when no CJK is found.
    """
    cjk = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', line)
    if not cjk:
        return None, None, False

    jp = _char_to_jp(cjk[-1])

    # Fallback: use jp_line if the single-char query failed
    if jp is None and jp_line:
        jp = jp_line[-1]

    return jp, _extract_final(jp), (_is_checked(jp) if jp else False)


def rhyme_consistency_score(
    lines:    list[str],
    jp_lines: list[list[str]] | None = None,
) -> float:
    """
    Score ∈ [0, 1]: sliding-window rhyme density.

    Parameters
    ----------
    lines    : original text lines (split_lines output)
    jp_lines : pre-converted token lists (used only as fallback per line)

    The score is computed from _line_rhyme_info(), which targets the last
    CJK character of each line directly, sidestepping any tokenisation
    ordering issues in the full-line library call.
    """
    if _JP_FUNC is None or not lines:
        return 0.0

    jp_map: dict[int, list[str]] = {}
    if jp_lines:
        # Build index: map valid jp_lines entries to their position
        valid_idx = [i for i, ln in enumerate(jp_lines) if ln]
        for rank, idx in enumerate(valid_idx):
            jp_map[idx] = jp_lines[idx]

    finals:  list[str | None] = []
    checked: list[bool]       = []

    for i, line in enumerate(lines):
        _, final, chk = _line_rhyme_info(line, jp_map.get(i))
        finals.append(final)
        checked.append(chk)

    if sum(1 for f in finals if f is not None) < 2:
        return 0.0
    return _rhyme_density(finals, checked, window=4)


# ── Template-based scheme scoring (diagnostics only) ────────────

_SCHEMES: dict[str, list[str]] = {
    'AAAA': ['A'],
    'AABB': ['A', 'A', 'B', 'B'],
    'ABAB': ['A', 'B', 'A', 'B'],
    'XBXB': ['X', 'B', 'X', 'B'],
    'AABA': ['A', 'A', 'B', 'A'],
    'ABBA': ['A', 'B', 'B', 'A'],
}


def _score_scheme(
    finals:  list[str | None],
    checked: list[bool],
    pattern: list[str],
) -> float:
    n = len(finals)
    if n < 2:
        return 0.0
    labels  = [pattern[i % len(pattern)] for i in range(n)]
    anchors: dict[str, tuple] = {}
    total = score = 0.0
    for i, lbl in enumerate(labels):
        if lbl == 'X' or finals[i] is None:
            continue
        if lbl not in anchors:
            anchors[lbl] = (finals[i], checked[i])
        else:
            f0, c0 = anchors[lbl]
            score += _pair_score(finals[i], f0, checked[i], c0)
            total += 1
    return score / total if total else 0.0


def rhyme_debug_info(
    lines:    list[str],
    jp_lines: list[list[str]] | None = None,
) -> dict:
    """
    Detailed diagnostic for the rhyme dimension.

    Shows (char, jp, final, checked) per line plus per-template scores
    and the primary rhyme_density metric.
    """
    if _JP_FUNC is None:
        return {'error': 'No jyutping library installed'}
    if not lines:
        return {'error': 'No lines provided'}

    jp_map: dict[int, list[str]] = {}
    if jp_lines:
        valid_idx = [i for i, ln in enumerate(jp_lines) if ln]
        for rank, idx in enumerate(valid_idx):
            jp_map[idx] = jp_lines[idx]

    data: list[tuple] = []   # (char, jp, final, checked)
    for i, line in enumerate(lines):
        cjk = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', line)
        if not cjk:
            continue
        jp, final, chk = _line_rhyme_info(line, jp_map.get(i))
        data.append((cjk[-1], jp, final, chk))

    if len(data) < 2:
        return {'error': 'Too few lines with CJK characters'}

    finals  = [d[2] for d in data]
    checked = [d[3] for d in data]

    scheme_scores = {
        name: round(_score_scheme(finals, checked, pat), 3)
        for name, pat in _SCHEMES.items()
    }
    best    = max(scheme_scores, key=scheme_scores.get)
    density = round(_rhyme_density(finals, checked, window=4), 3)

    return {
        'detected_finals': data,        # (char, jp, final, checked) per line
        'scheme_scores':   scheme_scores,
        'best_scheme':     best,
        'best_score':      scheme_scores[best],
        'rhyme_density':   density,     # ← the actual scoring metric
    }


# ══════════════════════════════════════════════════════════════
# § 3  LEXICAL DIVERSITY
# ══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    cjk = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
    try:
        import jieba
        words = [w for w in jieba.cut(text)
                 if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', w)]
        return words if words else cjk
    except ImportError:
        return cjk


def _mattr(tokens: list[str], window: int = 10) -> float:
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    windows = [tokens[i: i + window] for i in range(len(tokens) - window + 1)]
    return sum(len(set(w)) / window for w in windows) / len(windows)


def _yule_k_score(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    N   = len(tokens)
    fof = Counter(Counter(tokens).values())
    M2  = sum(m * m * v for m, v in fof.items())
    K   = 1e4 * (M2 - N) / (N * N)
    return 1.0 - min(1.0, max(0.0, K) / 200.0)


def lexical_diversity_score(text: str) -> float:
    """Score ∈ [0, 1]: average of MATTR and inverted Yule's K."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return (_mattr(tokens) + _yule_k_score(tokens)) / 2.0


# ══════════════════════════════════════════════════════════════
# § 4  STRUCTURAL REGULARITY
# ══════════════════════════════════════════════════════════════

def _cjk_len(line: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', line))


def structural_regularity_score(lines: list[str]) -> float:
    """
    Score ∈ [0, 1].
    (a) max(0, 1 − CV)  where CV = σ / μ of per-line CJK lengths.
    (b) fraction of lines with length ∈ [3, 11].
    """
    lengths = [_cjk_len(l) for l in lines if l.strip()]
    if not lengths:
        return 0.0
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    cv      = (statistics.stdev(lengths) / mean) if len(lengths) >= 2 else 0.0
    score_a = max(0.0, 1.0 - cv)
    score_b = sum(1 for l in lengths if 3 <= l <= 11) / len(lengths)
    return (score_a + score_b) / 2.0


# ══════════════════════════════════════════════════════════════
# § 5  SEMANTIC COHERENCE  (v5 — refrain + phrase reuse)
# ══════════════════════════════════════════════════════════════
#
# Why v4 scored low
# -----------------
# _repetition_coherence() used:
#   numerator   = count of UNIQUE lines that appear >1 time
#   denominator = count of all UNIQUE lines
#
# A double-chorus song with 38 lines / 6 repeated pairs gives:
#   6 / 32 = 0.19  →  scaled  0.19 / 0.60 = 0.31
#
# The fix:
#   numerator   = ALL instances of repeated lines  (both copies count)
#   denominator = ALL lines (including duplicates)
#
#   12 / 38 = 0.32  →  scaled  0.32 / 0.35 = 0.91  ✓
#
# Phrase reuse adds a second independent signal that works even when
# full lines are not identical (e.g. 正常/砒霜/異象 recur across variants).

def _refrain_score(texts: list[str]) -> float:
    """
    Fraction of all line-instances that are repeated lines.
    A double chorus where ~32 % of lines are repeats scores ≈ 1.0.
    """
    if not texts:
        return 0.0
    freq     = Counter(texts)
    repeated = sum(v for v in freq.values() if v > 1)   # ALL repeated instances
    return min(1.0, repeated / len(texts) / 0.35)


def _phrase_reuse_score(texts: list[str]) -> float:
    """
    Fraction of distinct CJK character bigrams (within each line)
    that appear more than once anywhere in the song.

    Captures recurring motifs without requiring full-line repetition.
    A reuse rate of 25 % or above scores 1.0.
    """
    all_bigrams: list[tuple] = []
    for line in texts:
        chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', line)
        all_bigrams.extend(zip(chars, chars[1:]))
    if not all_bigrams:
        return 0.0
    freq     = Counter(all_bigrams)
    repeated = sum(1 for v in freq.values() if v > 1)
    return min(1.0, repeated / len(freq) / 0.25)


def semantic_coherence_score(lines: list[str], word_vectors=None) -> float:
    """
    Score ∈ [0, 1].

    Sub-metric (a): refrain density   — fraction of lines that are repeats
    Sub-metric (b): phrase reuse      — fraction of CJK bigrams appearing 2+ times

    base = (a + b) / 2

    With word_vectors: blends base (⅔) with first/second-half cosine
    similarity (⅓).  Word vectors improve sensitivity to topic drift
    in through-composed songs but are not required.

    Typical ranges
    --------------
    Strong chorus song  (e.g. 異象)  :  0.7 – 0.95
    Single chorus       (one repeat)  :  0.4 – 0.65
    Through-composed    (no repeats)  :  0.0 – 0.25  ← low is honest here
    """
    texts = [l.strip() for l in lines if l.strip()]
    if not texts:
        return 0.0

    score_a = _refrain_score(texts)
    score_b = _phrase_reuse_score(texts)
    base    = (score_a + score_b) / 2.0

    if word_vectors is None:
        return base

    non_empty = [l for l in lines if l.strip()]
    mid = len(non_empty) // 2
    v1  = _avg_vector(' '.join(non_empty[:mid]), word_vectors)
    v2  = _avg_vector(' '.join(non_empty[mid:]), word_vectors)
    if v1 is None or v2 is None:
        return base

    # Rescale cosine [-1, 1] to [0, 1] before blending
    cos = (_cosine(v1, v2) + 1.0) / 2.0
    return (base * 2.0 + cos) / 3.0

# ══════════════════════════════════════════════════════════════
# § 6  NATURALNESS
# ══════════════════════════════════════════════════════════════

_PARTICLES: set[str] = set('嘅咗喺係唔啩囉咋喎㗎嗎呀啊哦哩啦')


def _bigram_entropy(text: str) -> float:
    """H_norm = H / log₂(V) ∈ [0, 1]."""
    chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
    if len(chars) < 2:
        return 0.0
    bigrams = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
    freq    = Counter(bigrams)
    total   = len(bigrams)
    H       = -sum((c / total) * math.log2(c / total) for c in freq.values())
    max_H   = math.log2(len(freq)) if len(freq) > 1 else 1.0
    return H / max_H


def _particle_score(text: str) -> float:
    """
    Inverted-U on particle density (peak at 8 %):
      density ≤ 0.08  →  rises linearly 0 → 1
      density > 0.08  →  falls linearly back to 0 at density = 0.25
    """
    cjk = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
    if not cjk:
        return 0.0
    ratio = sum(1 for c in cjk if c in _PARTICLES) / len(cjk)
    if ratio <= 0.08:
        return ratio / 0.08
    return max(0.0, 1.0 - (ratio - 0.08) / 0.17)


def naturalness_score(text: str) -> float:
    """Score ∈ [0, 1]: 0.8 × bigram entropy + 0.2 × particle score (sums to 1.0)."""
    return _bigram_entropy(text) * 0.8 + _particle_score(text) * 0.2


# ══════════════════════════════════════════════════════════════
# § MAIN EVALUATOR
# ══════════════════════════════════════════════════════════════

_GRADES: list[tuple] = [
    (0.85, 'A'),
    (0.70, 'B'),
    (0.55, 'C'),
    (0.40, 'D'),
    (0.00, 'F'),
]

_HINTS: dict[str, str] = {
    'tonal':     '聲調：避免連續相同聲調，注意每行第二、四字的平仄交替。',
    'rhyme':     '押韻：確保行末字韻母相近，建立一致的押韻密度。',
    'lexical':   '詞彙：增加用詞多樣性，減少同一詞語的頻繁重複。',
    'structure': '結構：統一行長，或建立規律的長短句交替模式。',
    'coherence': '連貫性：加入重複副歌段落以強化整體結構感。',
    'natural':   '自然度：適度使用語氣助詞，使語言更自然流暢。',
}


def evaluate_cantonese_lyrics(
    lyrics_text:  str,
    word_vectors: dict | None = None,
    weights:      dict | None = None,
) -> dict:
    """
    Evaluate Cantonese lyrics across six dimensions.

    Parameters
    ----------
    lyrics_text  : str
    word_vectors : dict | None   — optional {char: vector} for metric 5
    weights      : dict | None   — per-metric weights (need not sum to 1;
                                   normalised internally).
                                   Keys: tonal / rhyme / lexical /
                                         structure / coherence / natural

    Returns
    -------
    dict
        'scores'      {metric_key: float | None}   None = library missing
        'overall'     float ∈ [0, 1]
        'grade'       str   'A' – 'F'
        'suggestions' list  hints for metrics scoring < 0.5
        'rhyme_debug' dict  density + (char, jp, final, checked) per line
    """
    _default_w = {
        'tonal':     2,
        'rhyme':     1,
        'natural':   1,
        'lexical':   1,
        'structure': 1,
        'coherence': 1,
    }
    w = weights or _default_w
    w = {k: v / sum(w.values()) for k, v in w.items()}

    lines    = split_lines(lyrics_text)
    jp_lines = text_to_jp_lines(lyrics_text)   # used for tonal; also fallback for rhyme

    scores: dict[str, float | None] = {}
    if jp_lines is not None:
        scores['tonal'] = tonal_aesthetics_score(jp_lines)
        # v4: pass both lines AND jp_lines; _line_rhyme_info uses jp_lines
        # only as a per-line fallback when single-char lookup returns None
        scores['rhyme'] = rhyme_consistency_score(lines, jp_lines)
    else:
        scores['tonal'] = None
        scores['rhyme'] = None

    scores['lexical']   = lexical_diversity_score(lyrics_text)
    scores['structure'] = structural_regularity_score(lines)
    scores['coherence'] = semantic_coherence_score(lines, word_vectors)
    scores['natural']   = naturalness_score(lyrics_text)

    available = {k: v for k, v in scores.items() if v is not None}
    avail_w   = sum(w[k] for k in available)
    overall   = (sum(scores[k] * w[k] for k in available) / avail_w
                 if avail_w else 0.0)

    grade       = next(g for threshold, g in _GRADES if overall >= threshold)
    suggestions = [_HINTS[k] for k, v in scores.items()
                   if v is not None and v < 0.5]

    rdebug = (rhyme_debug_info(lines, jp_lines)
              if jp_lines is not None
              else {'error': 'No jyutping library installed'})

    return {
        'scores':      scores,
        'overall':     round(overall, 4),
        'grade':       grade,
        'suggestions': suggestions,
        'rhyme_debug': rdebug,
    }


# ══════════════════════════════════════════════════════════════
# § DEMO
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    sample = """\
明月光 為何又照地堂
寧願在公園躲藏 不想喝湯
任由目光 留在漫畫一角
為何望母親一眼就如罰留堂

孩童只盼望歡樂 大人只知道期望
為何都不大懂得努力體恤對方
大門外有蟋蟀 迴響卻如同幻覺
Shall we talk Shall we talk
就當重新手拖手去上學堂

陪我講 陪我講出我們最後何以生疏
誰怕講 誰會可悲得過孤獨探戈
難得 可以同座 何以 要忌諱赤裸
如果心聲真有療效 誰怕暴露更多
你別怕我

螢幕發光 無論什麼都看
情人在分手邊緣只敢喝湯
若沉默似金 還談什麼戀愛
寧願在發聲機器面前笑著忙

成人只寄望收穫 情人只聽見承諾
為何都不大懂得努力珍惜對方
螳螂面對蟋蟀 迴響也如同幻覺
Shall we talk Shall we talk
就算牙關開始打震 別說謊

陪我講 陪我講出我們最後何以生疏
誰怕講 誰會可悲得過孤獨探戈
難得 可以同座 何以 要忌諱赤裸
如果心聲真有療效 誰怕暴露更多

陪我講 陪我親身正視眼淚誰跌得多
無法講 除非彼此已失去了能力觸摸
鈴聲 可以寧靜 難過 卻避不過
如果沉默太沉重 別要輕輕帶過

明月光 為何未照地堂
孩兒在公司很忙 不需喝湯
And Shall we talk 斜陽白趕一趟
沉默令我聽得見葉兒聲聲降
"""

    # ── Library diagnostics first ─────────────────────────────
    diag = diagnose_jp_library()
    print('═' * 56)
    if not diag['installed']:
        print('  ⚠  未偵測到粵語拼音庫（pycantonese / ToJyutping）')
        print('     聲調及押韻指標將顯示為 N/A')
    else:
        print(f'  粵語拼音庫: {diag["library"]}')
        all_ok = all(v.get('ok') for v in diag['tests'].values() if 'ok' in v)
        status = '✓ 轉換正常' if all_ok else '⚠ 部分字符轉換異常 — 請檢查下方診斷'
        print(f'  庫測試:    {status}')
        if not all_ok:
            for char, info in diag['tests'].items():
                if not info.get('ok'):
                    print(f'    {char}: 預期 {info["expected"]}, '
                          f'實際 raw={info.get("raw")} norm={info.get("norm")}')
    print('═' * 56)

    result = evaluate_cantonese_lyrics(sample)

    _LABELS = {
        'tonal':     '聲調格律',
        'rhyme':     '押韻一致性',
        'lexical':   '詞彙豐富度',
        'structure': '結構規整度',
        'coherence': '語義連貫性',
        'natural':   '自然度',
    }

    print('   粵語歌詞評估報告（Term Project v4）')
    print('═' * 56)
    for key, label in _LABELS.items():
        val = result['scores'][key]
        if val is None:
            print(f'  {label:<8}  N/A   (需安裝 pycantonese 或 ToJyutping)')
        else:
            bar = '█' * int(val * 20)
            print(f'  {label:<8}  {val:.3f}  {bar}')
    print('─' * 56)
    print(f'  總分    {result["overall"]:.4f}   等級  {result["grade"]}')
    print('─' * 56)

    rd = result['rhyme_debug']
    if 'error' not in rd:
        print(f'\n  [押韻診斷]')
        print(f'    韻密度（實際評分）: {rd["rhyme_density"]:.3f}')
        print(f'    最佳模板（參考）:   {rd["best_scheme"]}  ({rd["best_score"]:.3f})')
        print('    各模板得分:',
              '  '.join(f'{k}={v}' for k, v in rd['scheme_scores'].items()))
        print(f'\n  {"行末字":<6} {"拼音":<12} {"韻母":<10} 入聲')
        print('  ' + '─' * 36)
        for char, jp, final, chk in rd['detected_finals']:
            jp_str    = jp    if jp    else '(無)'
            final_str = final if final else '(無)'
            tag       = '入'  if chk   else '  '
            print(f'  {char:<6} {jp_str:<12} {final_str:<10} {tag}')
    else:
        print(f'\n  [押韻診斷]  {rd["error"]}')

    print()
    if result['suggestions']:
        print('  改善建議：')
        for hint in result['suggestions']:
            print(f'    · {hint}')
    else:
        print('  各項指標均達標，暫無特別建議。')
    print('═' * 56)