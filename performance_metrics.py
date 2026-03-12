import string
import re
import unicodedata
"""
Considerations for Jawi Script
    *Unicode Normalization*: We use unicodedata.normalize('NFKC', text) to normalize Unicode characters, ensuring consistent encoding of characters that might have multiple forms.
    *Diacritics*: Arabic script often includes diacritics. Depending on your OCR system's handling of diacritics, you might choose to keep or remove them. The remove_diacritics function provided removes common Arabic diacritics.
    *Punctuation*: We've extended punctuation removal to include common Arabic punctuation marks.
    *Whitespace*: The code normalizes whitespace to ensure consistent tokenization when splitting words.
"""
def remove_diacritics(text):
    arabic_diacritics = re.compile("""
        ّ    | # Shadda
        َ    | # Fatha
        ً    | # Tanwin Fath
        ُ    | # Damma
        ٌ    | # Tanwin Damm
        ِ    | # Kasra
        ٍ    | # Tanwin Kasr
        ْ    | # Sukun
        ـ     # Tatwil/Kashida
    """, re.VERBOSE)
    return re.sub(arabic_diacritics, '', text)
def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = remove_diacritics(text)
    arabic_punctuation = '،؛؟٪۔٬۔'
    all_punctuation = string.punctuation + arabic_punctuation
    translator = str.maketrans('', '', all_punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i  # Deletion
    for j in range(n + 1):
        dp[0][j] = j  # Insertion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]
def compute_mean_CER(gt_texts, pred_texts, normalize=False):
    total_cer = 0
    num_samples = len(gt_texts)
    for gt_text, pred_text in zip(gt_texts, pred_texts):
        if normalize:
            gt_text = normalize_text(gt_text)
            pred_text = normalize_text(pred_text)
        gt_chars = list(gt_text)
        pred_chars = list(pred_text)
        distance = levenshtein_distance(gt_chars, pred_chars)
        max_len = max(len(gt_chars), len(pred_chars))
        cer = distance / max_len if max_len > 0 else 0
        total_cer += cer
    mean_cer = total_cer / num_samples if num_samples > 0 else 0
    return mean_cer
def compute_mean_WER(gt_texts, pred_texts, normalize=False):
    total_wer = 0
    num_samples = len(gt_texts)
    for gt_text, pred_text in zip(gt_texts, pred_texts):
        if normalize:
            gt_text = normalize_text(gt_text)
            pred_text = normalize_text(pred_text)
        gt_words = gt_text.split()
        pred_words = pred_text.split()
        distance = levenshtein_distance(gt_words, pred_words)
        max_len = max(len(gt_words), len(pred_words))
        wer = distance / max_len if max_len > 0 else 0
        total_wer += wer
    mean_wer = total_wer / num_samples if num_samples > 0 else 0
    return mean_wer
def evaluate_ocr(gt_texts, pred_texts):
    cer = compute_mean_CER(gt_texts, pred_texts, normalize=True)
    wer = compute_mean_WER(gt_texts, pred_texts, normalize=True)
    return cer, wer