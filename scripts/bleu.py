import collections
import math
import sys
import codecs

''' FULL_BLEU
    1: Individual(Smooth & NonSmooth) + Cumulative(Smooth & NonSmooth)(NLTK & TF_NMT) 
    0: Cumulative(Smooth & NonSmooth)(TF_NMT)  [Default]
'''
FULL_BLEU = 0
DEBUG = 0

if 1 == FULL_BLEU:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.bleu_score import SmoothingFunction 

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=True):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)

def round_for_list(x, precision):
    return [round(data, precision) for data in x]

def get_individual_BLEU_1_to_4_by_nltk(references, candidate):
    scores = []
    if DEBUG >= 3:
        print("references:", references)
        print("candidate:", candidate)
    smoothing_function = SmoothingFunction().method0
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(1, 0, 0, 0), smoothing_function=smoothing_function)) # 1 gram
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(0, 1, 0, 0), smoothing_function=smoothing_function)) # 2 gram
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(0, 0, 1, 0), smoothing_function=smoothing_function)) # 3 gram
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(0, 0, 0, 1), smoothing_function=smoothing_function)) # 4 gram

    scores = round_for_list(scores, 4)
    print("Individual(Smooth):        \t" + "\t".join(map(str, scores)))
    return scores

def get_cumulative_BLEU_1_to_4_by_nltk(references, candidate):
    scores = []
    smoothing_function = SmoothingFunction().method0
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(1, 0, 0, 0), smoothing_function=smoothing_function)) # 1 gram
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)) # 2 gram
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)) # 3 gram
    scores.append(100*corpus_bleu(references, candidate, \
        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)) # 4 gram
    scores = round_for_list(scores, 4)
    print("Cumulative(Smooth)(NLTK):\t" + "\t".join(map(str, scores)))
    return scores

def fetch_data(references_with_source_file, uniq_translation_file, is_merge_same_src=1):
    references_reader = codecs.open(references_with_source_file, 'r', 'utf-8')
    translation_reader = codecs.open(uniq_translation_file, 'r', 'utf-8')

    translations = []
    for line in translation_reader:
        slots = line.strip("\n").replace(' </s>', '').split()
        translations.append(slots)

    if 1 == is_merge_same_src:
        references = []
        last_query = ""
        refs_for_a_query = []
        for line in references_reader:
            slots = line.strip("\n").split("\t")
            query, ref = slots
            if query != last_query:
                if len(refs_for_a_query) > 0:
                    references.append(refs_for_a_query)
                refs_for_a_query = []
                last_query = query
            ref = ref.split()
            refs_for_a_query.append(ref)
        references.append(refs_for_a_query)
    elif 0 == is_merge_same_src:
        references = []
        for line in references_reader:
            slots = line.strip("\n").split("\t")
            query, ref = slots
            ref = ref.split()
            refs_for_a_query = [ref]
            references.append(refs_for_a_query) 

    if DEBUG >= 2:
        print("candidate size:", len(translations))
        print("references size:", len(references))
        for reference in references:
            print("reference len:", len(reference))
            print("reference:", reference)
    return references, translations

def get_bleus(references, translations):
    cumulative_smooth_set = []
    for N_of_gram in range(1, 5): # BLEU-1 to BLEU-4
        bleu_score, precisions, bp, ratio, translation_length, reference_length \
            = compute_bleu(references, translations, max_order=N_of_gram, smooth=True)
        cumulative_smooth_set.append(100 * bleu_score)
    cumulative_smooth_set = round_for_list(cumulative_smooth_set, 4)
    
    cumulative_non_smooth_set = []
    for N_of_gram in range(1, 5): # BLEU-1 to BLEU-4
        bleu_score, precisions, bp, ratio, translation_length, reference_length \
            = compute_bleu(references, translations, max_order=N_of_gram, smooth=False)
        cumulative_non_smooth_set.append(100 * bleu_score)
    cumulative_non_smooth_set = round_for_list(cumulative_non_smooth_set, 4)

    name_set = ["              Type             ", "Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4"]
    print("\t".join(name_set))
    print("Cumulative(Smooth):        \t" + "\t".join(map(str, cumulative_smooth_set)))
    print("Cumulative(NonSmooth):        \t" + "\t".join(map(str, cumulative_non_smooth_set)))

    if 1 == FULL_BLEU:
        get_individual_BLEU_1_to_4_by_nltk(references, translations)
        get_cumulative_BLEU_1_to_4_by_nltk(references, translations)

def main():
    is_merge_same_src = int(sys.argv[3])#1: merge(for normal seq2seq predict) 0: not merge(for line by line predict)
    references, translations = fetch_data(sys.argv[1], sys.argv[2], is_merge_same_src)
    get_bleus(references, translations)

main()
