import csv
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu as Bleuold
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from sentence_transformers import SentenceTransformer

class Bleu(Bleuold):
    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)
            bleu_scorer += (hypo[0], ref)
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        return score, scores

def cosine_similarity(vec1, vec2):
    vec1_np = vec1.cpu().numpy()
    vec2_np = vec2.cpu().numpy()
    dot_product = np.dot(vec1_np, vec2_np.T)
    norm_vec1 = np.linalg.norm(vec1_np, axis=1, keepdims=True)
    norm_vec2 = np.linalg.norm(vec2_np, axis=1, keepdims=True)
    cosine_sim = dot_product / np.dot(norm_vec1, norm_vec2.T)
    return cosine_sim

def calculate_metrics(csv_file_path):
    bleu4_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    references = {}
    hypotheses = {}
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for i, row in enumerate(reader):
            references[i] = [row[5]]
            hypotheses[i] = [row[6]]

    bleu4_score, _ = bleu4_scorer.compute_score(references, hypotheses)
    meteor_score, _ = meteor_scorer.compute_score(references, hypotheses)
    _, rouge_scores = rouge_scorer.compute_score(references, hypotheses)
    rouge_l_score = rouge_scores.mean()
    cider_score, _ = cider_scorer.compute_score(references, hypotheses)

    ref_sentences = [refs[0] for refs in references.values()]
    hyp_sentences = [hyps[0] for hyps in hypotheses.values()]
    ref_embeddings = sbert_model.encode(ref_sentences, convert_to_tensor=True)
    hyp_embeddings = sbert_model.encode(hyp_sentences, convert_to_tensor=True)
    cosine_scores = np.diag(cosine_similarity(ref_embeddings, hyp_embeddings))
    sbert_score = np.mean(cosine_scores)

    return {
        "BLEU-1": f"{bleu4_score[0]*100:.3f}",
        "BLEU-4": f"{bleu4_score[3]*100:.3f}",
        "METEOR": f"{meteor_score*100:.3f}",
        "ROUGE-L": f"{rouge_l_score*100:.3f}",
        "CIDER": f"{cider_score*100:.3f}",
        "sBERT": f"{sbert_score*100:.3f}"
    }

# Run and save to CSV
csv_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/test_scores/nonplayername.csv"
results = calculate_metrics(csv_path)

# Save result to CSV
output_metrics_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/test_scores/metrics_summary.csv"
with open(output_metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)


