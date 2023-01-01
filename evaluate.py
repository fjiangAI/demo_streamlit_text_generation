import numpy
from rouge.rouge import Rouge
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as b_score

sys.setrecursionlimit(10000000)


class Evaluator:
    def __init__(self):
        self.rouge = Rouge()

    def compute_rouge(self, source, target):
        """计算rouge-1、rouge-2、rouge-l
        """
        source, target = ' '.join(source), ' '.join(target)
        try:
            scores = self.rouge.get_scores(hyps=source, refs=target)
            return {
                'rouge-1': scores[0]['rouge-1']['f'],
                'rouge-2': scores[0]['rouge-2']['f'],
                'rouge-l': scores[0]['rouge-l']['f'],
            }
        except ValueError:
            return {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0,
            }

    def compute_rouges_directly(self, sources, targets):
        scores = {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
        for id, source in enumerate(sources):
            target = targets[id]
            score = self.compute_rouge(source, target)
            for k, v in scores.items():
                scores[k] = v + score[k]
        result = {k: v / len(targets) for k, v in scores.items()}
        result["rouge-all"] = 0.2 * result["rouge-1"] + 0.3 * result["rouge-2"] + 0.5 * result[
            "rouge-l"]
        return result

    def compute_bleu(self, bleu_type, sources, targets):
        bleu_weight_dict = {"bleu1": (1, 0, 0, 0), "bleu2": (0.5, 0.5, 0, 0), "bleu3": (0.33, 0.33, 0.33),
                            "bleu4": (0.25, 0.25, 0.25, 0.25)}
        score = 0.0
        for id, source in enumerate(sources):
            target = targets[id]
            source, target = ' '.join(source), ' '.join(target)
            score += sentence_bleu(references=[source.split(' ')], hypothesis=target.split(' '),
                                   smoothing_function=SmoothingFunction().method1, weights=bleu_weight_dict[bleu_type])

        score /= len(sources)
        return score

    def compute_bleu_directly(self, sources, targets):
        bleu_type_dict = {"bleu1": self.compute_bleu("bleu1", sources, targets),
                          "bleu2": self.compute_bleu("bleu2", sources, targets),
                          "bleu3": self.compute_bleu("bleu3", sources, targets),
                          "bleu4": self.compute_bleu("bleu4", sources, targets)}
        return bleu_type_dict

    def compute_bert_score_directly(self, sources, targets):
        score = 0.0
        P, R, score = b_score(targets, sources, lang="zh", verbose=True)
        score = score.numpy().tolist()
        score = numpy.mean(score)
        return score

    def compute_all_score(self, sources, targets):
        rouge_result = self.compute_rouges_directly(sources, targets)
        bleu_score = self.compute_bleu_directly(sources, targets)
        bert_score = self.compute_bert_score_directly(sources, targets)
        result_list = [rouge_result["rouge-1"], rouge_result["rouge-2"], rouge_result["rouge-l"],
                       bleu_score["bleu1"], bleu_score["bleu2"], bleu_score["bleu3"], bleu_score["bleu4"],
                       bert_score]
        return result_list
