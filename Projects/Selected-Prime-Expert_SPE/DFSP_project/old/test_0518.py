# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
from itertools import product
from pathlib import Path
import clip
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import cv2

from utils import *
from loss import loss_calu
from parameters import parser, YML_PATH
from dataset import CompositionDataset
from model.dfsp import DFSP

from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"


class Evaluator:
    """
    Evaluator class, adapted from:
    https://github.com/Tushar-N/attributes-as-operators

    With modifications from:
    https://github.com/ExplainableML/czsl
    """

    def __init__(self, dset, model):

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe',
        # 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [
            (dset.attr2idx[attr],
             dset.obj2idx[obj]) for attr,
            obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        # open world
        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        # masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle
        # setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            """
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            """
            _, pair_pred = _scores.topk(
                topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update(
            {"unbiased_open": get_pred_from_scores(orig_scores, topk)}
        )
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to(
            'cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        # Return only attributes that are in our pairs
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        # Repeat mask along pairs dimension
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias  # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        # sort returns indices of k largest values
        _, pair_pred = closed_scores.topk(topk, dim=1)
        # _, pair_pred = scores.topk(topk, dim=1)  # sort returns indices of k
        # largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(
            self,
            predictions,
            attr_truth,
            obj_truth,
            pair_truth,
            allpred,
            topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (
                attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            )
            obj_match = (
                obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            # Calculating class average accuracy

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(
                seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        # Closed world
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        # Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
            unseen_ind
        ]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[
            0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(
                scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats

#-----------更改內容--------------
#輸入pair [1962,2]
#輸出所有obj情況下的att 他是一個list
def construct_all_fv_proj_idx(dataset,pairs):
    all_fv_proj_idx = []
    for i in range(len(dataset.objs)):
        fv_proj_idx = 0
        fv_proj_idx = torch.Tensor().cuda()
        for obj_idx in range(len(pairs)):
            if pairs[obj_idx][1] ==i:
                fv_proj_idx = torch.cat([fv_proj_idx, pairs[obj_idx]], dim=0)
        fv_proj_idx=fv_proj_idx.reshape(-1,2)
        all_fv_proj_idx.append(fv_proj_idx)
    return all_fv_proj_idx

import os
from pathlib import Path
def plot_dot_reliability_diagram(ece_value, bin_confidences, bin_accuracies, model_index,test_dataset,config):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.scatter(bin_confidences, bin_accuracies, marker='o', color='blue', label="Model {}".format(model_index + 1))
    plt.xlabel("Confidence", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Reliability Diagram for Model {} (ECE={:.4f})".format(model_index + 1, ece_value.item()), fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.gca().set_axisbelow(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    Path(config.save_path + '/plt/'+str(len(test_dataset.pairs))).mkdir(parents=True, exist_ok=True)
    plt.savefig(config.save_path + '/plt/'+str(len(test_dataset.pairs))+'/'+test_dataset.phase+'_'+"dot_reliability_diagram_model_{}.png".format(model_index + 1))


def compute_ece(probs, targets, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0
    bin_confidences = []
    bin_accuracies = []
    confidences, _ = torch.max(probs, dim=1)  # Compute the confidence values
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            sample_indices = torch.where(in_bin)[0]
            bin_targets = targets[sample_indices]
            bin_probs = probs[sample_indices]
            true_prob_in_bin = (bin_targets == torch.argmax(bin_probs, dim=1)).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - true_prob_in_bin) * prop_in_bin
            bin_confidences.append(avg_confidence_in_bin.item())
            bin_accuracies.append(true_prob_in_bin.item())
        else:
            bin_confidences.append(None)
            bin_accuracies.append(None)

    return ece, bin_confidences, bin_accuracies



def compute_uce(probs, targets, n_bins=10):
    _, nattrs =probs.size()
    nattrs = torch.tensor(nattrs) 
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    uce = 0
    bin_uncertainties = []
    bin_errors = []
    prop_in_bin_values = []
    bin_n_samples = []
    bin_variances = []
    # Compute the uncertainty values (entropy)
    uncertainties = (1/torch.log(nattrs))*(-torch.sum(probs * torch.log(probs + 1e-12), dim=1))
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (uncertainties >= bin_lower) * (uncertainties < bin_upper)
        prop_in_bin = in_bin.float().mean()
        prop_in_bin_values.append(prop_in_bin.item() if prop_in_bin.item() > 0 else None)
        if prop_in_bin.item() > 0:
            sample_indices = torch.where(in_bin)[0]
            bin_targets = targets[sample_indices]
            bin_probs = probs[sample_indices]
            error_in_bin = (bin_targets != torch.argmax(bin_probs, dim=1)).float().mean()
            avg_uncertainty_in_bin = uncertainties[in_bin].mean()
            uce += torch.abs(avg_uncertainty_in_bin - error_in_bin) * prop_in_bin
            bin_uncertainties.append(avg_uncertainty_in_bin.item())
            bin_errors.append(error_in_bin.item())
            n_samples_in_bin = sample_indices.size(0)
            bin_n_samples.append(n_samples_in_bin)
            bin_variances.append(torch.var((bin_targets != torch.argmax(bin_probs, dim=1)).float()).item())
        else:
            bin_uncertainties.append(None)
            bin_errors.append(None)
            bin_n_samples.append(None)
            bin_variances.append(None)

    return uce, bin_uncertainties, bin_errors, prop_in_bin_values, bin_n_samples, bin_variances

def plot_dot_UCE_diagram(uce_value, bin_uncertainties, bin_errors, model_index, test_dataset, config, prop_in_bin_values, bin_n_samples, bin_variances, threshold=0.005):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    
    # 筛选prop_in_bin值大于等于threshold的点
    valid_indices = [i for i, prop in enumerate(prop_in_bin_values) if prop is not None and prop >= threshold]
    valid_bin_uncertainties = [bin_uncertainties[i] for i in valid_indices]
    valid_bin_errors = [bin_errors[i] for i in valid_indices]
    valid_prop_in_bin_values = [prop_in_bin_values[i] for i in valid_indices]
    valid_bin_n_samples  = [bin_n_samples[i] for i in valid_indices]
    valid_bin_variances  = [bin_variances[i] for i in valid_indices]
    
    plt.scatter(valid_bin_uncertainties, valid_bin_errors, marker='o', color='blue', label="Model {}".format(model_index + 1))
    plt.xlabel("Uncertainty", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.title("Reliability Diagram for Model {} (UCE={:.4f})".format(model_index + 1, uce_value.item()), fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.gca().set_axisbelow(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    for i, txt in enumerate(valid_bin_n_samples):
        plt.annotate("n={}".format(txt), (valid_bin_uncertainties[i], valid_bin_errors[i]), fontsize=8, ha='center', va='bottom', textcoords="offset points", xytext=(0,5))
        plt.annotate("var={:.2f}".format(valid_bin_variances[i]), (valid_bin_uncertainties[i], valid_bin_errors[i]), fontsize=8, ha='center', va='bottom', textcoords="offset points", xytext=(0,20))
#     plt.savefig(config.save_path + '/plt/'+str(len(test_dataset.pairs))+'/'+test_dataset.phase+'_'+"UCE_model_{}.png".format(model_index + 1))
    Path(config.save_path + '/plt/'+str(len(test_dataset.pairs))).mkdir(parents=True, exist_ok=True)
    plt.savefig(config.save_path + '/plt/'+str(len(test_dataset.pairs))+'/'+test_dataset.phase+'_'+"UCE_model_{}.png".format(model_index + 1))
### 5/1新增
def choose_best_expert(probs_expert1, probs_expert2, targets,targets_pairs,pairs ,n_bins=10):
    uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1, bin_variances_ep1 = compute_uce(probs_expert1, targets, n_bins)
    uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2, bin_variances_ep2 = compute_uce(probs_expert2, targets_pairs, n_bins)

    # Compute uncertainties for both experts
    _, nattrs = probs_expert1.size()
    _, npairs = probs_expert2.size()
    nattrs = torch.tensor(nattrs)
    npairs = torch.tensor(npairs)
    uncertainties_expert1 = (1/torch.log(nattrs))*(-torch.sum(probs_expert1 * torch.log(probs_expert1 + 1e-12), dim=1))
    uncertainties_expert2 = (1/torch.log(npairs))*(-torch.sum(probs_expert2 * torch.log(probs_expert2 + 1e-12), dim=1))
    # Find error rates for both experts
    error_rates_expert1 = find_error_rates(uncertainties_expert1, bin_uncertainties_expert1, bin_errors_expert1)
    error_rates_expert2 = find_error_rates(uncertainties_expert2, bin_uncertainties_expert2, bin_errors_expert2)
    # Choose the expert with lower error rate for each sample
    chosen_expert = (error_rates_expert1 < error_rates_expert2)

    # Get the predictions from both experts
    preds_expert1 = torch.argmax(probs_expert1, dim=1)
    preds_expert2_pairs = torch.argmax(probs_expert2, dim=1)
    preds_expert2 = pairs[preds_expert2_pairs][:, 0].cpu()

    # Choose the final prediction based on the chosen expert
    final_predictions = torch.where(chosen_expert, preds_expert1, preds_expert2)
    return final_predictions

def find_error_rates(uncertainties, bin_uncertainties, bin_errors):
    error_rates = []
    for uncertainty in uncertainties:
        found = False
        for idx, (bin_uncertainty_lower, bin_uncertainty_upper, bin_error) in enumerate(zip(bin_uncertainties[:-1], bin_uncertainties[1:], bin_errors)):
            if bin_uncertainty_lower is not None and bin_uncertainty_upper is not None and bin_error is not None:
                if bin_uncertainty_lower <= uncertainty.item() < bin_uncertainty_upper:

                    error_rates.append(bin_error)
                    found = True
                    break
        if not found:
            if bin_uncertainties[0] is not None: 
                if 0 <=uncertainty< bin_uncertainties[0]:
                    error_rates.append(bin_errors[0])
                else:    
                    for bin_error in reversed(bin_errors):
                        if bin_error is not None:
                            error_rates.append(bin_error)
                            break
            else:
                if bin_uncertainties[1] is not None: 
                    error_rates.append(bin_errors[1])
                else:
                    error_rates.append(bin_errors[2])
    return torch.tensor(error_rates)

def accuracy(y_true, y_pred):

    # 計算正確預測的數量
    correct_predictions = torch.sum(y_true == y_pred)

    # 計算準確度
    accuracy = correct_predictions.item() / y_true.size(0)

    return accuracy
def choose_best_expert_ex(probs_expert1, probs_expert2, targets, n_bins=10):
    # Compute UCE and bin values for both experts
    uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1, bin_variances_ep1 = compute_uce(probs_expert1, targets, n_bins)
    uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2, bin_variances_ep2 = compute_uce(probs_expert2, targets, n_bins)

    # Compute uncertainties for both experts
    _, nattrs = probs_expert1.size()
    nattrs = torch.tensor(nattrs)
    uncertainties_expert1 = (1/torch.log(nattrs))*(-torch.sum(probs_expert1 * torch.log(probs_expert1 + 1e-12), dim=1))
    uncertainties_expert2 = (1/torch.log(nattrs))*(-torch.sum(probs_expert2 * torch.log(probs_expert2 + 1e-12), dim=1))
    # Find error rates for both experts
    error_rates_expert1 = find_error_rates(uncertainties_expert1, bin_uncertainties_expert1, bin_errors_expert1)
    error_rates_expert2 = find_error_rates(uncertainties_expert2, bin_uncertainties_expert2, bin_errors_expert2)
    # Choose the expert with lower error rate for each sample
    chosen_expert = (error_rates_expert1 < error_rates_expert2)

    # Get the predictions from both experts
    preds_expert1 = torch.argmax(probs_expert1, dim=1)
    preds_expert2 = torch.argmax(probs_expert2, dim=1)

    # Choose the final prediction based on the chosen expert
    final_predictions = torch.where(chosen_expert, preds_expert1, preds_expert2)

    return final_predictions

def cal_all_stats(S_logits_expert1,all_logits,all_attr_gt,all_pair_gt,pairs,yfs,stats):

    # 找到all_obj_gt中每個元素對於在pairs當中的位置
    obj_idxs = [torch.where(pairs[:, 1] == obj)[0] for obj in all_obj_gt]
    # 並將對應的位置將S_all_logits>S_logit_attr_ours
    attr_exp2 = torch.Tensor()
    for i,obj_idx in zip(range(len(obj_idxs)),obj_idxs):
        attr_exp2 =  torch.cat([attr_exp2 ,all_logits[i,obj_idx].unsqueeze(0)], dim=0)
    S_attr_exp2 = F.softmax(attr_exp2, dim=1)
    
    S_all_logits = F.softmax(all_logits, dim=1)
    #計算準確度
    table_pred = choose_best_expert(S_logits_expert1, S_all_logits, all_attr_gt,all_pair_gt,pairs)
    table_acc= accuracy(table_pred,all_attr_gt)

    table_expert = choose_best_expert_ex(S_logits_expert1, S_attr_exp2, all_attr_gt)
    table_acc_ex= accuracy(table_expert,all_attr_gt)

    preds_expert1 = torch.argmax(S_logits_expert1, dim=1)
    expert1_acc= accuracy(preds_expert1,all_attr_gt)
    

    preds_expert2_pairs = torch.argmax(S_all_logits, dim=1)
    preds_expert2 = pairs[preds_expert2_pairs][:, 0].cpu()    
    expert2_acc=accuracy(preds_expert2,all_attr_gt)

    preds_expert2_ex = torch.argmax(S_attr_exp2, dim=1)
    expert2_acc_ex= accuracy(preds_expert2_ex,all_attr_gt)

    
    #計算UCE
    uce_and_bin_values_all_logits_attr = [compute_uce(S_logits_expert1, all_attr_gt)]
    uce_ep1=uce_and_bin_values_all_logits_attr[0][0]    
    #計算ep2 UCE    
    uce_and_bin_values = [compute_uce(S_all_logits, all_pair_gt)]
    uce_ep2=uce_and_bin_values[0][0]   
    #計算ep2_ex 根據物體的資訊計算UCE
    uce_and_bin_values = [compute_uce(S_attr_exp2, all_attr_gt)]
    uce_ep2_ex=uce_and_bin_values[0][0]   
    
    stats['attr_acc_uce_ep1'+yfs] = uce_ep1
    stats['attr_acc_uce_ep2'+yfs] = uce_ep2
    stats['attr_acc_uce_ep2_ex'+yfs] = uce_ep2_ex
    
    stats['attr_acc_table_acc'+yfs] = table_acc
    stats['attr_acc_table_acc_ex'+yfs] = table_acc_ex
    
    stats['attr_acc_ep1_cor'+yfs] = expert1_acc
    stats['attr_acc_ep2_cor'+yfs] = expert2_acc    
    stats['attr_acc_ep2_cor_ex'+yfs] = expert2_acc_ex 

def predict_logits(model, dataset, config):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        text_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    # print(text_rep.shape)
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False)
    all_logits = torch.Tensor()
    loss = 0
    all_logits_attr_org = torch.Tensor()
    all_logits_attr_our = torch.Tensor()
    all_fv_proj_idx =construct_all_fv_proj_idx(dataset,pairs)
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            batch_img = data[0].cuda()
#             predict = model(batch_img, pairs)
#             predict = model(batch_img, pairs,True,data[2][0],all_fv_proj_idx=all_fv_proj_idx) ##這個是一張一張算
            predict = model(batch_img, pairs,True,data[2],all_fv_proj_idx=all_fv_proj_idx)  ###更改批量計算
            logits = predict[0]
            loss += loss_calu(predict, data, config)
            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            #------------------------------
            logits_attr_org,logits_attr_our= predict[1].cpu(),predict[4].cpu()
            all_logits_attr_org = torch.cat([all_logits_attr_org, logits_attr_org], dim=0)
            all_logits_attr_our = torch.cat([all_logits_attr_our, logits_attr_our], dim=0)
            logits_attrs_list = [all_logits_attr_org,all_logits_attr_our]
            #------------------------------
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)


    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt,logits_attrs_list,loss / len(dataloader)


def threshold_with_feasibility(
        logits,
        seen_mask,
        threshold=None,
        feasiblity=None):
    """Function to remove infeasible compositions.

    Args:
        logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        seen_mask (torch.tensor): the seen mask with binary
        threshold (float, optional): the threshold value.
            Defaults to None.
        feasiblity (torch.Tensor, optional): the feasibility.
            Defaults to None.

    Returns:
        torch.Tensor: the logits after filtering out the
            infeasible compositions.
    """
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasiblity >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score


def test(
        test_dataset,
        logits_attrs_list, ###新增
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    """Function computes accuracy on the validation and
    test dataset.

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    ###----------------新增預測att--------------
    attr2idx = test_dataset.attr2idx
    obj2idx = test_dataset.obj2idx
    pairs_dataset = test_dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()

    predictions = {
        pair_name: all_logits[:, i]
        for i, pair_name in enumerate(test_dataset.pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=1e3, topk=1
    )

    #-----------------------------------------
    all_logits_attr = torch.zeros(all_logits.size(0), len(test_dataset.attrs))
    for i, (attr_idx, obj_idx) in enumerate(pairs):
        all_logits_attr[:, attr_idx] += all_logits[:, i]
    all_logits_attr /= len(test_dataset.objs)
    att_ped_org = torch.argmax(all_logits_attr,dim=1)
    S_logits_expert1 = F.softmax(all_logits_attr, dim=1)
    S_all_logits = F.softmax(all_logits, dim=1)
    #計算ep1 ECE
    ece_and_bin_values_all_logits_attr = [compute_ece(S_logits_expert1, all_attr_gt)]
    for i, (ece_value, bin_confidences, bin_accuracies) in enumerate(ece_and_bin_values_all_logits_attr):
        plot_dot_reliability_diagram(ece_value, bin_confidences, bin_accuracies, 0, test_dataset,config)
    ece_ep1=ece_and_bin_values_all_logits_attr[0][0]
    #計算ep1 UCE
    uce_and_bin_values_all_logits_attr = [compute_uce(S_logits_expert1, all_attr_gt)]
    for i, (uce_value, bin_uncertainties, bin_errors, prop_in_bin_values, bin_n_samples, bin_variances) in enumerate(uce_and_bin_values_all_logits_attr):
        plot_dot_UCE_diagram(uce_value, bin_uncertainties, bin_errors, 0, test_dataset, config, prop_in_bin_values, bin_n_samples, bin_variances)
    uce_ep1=uce_and_bin_values_all_logits_attr[0][0]

    S_logits_expert2 = F.softmax(logits_attrs_list[1], dim=1)
#     #計算ep2 ECE
#     ece_and_bin_values = [compute_ece(S_logits_expert2, all_attr_gt)]
#     for i, (ece_value, bin_confidences, bin_accuracies) in enumerate(ece_and_bin_values):
#         plot_dot_reliability_diagram(ece_value, bin_confidences, bin_accuracies, 1,test_dataset,config)
#     ece_ep2=ece_and_bin_values[0][0]
#     #計算ep2 UCE    
#     uce_and_bin_values = [compute_uce(S_logits_expert2, all_attr_gt)]
#     for i, (uce_value, bin_uncertainties, bin_errors, prop_in_bin_values, bin_n_samples, bin_variances) in enumerate(uce_and_bin_values):
#         plot_dot_UCE_diagram(uce_value, bin_uncertainties, bin_errors, 1, test_dataset, config, prop_in_bin_values, bin_n_samples, bin_variances)
#     uce_ep2=uce_and_bin_values[0][0]   

    #計算pair pred uce
    uce_and_bin_values = [compute_uce(S_all_logits, all_pair_gt)]
    for i, (uce_value, bin_uncertainties, bin_errors, prop_in_bin_values, bin_n_samples, bin_variances) in enumerate(uce_and_bin_values):
        plot_dot_UCE_diagram(uce_value, bin_uncertainties, bin_errors, 1, test_dataset, config, prop_in_bin_values, bin_n_samples, bin_variances)
    uce_ep2=uce_and_bin_values[0][0]      

    #5/1新增 使用UCE和softmax/2計算final pred
    
    table_pred = choose_best_expert(S_logits_expert1, S_all_logits, all_attr_gt,all_pair_gt,pairs)
    table_acc= accuracy(table_pred,all_attr_gt)
#     combined_softmax = S_logits_expert1 + S_logits_expert2
#     # 3. 將相加後的結果除以 2
#     averaged_softmax = combined_softmax / 2
#     # 4. 找出概率最大的類別作為最終預測結果
#     softmax_pred = torch.argmax(averaged_softmax, dim=1)
#     softmax_acc =accuracy(softmax_pred,all_attr_gt)            
    
    attr_acc = float(torch.mean(
        (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))   ##先計算原本的attr 和obj acc
    obj_acc = float(torch.mean(
        (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))
        
    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=1,
    )

    attr_acc_vote,attr_acc_ep1_cor,_ = cal_final_expert(all_logits_attr,logits_attrs_list[1],all_attr_gt)


    
    
    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc

###----------------新增預測att--------------
    stats['attr_acc_vote'] = attr_acc_vote
    stats['attr_acc_uce_ep1'] = uce_ep1
    stats['attr_acc_uce_ep2'] = uce_ep2
    stats['attr_acc_table_acc'] = table_acc
#     stats['attr_acc_softmax_acc'] = softmax_acc
###----------------新增預測att--------------
    stats['attr_acc_ep1_cor'] = attr_acc_ep1_cor
    stats['attr_acc_ep2_cor'] = attr_acc
    return stats
def cal_final_expert(logits_expert1,logits_expert2,ground_truth):
    # Create a softmax layer
    softmax = nn.Softmax(dim=1)

    # Calculate the softmax output for EXPERT1 and EXPERT2
    softmax_output_expert1 = softmax(logits_expert1)
    softmax_output_expert2 = softmax(logits_expert2)

# The softmax_output_expert1 and softmax_output_expert2 variables now contain the softmax probabilities for EXPERT1 and EXPERT2 respectively
    # Calculate confidences
    confidence_expert1 = torch.max(softmax_output_expert1, dim=1)[0]
    confidence_expert2 = torch.max(softmax_output_expert2, dim=1)[0]

    # Get the predictions from each expert
    predictions_expert1 = torch.argmax(softmax_output_expert1, dim=1)
    predictions_expert2 = torch.argmax(softmax_output_expert2, dim=1)

    # Compare confidences and choose the predictions from the expert with the highest confidence
    chosen_predictions = torch.where(confidence_expert1 > confidence_expert2, predictions_expert1, predictions_expert2)

    # Calculate the accuracy for each expert
    accuracy_expert1 = (predictions_expert1 == ground_truth).float()
    accuracy_expert2 = (predictions_expert2 == ground_truth).float()

    # Choose the accuracy corresponding to the highest confidence
    chosen_accuracies = torch.where(confidence_expert1 > confidence_expert2, accuracy_expert1, accuracy_expert2)

    # Calculate the overall accuracy
    overall_accuracy = chosen_accuracies.mean().item()


    return overall_accuracy,accuracy_expert1.mean().item(),accuracy_expert2.mean().item()

def test__yfs(
        test_dataset,
        stats,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    
    attr2idx = test_dataset.attr2idx
    obj2idx = test_dataset.obj2idx
    pairs_dataset = test_dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    
    all_logits_attr = torch.zeros(all_logits.size(0), len(test_dataset.attrs))
    for i, (attr_idx, obj_idx) in enumerate(pairs):
        all_logits_attr[:, attr_idx] += all_logits[:, i]
    all_logits_attr /= len(test_dataset.objs)
    att_ped_org = torch.argmax(all_logits_attr,dim=1)
    S_logits_expert1 = F.softmax(all_logits_attr, dim=1)
    S_all_logits = F.softmax(all_logits, dim=1)

    yfs = "_yfs"
    cal_all_stats(S_logits_expert1,all_logits,all_attr_gt,all_pair_gt,pairs,yfs,stats)
    return stats     


if __name__ == "__main__":
    config = parser.parse_args()
    load_args(YML_PATH[config.dataset], config)

    # set the seed value
    
    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")


    dataset_path = config.dataset_path

    print('loading validation dataset')
    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural',
                                     open_world=config.open_world)

    print('loading test dataset')
    test_dataset = CompositionDataset(dataset_path,
                                      phase='test',
                                      split='compositional-split-natural',
                                      open_world=config.open_world)

    allattrs = val_dataset.attrs
    allobj = val_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    
    model = DFSP(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(config.load_model))

    print('evaluating on the validation set')
    if config.open_world and config.threshold is None:
        evaluator = Evaluator(val_dataset, model=None)
        feasibility_path = os.path.join(
            DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(
            feasibility_path,
            map_location='cpu')['feasibility']
        seen_mask = val_dataset.seen_mask.to('cpu')
        min_feasibility = (unseen_scores + seen_mask * 10.).min()
        max_feasibility = (unseen_scores - seen_mask * 10.).max()
        thresholds = np.linspace(
            min_feasibility,
            max_feasibility,
            num=config.threshold_trials)
        best_auc = 0.
        best_th = -10
        val_stats = None
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt,logits_attrs_list ,loss_avg = predict_logits(
                model, val_dataset, device, config)
            for th in thresholds:
                temp_logits = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
                results = test(
                    val_dataset,
                    logits_attrs_list,   ###新增
                    evaluator,
                    temp_logits,
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config
                )
                auc = results['AUC']
                if auc > best_auc:
                    best_auc = auc
                    best_th = th
                    print('New best AUC', best_auc)
                    print('Threshold', best_th)
                    val_stats = copy.deepcopy(results)
    else:
        best_th = config.threshold
        evaluator = Evaluator(val_dataset, model=None)
        feasibility_path = os.path.join(
            DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(
            feasibility_path,
            map_location='cpu')['feasibility']
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt,logits_attrs_list ,loss_avg = predict_logits(
                model, val_dataset, config)
            if config.open_world:
                print('using threshold: ', best_th)
                all_logits_ = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
            results = test(
                val_dataset,
                logits_attrs_list,   ###新增
                evaluator,
                all_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config
            )
            
            results =test__yfs(test_dataset,
                    results,
                    evaluator,
                    all_logits_,
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config)
        val_stats = copy.deepcopy(results)
        result = ""
        for key in val_stats:
             # 將 PyTorch 張量轉換為標量
            val = val_stats[key].item() if isinstance(val_stats[key], torch.Tensor) else val_stats[key]
            result = result + key + "  " + str(round(val, 4)) + "| "
        print(result)

    print('evaluating on the test set')
    with torch.no_grad():
        evaluator = Evaluator(test_dataset, model=None)
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, logits_attrs_list,loss_avg = predict_logits(
            model, test_dataset, config)
        if config.open_world and best_th is not None:
            print('using threshold: ', best_th)
            all_logits_ = threshold_with_feasibility(
                all_logits,
                test_dataset.seen_mask,
                threshold=best_th,
                feasiblity=unseen_scores)
        test_stats = test(
            test_dataset,
            logits_attrs_list,   ###新增
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
        
        test_stats =test__yfs(test_dataset,
                test_stats,
                evaluator,
                all_logits_,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config)

        result = ""
        for key in test_stats:
             # 將 PyTorch 張量轉換為標量
            test = test_stats[key].item() if isinstance(test_stats[key], torch.Tensor) else test_stats[key]
            result = result + key + "  " + str(round(test, 4)) + "| "
        print(result)
        
    results = {
        'val': val_stats,
        'test': test_stats,
    }

    if best_th is not None:
        results['best_threshold'] = best_th

    if config.open_world:
        result_path = config.load_model[:-2] + "open.calibrated.json"
    else:
        result_path = config.load_model[:-2] + "closed.json"

    with open(result_path, 'w+') as fp:
        json.dump(results, fp, default=lambda o: o.item() if isinstance(o, torch.Tensor) else o)

    print("done!")
