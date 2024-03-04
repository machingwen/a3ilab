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
from typing import List, Optional
from utils import *
from loss import loss_calu
from parameters import parser, YML_PATH
from dataset import CompositionDataset
from model.dfsp import DFSP
import yaml
from sklearn.metrics import confusion_matrix 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from collections import Counter
from typing import List

import csv
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
def choose_best_expert(probs_expert1, probs_expert2, targets,targets_pairs,pairs ,test_dataset,val_uce_list_ep1,val_uce_list_ep2,weight_ep1,weight_ep2,n_bins=10):


#     uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1, bin_variances_ep1 = compute_uce(probs_expert1, targets, n_bins)
#     uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2, bin_variances_ep2 = compute_uce(probs_expert2, targets_pairs, n_bins)
    uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1,bin_variances_ep1 = val_uce_list_ep1[0],val_uce_list_ep1[1],val_uce_list_ep1[2],val_uce_list_ep1[3],val_uce_list_ep1[4],val_uce_list_ep1[5]
    uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2,bin_variances_ep2 = val_uce_list_ep2[0],val_uce_list_ep2[1],val_uce_list_ep2[2],val_uce_list_ep2[3],val_uce_list_ep2[4],val_uce_list_ep2[5]

    # Compute uncertainties for both experts
    _, nattrs = probs_expert1.size()
    _, nattrs = probs_expert2.size()
    nattrs = torch.tensor(nattrs)
    nattrs = torch.tensor(nattrs)
    uncertainties_expert1 = (1/torch.log(nattrs))*(-torch.sum(probs_expert1 * torch.log(probs_expert1 + 1e-12), dim=1))
    uncertainties_expert2 = (1/torch.log(nattrs))*(-torch.sum(probs_expert2 * torch.log(probs_expert2 + 1e-12), dim=1))
    # Find error rates for both experts
    error_rates_expert1 = find_error_rates(uncertainties_expert1, bin_uncertainties_expert1, bin_errors_expert1)
    error_rates_expert2 = find_error_rates(uncertainties_expert2, bin_uncertainties_expert2, bin_errors_expert2)
    # Choose the expert with lower error rate for each sample
    
    error_rates_expert1 = (error_rates_expert1/weight_ep1)
    error_rates_expert2 = (error_rates_expert2/weight_ep2)
    
    
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
            found_= False
            if bin_uncertainties[0] is not None: 
                if 0 <=uncertainty< bin_uncertainties[0]:
                    error_rates.append(bin_errors[0])
                    found_= True
                else:    
                    for bin_error in reversed(bin_errors):
                        if bin_error is not None:
                            error_rates.append(bin_error)
                            found_= True
                            break
            else:
                if bin_uncertainties[1] is not None: 
                    error_rates.append(bin_errors[1])
                    found_= True
                else:
                    error_rates.append(bin_errors[2])
                    found_= True
                    
            if not found_:
                if bin_uncertainties[4] is not None and 0 <=uncertainty< bin_uncertainties[4]: 
                    error_rates.append(bin_errors[4])
                    found_= True
            if not found_:
                if bin_uncertainties[8] is not None and bin_uncertainties[8] <=uncertainty< 1: 
                    error_rates.append(bin_errors[8])
                    found_= True
                

    return torch.tensor(error_rates)

def accuracy(y_true, y_pred):

    # 計算正確預測的數量
    correct_predictions = torch.sum(y_true == y_pred)

    # 計算準確度
    accuracy = correct_predictions.item() / y_true.size(0)

    return accuracy
def choose_best_expert_ex(probs_expert1, probs_expert2, targets,test_dataset,val_uce_list_ep1,val_uce_list_ep2,weight_ep1,weight_ep2, n_bins=10):
    # Compute UCE and bin values for both experts

#     uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1, bin_variances_ep1 = compute_uce(probs_expert1, targets, n_bins)
#     uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2, bin_variances_ep2 = compute_uce(probs_expert2, targets, n_bins)

    uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1,bin_variances_ep1 = val_uce_list_ep1[0],val_uce_list_ep1[1],val_uce_list_ep1[2],val_uce_list_ep1[3],val_uce_list_ep1[4],val_uce_list_ep1[5]
    uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2,bin_variances_ep2 = val_uce_list_ep2[0],val_uce_list_ep2[1],val_uce_list_ep2[2],val_uce_list_ep2[3],val_uce_list_ep2[4],val_uce_list_ep2[5]

    # Compute uncertainties for both experts
    _, nattrs = probs_expert1.size()
    nattrs = torch.tensor(nattrs)
    uncertainties_expert1 = (1/torch.log(nattrs))*(-torch.sum(probs_expert1 * torch.log(probs_expert1 + 1e-12), dim=1))
    uncertainties_expert2 = (1/torch.log(nattrs))*(-torch.sum(probs_expert2 * torch.log(probs_expert2 + 1e-12), dim=1))
    # Find error rates for both experts
    error_rates_expert1 = find_error_rates(uncertainties_expert1, bin_uncertainties_expert1, bin_errors_expert1)
    error_rates_expert2 = find_error_rates(uncertainties_expert2, bin_uncertainties_expert2, bin_errors_expert2)
    # Choose the expert with lower error rate for each sample
    
    error_rates_expert1 = (error_rates_expert1/weight_ep1)
    error_rates_expert2 = (error_rates_expert2/weight_ep2)
    
    chosen_expert = (error_rates_expert1 < error_rates_expert2)

    # Get the predictions from both experts
    preds_expert1 = torch.argmax(probs_expert1, dim=1)
    preds_expert2 = torch.argmax(probs_expert2, dim=1)

    # Choose the final prediction based on the chosen expert
    final_predictions = torch.where(chosen_expert, preds_expert1, preds_expert2)

    return final_predictions


def choose_best_three_expert(probs_expert1,probs_expert2,probs_expert3,pairs, targets,targets_pairs,test_dataset,val_uce_list_ep1,val_uce_list_ep2,val_uce_list_ep3,weight_ep1,weight_ep2,weight_ep3, n_bins=10):


    uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1,bin_variances_ep1 = val_uce_list_ep1[0],val_uce_list_ep1[1],val_uce_list_ep1[2],val_uce_list_ep1[3],val_uce_list_ep1[4],val_uce_list_ep1[5]
    uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2,bin_variances_ep2 = val_uce_list_ep2[0],val_uce_list_ep2[1],val_uce_list_ep2[2],val_uce_list_ep2[3],val_uce_list_ep2[4],val_uce_list_ep2[5]
    uce_expert3, bin_uncertainties_expert3, bin_errors_expert3, prop_in_bin_values_expert3,bin_n_samples_ep3,bin_variances_ep3 = val_uce_list_ep3[0],val_uce_list_ep3[1],val_uce_list_ep3[2],val_uce_list_ep3[3],val_uce_list_ep3[4],val_uce_list_ep3[5]



    # Compute uncertainties for both experts
    _, nattrs = probs_expert1.size()

    nattrs = torch.tensor(nattrs)

    uncertainties_expert1 = (1/torch.log(nattrs))*(-torch.sum(probs_expert1 * torch.log(probs_expert1 + 1e-12), dim=1))
    uncertainties_expert2 = (1/torch.log(nattrs))*(-torch.sum(probs_expert2 * torch.log(probs_expert2 + 1e-12), dim=1))
    uncertainties_expert3 = (1/torch.log(nattrs))*(-torch.sum(probs_expert3 * torch.log(probs_expert3 + 1e-12), dim=1))

    # Find error rates for both experts
    error_rates_expert1 = find_error_rates(uncertainties_expert1, bin_uncertainties_expert1, bin_errors_expert1)
    error_rates_expert2 = find_error_rates(uncertainties_expert2, bin_uncertainties_expert2, bin_errors_expert2)
    error_rates_expert3 = find_error_rates(uncertainties_expert3, bin_uncertainties_expert3, bin_errors_expert3)
    # Choose the expert with lower error rate for each sample
    error_rates_expert1 = (error_rates_expert1/weight_ep1)
    error_rates_expert2 = (error_rates_expert2/weight_ep2)
    error_rates_expert3 = (error_rates_expert3/weight_ep3)
    
    
    # Get the predictions from both experts
    preds_expert1 = torch.argmax(probs_expert1, dim=1)
    preds_expert2 = torch.argmax(probs_expert2, dim=1)
    preds_expert3 = torch.argmax(probs_expert3, dim=1)

    # 將三個錯誤率堆疊成一個張量
    error_rates = torch.stack([error_rates_expert1, error_rates_expert2, error_rates_expert3])

    # 找出最小錯誤率的索引
    _, min_error_rate_indices = torch.min(error_rates, dim=0)

    # 根據最小錯誤率的索引選擇最終的預測
    final_predictions = torch.where(min_error_rate_indices == 0, preds_expert1, 
                                    torch.where(min_error_rate_indices == 1, preds_expert2, preds_expert3))
    return final_predictions

#-----------------------10/25-----------------------
std_deviation_per_position = None
error_rates = None
POE_pred = None
final_predictions = None
def choose_best_three_expert_new(probs_expert1,probs_expert2,probs_expert3 ,targets,val_uce_list_ep1,val_uce_list_ep2,val_uce_list_ep3,phase, n_bins=10):
    global std_deviation_per_position
    global error_rates
    global threshold_
    global POE_pred
    global final_predictions
    global global_thresholds
#     global global_a
    uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1,bin_variances_ep1 = val_uce_list_ep1[0],val_uce_list_ep1[1],val_uce_list_ep1[2],val_uce_list_ep1[3],val_uce_list_ep1[4],val_uce_list_ep1[5]
    uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2,bin_variances_ep2 = val_uce_list_ep2[0],val_uce_list_ep2[1],val_uce_list_ep2[2],val_uce_list_ep2[3],val_uce_list_ep2[4],val_uce_list_ep2[5]
    uce_expert3, bin_uncertainties_expert3, bin_errors_expert3, prop_in_bin_values_expert3,bin_n_samples_ep3,bin_variances_ep3 = val_uce_list_ep3[0],val_uce_list_ep3[1],val_uce_list_ep3[2],val_uce_list_ep3[3],val_uce_list_ep3[4],val_uce_list_ep3[5]



    # Compute uncertainties for both experts
    _, nattrs = probs_expert1.size()
    nattrs = torch.tensor(nattrs)
    uncertainties_expert1 = (1/torch.log(nattrs))*(-torch.sum(probs_expert1 * torch.log(probs_expert1 + 1e-12), dim=1))
    uncertainties_expert2 = (1/torch.log(nattrs))*(-torch.sum(probs_expert2 * torch.log(probs_expert2 + 1e-12), dim=1))
    uncertainties_expert3 = (1/torch.log(nattrs))*(-torch.sum(probs_expert3 * torch.log(probs_expert3 + 1e-12), dim=1))

    # Find error rates for both experts
    error_rates_expert1 = find_error_rates(uncertainties_expert1, bin_uncertainties_expert1, bin_errors_expert1)
    error_rates_expert2 = find_error_rates(uncertainties_expert2, bin_uncertainties_expert2, bin_errors_expert2)
    error_rates_expert3 = find_error_rates(uncertainties_expert3, bin_uncertainties_expert3, bin_errors_expert3)
    # Choose the expert with lower error rate for each sample

    # Get the predictions from both experts
    preds_expert1 = torch.argmax(probs_expert1, dim=1)
    preds_expert2 = torch.argmax(probs_expert2, dim=1)
    preds_expert3 = torch.argmax(probs_expert3, dim=1)

    # 將三個錯誤率堆疊成一個張量
    error_rates = torch.stack([error_rates_expert1, error_rates_expert2, error_rates_expert3])
    
    std_deviation_per_position = torch.std(error_rates, dim=0)
    mean_value = torch.mean(std_deviation_per_position)
#     print("mean: ",mean_value)
#     print(std_deviation_per_position)
    # 找出最小錯誤率的索引
    _, min_error_rate_indices = torch.min(error_rates, dim=0)
    
    POE_probs_ = torch.stack([probs_expert1, probs_expert2,probs_expert3])
    POE_probs = product_of_experts(POE_probs_)
    POE_pred = np.argmax(POE_probs, axis=1)
    
    POE_pred = torch.tensor(POE_pred)  # Convert numpy array to torch tensor

    SOE_probs_ = (probs_expert1+probs_expert2+probs_expert3)/3
    SOE_pred = np.argmax(SOE_probs_, axis=1)
    SOE_acc_ =accuracy(SOE_pred,targets)
    # 根據最小錯誤率的索引選擇最終的預測
    final_predictions = torch.where(min_error_rate_indices == 0, preds_expert1,
                                      torch.where(min_error_rate_indices == 1, preds_expert2, preds_expert3))

    initial_predictions_probs = torch.where(min_error_rate_indices.unsqueeze(-1) == 0, probs_expert1,
                                           torch.where(min_error_rate_indices.unsqueeze(-1) == 1, probs_expert2, probs_expert3))
    
    POE_probs_ = torch.stack([POE_probs,SOE_probs_,initial_predictions_probs])
    POE_initial_predictions_probs = product_of_experts(POE_probs_)
    SOE_initial_predictions_probs = (SOE_probs_+POE_probs+initial_predictions_probs)/3
    
    POE_final_predictions = np.argmax(POE_initial_predictions_probs, axis=1)
    
    SOE_final_predictions = np.argmax(SOE_initial_predictions_probs, axis=1)
    
    POE_acc =accuracy(POE_final_predictions,targets)
    SOE_acc =accuracy(SOE_final_predictions,targets)
    SPE_acc =accuracy(final_predictions,targets)
    
#     print("POE_SPE_acc: ",POE_acc,"SOE_SPE_acc: ",SOE_acc)
      


    def cal_std_acc(error_rates, SOE_pred, final_predictions, targets):
        global global_a
        # 计算SOE_pred和final_predictions的准确度
        SOE_acc = accuracy(SOE_pred, targets)
        final_predictions_acc = accuracy(final_predictions, targets)
#         print('SOE_acc', SOE_acc, 'final_predictions_acc', final_predictions_acc)

        if SOE_acc > final_predictions_acc:
            a_values = np.arange(1, 0, -0.01)
        else:
            a_values = np.arange(0, 1, 0.01)

        for a in a_values:
            # 1. 计算每个样本的方差
            variances_per_sample = torch.var(error_rates, dim=0)

            # 2. 计算第25百分位数，并使用a来控制这个值
            threshold_value = torch.quantile(variances_per_sample, a) 

            # 3. 根据每个样本的方差选择SOE_pred或final_predictions
            mask = variances_per_sample > threshold_value
            chosen_predictions = torch.where(mask, final_predictions, SOE_pred)

            std_acc = accuracy(chosen_predictions, targets)

            if len(global_a) < 10:
                if SOE_acc > final_predictions_acc and std_acc > SOE_acc:
#                     print(f"a: {a:.2f}, threshold_value: {threshold_value:.10f}, mask: {mask.sum()}, std_acc: {std_acc}")
                    global_a.append(a)
                elif SOE_acc < final_predictions_acc and std_acc > final_predictions_acc:
#                     print(f"a: {a:.2f}, threshold_value: {threshold_value:.10f}, mask: {mask.sum()}, std_acc: {std_acc}")
                    global_a.append(a)



    def evaluate_with_thresholds(error_rates, SOE_pred, final_predictions, targets, global_a):

        # Compute accuracy based on a given a_value
        def compute_acc_with_a(a_value):
            variances_per_sample = torch.var(error_rates, dim=0)
            threshold_value = torch.quantile(variances_per_sample, a_value)
            mask = variances_per_sample > threshold_value
            chosen_predictions = torch.where(mask, final_predictions, SOE_pred)
            return accuracy(chosen_predictions, targets)
        acc_list = []
        # Evaluate using each a_value in global_a
        for a_value in global_a:
            acc_ = compute_acc_with_a(a_value)
            acc_list.append(acc_)
#             print(f"Using a_value: {a_value:.2f}, ERV-SoP Accuracy: {acc:.4f}")

        print("acc_list:---------------- ",acc_list)

        if acc_list == []:
            ERV_SoP_acc = 0.0
        else:
            ERV_SoP_acc = acc_list[0]

        return ERV_SoP_acc
#         print("Using the Threshold",a_value,"MOM acc: ",MOM_ACC)
        


    if phase == 'val':
        global global_a
        print('------------------global_a------val---------')
        print(global_a)
        cal_std_acc(error_rates, SOE_pred, final_predictions, targets)
        ERV_SoP_acc = 0.0
#         import ipdb 
#         ipdb.set_trace()
    else:
        ERV_SoP_acc = evaluate_with_thresholds(error_rates, SOE_pred, final_predictions, targets, global_a)
        cal_std_acc(error_rates, SOE_pred, final_predictions, targets)
    
#     print('global_a:', global_a)
    if ERV_SoP_acc == []:
        ERV_SoP_acc = 0.0

    return SOE_final_predictions,ERV_SoP_acc






def calculate_weighted_multiclass_fdr_for(predictions, ground_truth):
    # Get the number of classes
    num_classes = np.unique(np.concatenate((predictions, ground_truth))).shape[0]

    # Create the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    FDR = []
    FOR = []
    weights = []

    # For each class
    for i in range(num_classes):
        FP = cm[:,i].sum() - cm[i,i]
        FN = cm[i,:].sum() - cm[i,i]
        TP = cm[i,i]
        TN = cm.sum() - FP - FN - TP

        FDR_class_i = FP / (FP + TP) if (FP + TP) > 0 else 0
        FOR_class_i = FN / (FN + TN) if (FN + TN) > 0 else 0

        # Calculate the weight for this class
        weight_class_i = np.sum(ground_truth == i)
#         print(FDR_class_i,weight_class_i)
        FDR.append(FDR_class_i * weight_class_i)
        FOR.append(FOR_class_i * weight_class_i)
        weights.append(weight_class_i)

    return np.sum(FDR) / np.sum(weights), np.sum(FOR) / np.sum(weights)

# ## 6/4新增加權投票

def voting(preds_expert1: List[int], preds_expert2: List[int], preds_expert3: List[int], default_expert: int) -> torch.Tensor:
    assert default_expert in [1, 2, 3], "Default expert must be either 1, 2, or 3"
    
    final_preds = []
    for p1, p2, p3 in zip(preds_expert1, preds_expert2, preds_expert3):
        vote_counts = Counter([p1, p2, p3])
        max_vote_count = max(vote_counts.values())
        most_common = [k for k, v in vote_counts.items() if v == max_vote_count]
        
        if len(most_common) > 1:
            if default_expert == 1:
                final_preds.append(p1)
            elif default_expert == 2:
                final_preds.append(p2)
            else:  # default_expert == 3
                final_preds.append(p3)
        else:
            final_preds.append(most_common[0])
    
    final_preds = torch.tensor(final_preds, dtype=torch.int64)  # or your desired data type
    return final_preds
def weighted_voting(preds_expert1: List[int], preds_expert2: List[int], preds_expert3: List[int], weights: List[float], default_expert: int) -> torch.Tensor:
    assert default_expert in [1, 2, 3], "Default expert must be either 1, 2, or 3"
    
    final_preds = []
    for p1, p2, p3 in zip(preds_expert1, preds_expert2, preds_expert3):
        weighted_vote_counts = Counter()
        for pred, weight in zip([p1, p2, p3], weights):
            weighted_vote_counts[pred] += weight
        
        max_vote_count = max(weighted_vote_counts.values())
        most_common = [k for k, v in weighted_vote_counts.items() if v == max_vote_count]
        
        if len(most_common) > 1:
            if default_expert == 1:
                final_preds.append(p1)
            elif default_expert == 2:
                final_preds.append(p2)
            else:  # default_expert == 3
                final_preds.append(p3)
        else:
            final_preds.append(most_common[0])
    
    final_preds = torch.tensor(final_preds, dtype=torch.int64)  # or your desired data type
    return final_preds
def product_of_experts(predictions):
    # Multiply predictions together
    product = torch.prod(predictions, dim=0)
    
    # Normalize result
    product /= torch.sum(product)
    
    return product

def weighted_voting_(prob_expert1: List[float], prob_expert2: List[float], weights: List[float], default_expert: int) -> torch.Tensor:
    assert default_expert in [1, 2], "Default expert must be either 1 or 2"

    final_probs = []
    for p1, p2 in zip(prob_expert1, prob_expert2):
        weighted_probs = Counter()
        for prob, weight in zip([p1, p2], weights):
            weighted_probs[prob] += weight

        max_prob_count = max(weighted_probs.values())
        most_common = [k for k, v in weighted_probs.items() if v == max_prob_count]

        if len(most_common) > 1:
            if default_expert == 1:
                final_probs.append(p1)
            else:  # default_expert == 2
                final_probs.append(p2)
        else:
            final_probs.append(most_common[0])

    final_probs = torch.tensor(final_probs, dtype=torch.float32)  # or your desired data type
    return final_probs

def voting_(prob_expert1: Optional[List[float]], prob_expert2: Optional[List[float]], default_expert: int) -> torch.Tensor:
    assert default_expert in [1, 2], "Default expert must be either 1 or 2"

    final_probs = []
    for p1, p2 in zip(prob_expert1, prob_expert2):
        prob_counts = Counter([p for p in [p1, p2] if p is not None])
        if not prob_counts:  # All predictions are None
            final_probs.append(None)
            continue

        max_prob_count = max(prob_counts.values())
        most_common = [k for k, v in prob_counts.items() if v == max_prob_count]

        if len(most_common) > 1:
            if default_expert == 1:
                final_probs.append(p1)
            else:  # default_expert == 2
                final_probs.append(p2)
        else:
            final_probs.append(most_common[0])

    final_probs = torch.tensor(final_probs, dtype=torch.float32)  # or your desired data type
    return final_probs


###儲存檔案
def save_tensors(phase, train_look_up_table, feat_path_root,
                 S_attr_exp1, S_attr_exp2, S_attr_exp3, all_attr_gt):
    """
    Save tensors based on given naming conditions.

    Args:
    - phase (str): Either 'val' or 'test'.
    - train_look_up_table (str): Lookup table type, e.g. "methodA".
    - feat_path_root (str): The root directory to save the tensor, e.g. './feat/'.
    - S_attr_exp1, S_attr_exp2, S_attr_exp3, all_attr_gt: The tensors to be saved.

    Returns:
    - None
    """
    
    # Construct the file names based on given parameters
    file_names = {
        "S_attr_exp1": f"S_attr_exp1_{phase}_{train_look_up_table}.pt",
        "S_attr_exp2": f"S_attr_exp2_{phase}_{train_look_up_table}.pt",
        "S_attr_exp3": f"S_attr_exp3_{phase}_{train_look_up_table}.pt",
        "all_attr_gt": f"all_attr_gt_{phase}_{train_look_up_table}.pt"
    }
    
    tensors = {
        "S_attr_exp1": S_attr_exp1,
        "S_attr_exp2": S_attr_exp2,
        "S_attr_exp3": S_attr_exp3,
        "all_attr_gt": all_attr_gt
    }
    
    # Save the tensors
    for name, tensor in tensors.items():
        file_path = feat_path_root + file_names[name]
        torch.save(tensor, file_path)
        print(f"Tensor {name} saved to {file_path}")

def cal_all_stats(S_attr_exp1,all_logits,all_logits_org,all_attr_gt,all_pair_gt,pairs,yfs,stats,test_dataset,config):

    global val_uce_list_ep1
    global val_uce_list_ep2
    global val_uce_list_ep3
    global use_fs
    global val_acc_ep1
    global val_acc_ep2
    global val_acc_ep3
    global val_uce_list_ep2_att
    
    
    # 找到all_obj_gt中每個元素對於在pairs當中的位置
    obj_idxs = [torch.where(pairs[:, 1] == obj)[0] for obj in all_obj_gt]
    # 並將對應的位置將S_all_logits>S_logit_attr_ours
    attr_exp3 = torch.Tensor()
    for i,obj_idx in zip(range(len(obj_idxs)),obj_idxs):
        attr_exp3 =  torch.cat([attr_exp3 ,all_logits[i,obj_idx].unsqueeze(0)], dim=0)
    S_attr_exp3 = F.softmax(attr_exp3, dim=1)
    
    if config.MCDP:
        S_pair_exp2 = F.softmax(all_logits, dim=1)
    else:    
        S_pair_exp2 = F.softmax(all_logits_org, dim=1)
        
        
    a = len(torch.unique(pairs[:, 0])) # 屬性的數量
    b = len(torch.unique(pairs[:, 1])) # 物體的數量

    S_attr_exp2 = torch.zeros((S_pair_exp2.shape[0], a))

    for i in range(a):
        start_idx = i * b
        end_idx = start_idx + b
        S_attr_exp2[:, i], _ = torch.max(S_pair_exp2[:, start_idx:end_idx], dim=1)
        
    
    
    if test_dataset.phase == "val" and val_uce_list_ep1==[]:
#         save_tensors(test_dataset.phase , config.train_look_up_table , "./feat/", S_attr_exp1, S_attr_exp2, S_attr_exp3, all_attr_gt)
        
        uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1, bin_variances_ep1 = compute_uce(S_attr_exp1, all_attr_gt)

        val_uce_list_ep1.append(uce_expert1)
        val_uce_list_ep1.append(bin_uncertainties_expert1)
        val_uce_list_ep1.append(bin_errors_expert1)
        val_uce_list_ep1.append(prop_in_bin_values_expert1)
        val_uce_list_ep1.append(bin_n_samples_ep1)
        val_uce_list_ep1.append(bin_variances_ep1)
        
        uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2, bin_variances_ep2 = compute_uce(S_attr_exp2, all_attr_gt)
        val_uce_list_ep2.append(uce_expert2)
        val_uce_list_ep2.append(bin_uncertainties_expert2)
        val_uce_list_ep2.append(bin_errors_expert2)
        val_uce_list_ep2.append(prop_in_bin_values_expert2)
        val_uce_list_ep2.append(bin_n_samples_ep2)
        val_uce_list_ep2.append(bin_variances_ep2)
        
        uce_expert3, bin_uncertainties_expert3, bin_errors_expert3, prop_in_bin_values_expert3,bin_n_samples_ep3, bin_variances_ep3 = compute_uce(S_attr_exp3, all_attr_gt)
        val_uce_list_ep3.append(uce_expert3)
        val_uce_list_ep3.append(bin_uncertainties_expert3)
        val_uce_list_ep3.append(bin_errors_expert3)
        val_uce_list_ep3.append(prop_in_bin_values_expert3)
        val_uce_list_ep3.append(bin_n_samples_ep3)
        val_uce_list_ep3.append(bin_variances_ep3)
        
        MOM_probs,_ = choose_best_three_expert_new(S_attr_exp1,S_attr_exp2,S_attr_exp3,all_attr_gt,val_uce_list_ep1,val_uce_list_ep2,val_uce_list_ep3,test_dataset.phase)
        MOM_acc_val = accuracy(MOM_probs,all_attr_gt)

    if test_dataset.phase == "val" and use_fs:
        val_uce_list_ep1 =[]
        val_uce_list_ep2 =[]
        val_uce_list_ep3 =[]
        
        
        uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1, bin_variances_ep1 = compute_uce(S_attr_exp1, all_attr_gt)

        val_uce_list_ep1.append(uce_expert1)
        val_uce_list_ep1.append(bin_uncertainties_expert1)
        val_uce_list_ep1.append(bin_errors_expert1)
        val_uce_list_ep1.append(prop_in_bin_values_expert1)
        val_uce_list_ep1.append(bin_n_samples_ep1)
        val_uce_list_ep1.append(bin_variances_ep1)
        
        uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2, bin_variances_ep2 = compute_uce(S_attr_exp2, all_attr_gt)
        val_uce_list_ep2.append(uce_expert2)
        val_uce_list_ep2.append(bin_uncertainties_expert2)
        val_uce_list_ep2.append(bin_errors_expert2)
        val_uce_list_ep2.append(prop_in_bin_values_expert2)
        val_uce_list_ep2.append(bin_n_samples_ep2)
        val_uce_list_ep2.append(bin_variances_ep2)
        
        uce_expert3, bin_uncertainties_expert3, bin_errors_expert3, prop_in_bin_values_expert3,bin_n_samples_ep3, bin_variances_ep3 = compute_uce(S_attr_exp3, all_attr_gt)
        val_uce_list_ep3.append(uce_expert3)
        val_uce_list_ep3.append(bin_uncertainties_expert3)
        val_uce_list_ep3.append(bin_errors_expert3)
        val_uce_list_ep3.append(prop_in_bin_values_expert3)
        val_uce_list_ep3.append(bin_n_samples_ep3)
        val_uce_list_ep3.append(bin_variances_ep3)
        print("use_val_fs")
        use_fs = False
    
    weight_ep1 = 1.0
    weight_ep2 = 1.0
    weight_ep3 = 1.0
    if config.weighted:
        if test_dataset.phase == 'test':
            print("using test weighted voting")
            weight_ep1,weight_ep2,weight_ep3=val_acc_ep1[0],val_acc_ep2[1],val_acc_ep3[1]
            print(weight_ep1,weight_ep2,weight_ep3)
            
        #在test時載入val acc的數值 06/21
    
#     save_tensors(config.dataset + test_dataset.phase , yfs + config.train_look_up_table , "./feat/", S_attr_exp1, S_attr_exp2, S_attr_exp3, all_attr_gt)
    #計算table準確度
    table_pred_ep12 = choose_best_expert_ex(S_attr_exp1, S_attr_exp2, all_attr_gt,test_dataset,val_uce_list_ep1,val_uce_list_ep2,weight_ep1,weight_ep2)
    table_acc_ep12 = accuracy(table_pred_ep12,all_attr_gt)

    table_expert_ep13 = choose_best_expert_ex(S_attr_exp1, S_attr_exp3, all_attr_gt,test_dataset,val_uce_list_ep1,val_uce_list_ep3,weight_ep1,weight_ep3)
    table_acc_ep13 = accuracy(table_expert_ep13,all_attr_gt)
    
    table_pred_ep23 = choose_best_expert_ex(S_attr_exp2, S_attr_exp3, all_attr_gt,test_dataset,val_uce_list_ep2,val_uce_list_ep3,weight_ep2,weight_ep3)
    table_acc_ep23 = accuracy(table_pred_ep23,all_attr_gt)
    
    # 計算3位專家綜合準確度
    tabel_pred_ep123 = choose_best_three_expert(S_attr_exp1,S_attr_exp2,S_attr_exp3,pairs,all_attr_gt,all_pair_gt,test_dataset,val_uce_list_ep1,val_uce_list_ep2,val_uce_list_ep3,weight_ep1,weight_ep2,weight_ep3)
    tabel_acc_ep123 = accuracy(tabel_pred_ep123,all_attr_gt)
    
    MOM_probs,ERV_SoP_acc = choose_best_three_expert_new(S_attr_exp1,S_attr_exp2,S_attr_exp3,all_attr_gt,val_uce_list_ep1,val_uce_list_ep2,val_uce_list_ep3,test_dataset.phase)
    MOM_acc = accuracy(MOM_probs,all_attr_gt)
    #計算單一專家準確度
    preds_expert1 = torch.argmax(S_attr_exp1, dim=1)
    expert1_acc= accuracy(preds_expert1,all_attr_gt)
    

    preds_expert2_pairs = torch.argmax(S_pair_exp2, dim=1)
    preds_expert2 = pairs[preds_expert2_pairs][:, 0].cpu()    
    expert2_acc=accuracy(preds_expert2,all_attr_gt)
    
    ###20230912更改 新增att_exp2

    a = len(torch.unique(pairs[:, 0])) # 屬性的數量
    b = len(torch.unique(pairs[:, 1])) # 物體的數量

    S_attr_exp2 = torch.zeros((S_pair_exp2.shape[0], a))

    for i in range(a):
        start_idx = i * b
        end_idx = start_idx + b
        S_attr_exp2[:, i], _ = torch.max(S_pair_exp2[:, start_idx:end_idx], dim=1)
    ###20230912更改 新增att_exp2
    

    preds_expert3 = torch.argmax(S_attr_exp3, dim=1)
    expert3_acc= accuracy(preds_expert3,all_attr_gt)
    
    #計算f1-score
    f1_expert1 = f1_score(all_attr_gt.cpu().numpy(), preds_expert1.cpu().numpy(), average='macro')
    f1_expert2 = f1_score(all_attr_gt.cpu().numpy(), preds_expert2.cpu().numpy(), average='macro')
    f1_expert3 = f1_score(all_attr_gt.cpu().numpy(), preds_expert3.cpu().numpy(), average='macro')
    
    #計算ep1 UCE
    uce_and_bin_values_all_logits_attr = [compute_uce(S_attr_exp1, all_attr_gt)]
    uce_ep1=uce_and_bin_values_all_logits_attr[0][0] 
    #計算ep2 UCE    
    uce_and_bin_values = [compute_uce(S_attr_exp2, all_attr_gt)]
    uce_ep2=uce_and_bin_values[0][0]   
    #計算ep3 根據物體的資訊計算UCE
    uce_and_bin_values = [compute_uce(S_attr_exp3, all_attr_gt)]
    uce_ep3=uce_and_bin_values[0][0]   
    
    # 計算FOR FDR
    FDR_expert1, FOR_expert1 = calculate_weighted_multiclass_fdr_for(preds_expert1.cpu().numpy(), all_attr_gt.cpu().numpy())
    FDR_expert2, FOR_expert2 = calculate_weighted_multiclass_fdr_for(preds_expert2.cpu().numpy(), all_attr_gt.cpu().numpy())
    FDR_expert3, FOR_expert3 = calculate_weighted_multiclass_fdr_for(preds_expert3.cpu().numpy(), all_attr_gt.cpu().numpy())
    
    ### 消融實驗
    #Product of Experts 綜合ep13
    POE_probs_ = torch.stack([S_attr_exp1,S_attr_exp2, S_attr_exp3])
    POE_probs = product_of_experts(POE_probs_)
    POE_pred = np.argmax(POE_probs, axis=1)
    POE_acc_123 =accuracy(POE_pred,all_attr_gt)
    
    SOE_probs = (S_attr_exp1+S_attr_exp2+ S_attr_exp3)/3
    SOE_pred = np.argmax(SOE_probs, axis=1)
    SOE_acc_123 =accuracy(SOE_pred,all_attr_gt)
    
    ###20230912更改 新增SOE POE12 13 23    
    def calculate_SOE_POE(ep1, ep2, ep3=None):
        if ep3 is not None:
            SOE_probs = (ep1 + ep2 + ep3) / 3
            POE_probs = product_of_experts(torch.stack([ep1, ep2, ep3]))
        else:
            SOE_probs = (ep1 + ep2) / 2
            POE_probs = product_of_experts(torch.stack([ep1, ep2]))

        SOE_pred = torch.argmax(SOE_probs, dim=1)
        POE_pred = torch.argmax(POE_probs, dim=1)

        return SOE_pred, POE_pred
    SOE_pred_12, POE_pred_12 = calculate_SOE_POE(S_attr_exp1, S_attr_exp2)
    SOE_pred_23, POE_pred_23 = calculate_SOE_POE(S_attr_exp2, S_attr_exp3)
    SOE_pred_13, POE_pred_13 = calculate_SOE_POE(S_attr_exp1, S_attr_exp3)

    SOE_acc_12 = accuracy(SOE_pred_12, all_attr_gt)
    SOE_acc_23 = accuracy(SOE_pred_23, all_attr_gt)
    SOE_acc_13 = accuracy(SOE_pred_13, all_attr_gt)

    POE_acc_12 = accuracy(POE_pred_12, all_attr_gt)
    POE_acc_23 = accuracy(POE_pred_23, all_attr_gt)
    POE_acc_13 = accuracy(POE_pred_13, all_attr_gt)
    
    ###20230912更改 新增簡單投票
    
    simple_voting_pred_12 = voting_(preds_expert1, preds_expert2, default_expert=2)
    simple_voting_acc_12 = accuracy(simple_voting_pred_12, all_attr_gt)

    simple_voting_pred_23 = voting_(preds_expert2, preds_expert3, default_expert=2)
    simple_voting_acc_23 = accuracy(simple_voting_pred_23, all_attr_gt)

    simple_voting_pred_13 = voting_(preds_expert1, preds_expert3, default_expert=2)
    simple_voting_acc_13 = accuracy(simple_voting_pred_13, all_attr_gt)

    # 回到原始函數，進行三個專家的簡單投票
    simple_voting_pred_123 = voting(preds_expert1, preds_expert2, preds_expert3, default_expert=3)
    simple_voting_acc_123 = accuracy(simple_voting_pred_123, all_attr_gt)

    # 加權投票，都不一樣選2
    weighted_voting_pred_12 = weighted_voting_(preds_expert1, preds_expert2, weights=[0.2, 0.7], default_expert=1)
    weighted_voting_acc_12 = accuracy(weighted_voting_pred_12, all_attr_gt)

    weighted_voting_pred_23 = weighted_voting_(preds_expert2, preds_expert3, weights=[0.2, 0.7], default_expert=1)
    weighted_voting_acc_23 = accuracy(weighted_voting_pred_23, all_attr_gt)

    weighted_voting_pred_13 = weighted_voting_(preds_expert1, preds_expert3, weights=[0.7, 0.2], default_expert=1)
    weighted_voting_acc_13 = accuracy(weighted_voting_pred_13, all_attr_gt)

    weighted_voting_pred_123 = weighted_voting(preds_expert1, preds_expert2, preds_expert3, weights=[0.1, 0.4, 0.5], default_expert=2)
    weighted_voting_acc_123 = accuracy(weighted_voting_pred_123, all_attr_gt)
    
    
    ###20230912更改 新增SOE POE12 13 23 
    
    if config.weighted:
        print("using val weighted voting")
        if test_dataset.phase == 'val':
            val_acc_ep1.append(expert1_acc)
            val_acc_ep2.append(expert2_acc)
            val_acc_ep3.append(expert3_acc)
            #在val時載入append val acc的數值 06/21
    if yfs == "":
        stats['attr_acc_simple_voting_acc_123'+yfs] = simple_voting_acc_123
        stats['attr_acc_simple_voting_acc_12'+yfs] = simple_voting_acc_12
        stats['attr_acc_simple_voting_acc_13'+yfs] = simple_voting_acc_13
        stats['attr_acc_simple_voting_acc_23'+yfs] = simple_voting_acc_23

        stats['attr_acc_weighted_voting_acc_123'+yfs] = weighted_voting_acc_123
        stats['attr_acc_weighted_voting_acc_12'+yfs] = weighted_voting_acc_12
        stats['attr_acc_weighted_voting_acc_13'+yfs] = weighted_voting_acc_13
        stats['attr_acc_weighted_voting_acc_23'+yfs] = weighted_voting_acc_23
    
        stats['attr_acc_uce_ep1'+yfs] = uce_ep1
        stats['attr_acc_uce_ep2'+yfs] = uce_ep2
        stats['attr_acc_uce_ep3'+yfs] = uce_ep3
        
        stats['FDR_expert1'+yfs] = FDR_expert1
        stats['FDR_expert2'+yfs] = FDR_expert2    
        stats['FDR_expert3'+yfs] = FDR_expert3
        stats['FOR_expert1'+yfs] = FOR_expert1
        stats['FOR_expert2'+yfs] = FOR_expert2    
        stats['FOR_expert3'+yfs] = FOR_expert3
        
        stats['attr_acc_f1_expert1'+yfs] = f1_expert1
        stats['attr_acc_f1_expert2'+yfs] = f1_expert2    
        stats['attr_acc_f1_expert3'+yfs] = f1_expert3
        
        stats['attr_acc_POE_acc_123'+yfs] = POE_acc_123
        stats['attr_acc_POE_acc_12'+yfs] = POE_acc_12
        stats['attr_acc_POE_acc_13'+yfs] = POE_acc_13
        stats['attr_acc_POE_acc_23'+yfs] = POE_acc_23

        stats['attr_acc_SOE_acc_123'+yfs] =SOE_acc_123
        stats['attr_acc_SOE_acc_12'+yfs] = SOE_acc_12
        stats['attr_acc_SOE_acc_13'+yfs] = SOE_acc_13
        stats['attr_acc_SOE_acc_23'+yfs] = SOE_acc_23

        stats['attr_acc_table_acc_ep12'+yfs] = table_acc_ep12
        stats['attr_acc_table_acc_ep23'+yfs] = table_acc_ep23
        stats['attr_acc_table_acc_ep13'+yfs] = table_acc_ep13
        stats['attr_acc_tabel_acc_ep123'+yfs] = tabel_acc_ep123

        stats['MOM_acc'] = MOM_acc
        stats['ERV_SoP_acc'] = ERV_SoP_acc
    
    stats['attr_acc_ep1_cor'] = expert1_acc
    stats['attr_acc_ep2_cor'] = expert2_acc    
    stats['attr_acc_ep3_cor'] = expert3_acc




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
    
    if config.MCDP:
        enable_dropout(model)
        
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
#             break

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
        all_logits_,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config,all_logits_org = None):  
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
    if config.dataset =='mit-states':
        all_logits_attr = torch.zeros(all_logits.size(0), len(test_dataset.attrs))
        for i, (attr_idx, obj_idx) in enumerate(pairs):
            all_logits_attr[:, attr_idx] += all_logits[:, i]
        all_logits_attr /= len(test_dataset.objs)

        att_ped_org = torch.argmax(all_logits_attr,dim=1)
        S_attr_exp1 = F.softmax(all_logits_attr, dim=1)
    else:
        
        all_logits_attr = torch.zeros(all_logits_.size(0), len(test_dataset.attrs))
        for i, (attr_idx, obj_idx) in enumerate(pairs):
            all_logits_attr[:, attr_idx] += all_logits_[:, i]
        all_logits_attr /= len(test_dataset.objs)

        att_ped_org = torch.argmax(all_logits_attr,dim=1)
        S_attr_exp1 = F.softmax(all_logits_attr, dim=1)
        #有使用ys



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
    #--------------------------------------
    if config.open_world == True:
        yfs = ""
        
    
    cal_all_stats(S_attr_exp1,all_logits_,all_logits_org,all_attr_gt,all_pair_gt,pairs,yfs,stats,test_dataset,config)
    #--------------------------------------
    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc

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
        all_logits_org,   #新增原本尚未改過的all_logit
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    
    global use_fs
    if config.train_look_up_table == "val_use_fs":
        use_fs = True
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
    S_attr_exp1 = F.softmax(all_logits_attr, dim=1)
    
    S_pair_exp2 = F.softmax(all_logits_org, dim=1)
    
    obj_idxs = [torch.where(pairs[:, 1] == obj)[0] for obj in all_obj_gt]    
    attr_exp3 = torch.Tensor()
    for i,obj_idx in zip(range(len(obj_idxs)),obj_idxs):
        attr_exp3 =  torch.cat([attr_exp3 ,all_logits[i,obj_idx].unsqueeze(0)], dim=0)
    S_attr_exp3 = F.softmax(attr_exp3, dim=1)
    #計算ep1 attr UCE
    uce_and_bin_values_all_logits_attr = [compute_uce(S_attr_exp1, all_attr_gt)]
    for i, (uce_value, bin_uncertainties, bin_errors, prop_in_bin_values, bin_n_samples, bin_variances) in enumerate(uce_and_bin_values_all_logits_attr):
        plot_dot_UCE_diagram(uce_value, bin_uncertainties, bin_errors, 0, test_dataset, config, prop_in_bin_values, bin_n_samples, bin_variances)
    uce_ep1=uce_and_bin_values_all_logits_attr[0][0]

    #計算ep2 pair UCE
    uce_and_bin_values = [compute_uce(S_pair_exp2, all_pair_gt)]
    for i, (uce_value, bin_uncertainties, bin_errors, prop_in_bin_values, bin_n_samples, bin_variances) in enumerate(uce_and_bin_values):
        plot_dot_UCE_diagram(uce_value, bin_uncertainties, bin_errors, 1, test_dataset, config, prop_in_bin_values, bin_n_samples, bin_variances)
        
    #計算ep3 attr UCE
        uce_and_bin_values = [compute_uce(S_attr_exp3, all_attr_gt)]
        for i, (uce_value, bin_uncertainties, bin_errors, prop_in_bin_values, bin_n_samples, bin_variances) in enumerate(uce_and_bin_values):
            plot_dot_UCE_diagram(uce_value, bin_uncertainties, bin_errors, 2, test_dataset, config, prop_in_bin_values, bin_n_samples, bin_variances)

    yfs = "_yfs"
    cal_all_stats(S_attr_exp1,all_logits_,all_logits_org,all_attr_gt,all_pair_gt,pairs,yfs,stats,test_dataset,config)
    return stats    

def test_yfs_ep2_nfs_ep1(
        test_dataset,
        stats,
        evaluator,
        all_logits,   #這個是nfs
        all_logits_,  #這個是yfs
        all_logits_org,  #新增原本尚未改過的all_logit
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    
    global val_uce_list_ep1
    global val_uce_list_ep2
    global val_uce_list_ep3
    global val_acc_ep1
    global val_acc_ep2
    global val_acc_ep3
    
    
    weight_ep1 = 1.0
    weight_ep2 = 1.0
    weight_ep3 = 1.0
    if config.weighted:
        if test_dataset.phase == 'test':
            print("using test weighted voting")
            weight_ep1,weight_ep2,weight_ep3=val_acc_ep1[0],val_acc_ep2[1],val_acc_ep3[1]
            print(weight_ep1,weight_ep2,weight_ep3)
            
    
    
    
    attr2idx = test_dataset.attr2idx
    obj2idx = test_dataset.obj2idx
    pairs_dataset = test_dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    #nfs_ep1 使用all_logits 
    all_logits_attr = torch.zeros(all_logits.size(0), len(test_dataset.attrs))
    for i, (attr_idx, obj_idx) in enumerate(pairs):
        all_logits_attr[:, attr_idx] += all_logits[:, i]
    all_logits_attr /= len(test_dataset.objs)
    att_ped_org = torch.argmax(all_logits_attr,dim=1)
    
    # yfs_ep3 使用all_logits_
    obj_idxs = [torch.where(pairs[:, 1] == obj)[0] for obj in all_obj_gt]
    attr_exp3 = torch.Tensor()
    for i,obj_idx in zip(range(len(obj_idxs)),obj_idxs):
        attr_exp3 =  torch.cat([attr_exp3 ,all_logits_[i,obj_idx].unsqueeze(0)], dim=0)
        
    
    
    S_attr_exp1 = F.softmax(all_logits_attr, dim=1)   #這個是nfs_ep1
    S_pair_exp2 = F.softmax(all_logits_org, dim=1)       #這個是yfs_ep2_org
    S_attr_exp3 = F.softmax(attr_exp3, dim=1)         #這個是yfs_ep3
    
    table_pred_yfs_ep2_nfs_ep1 = choose_best_expert(S_attr_exp1, S_pair_exp2, all_attr_gt,all_pair_gt,pairs,test_dataset,val_uce_list_ep1,val_uce_list_ep2,weight_ep1,weight_ep2)
    table_acc_yfs_ep2_nfs_ep1= accuracy(table_pred_yfs_ep2_nfs_ep1,all_attr_gt)
    
    table_pred_yfs_ep3_nfs_ep1 = choose_best_expert_ex(S_attr_exp1, S_attr_exp3, all_attr_gt,test_dataset,val_uce_list_ep1,val_uce_list_ep3,weight_ep1,weight_ep3)
    table_acc_yfs_ep3_nfs_ep1= accuracy(table_pred_yfs_ep3_nfs_ep1,all_attr_gt) 
        
    
    table_pred_yfs_ep23_nfs_ep1 = choose_best_three_expert(S_attr_exp1,S_pair_exp2,S_attr_exp3,pairs,all_attr_gt,all_pair_gt,test_dataset,val_uce_list_ep1,val_uce_list_ep2,val_uce_list_ep3,weight_ep1,weight_ep2,weight_ep3)
    table_acc_yfs_ep23_nfs_ep1 = accuracy(table_pred_yfs_ep23_nfs_ep1,all_attr_gt)

    stats['table_acc_yfs_ep2_nfs_ep1'] = table_acc_yfs_ep2_nfs_ep1
    stats['table_acc_yfs_ep3_nfs_ep1'] = table_acc_yfs_ep3_nfs_ep1
    stats['table_acc_yfs_ep23_nfs_ep1'] = table_acc_yfs_ep23_nfs_ep1
    
    return stats  


def construct_uce_table(model, train_dataset,train_dataset_CL,  config):
    with torch.no_grad():
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt,logits_attrs_list ,loss_avg = predict_logits(
            model, train_dataset,  config)
    if config.train_look_up_table == True:
        feasibility_path = os.path.join(
            DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(
            feasibility_path,
            map_location='cpu')['feasibility']
        best_th = config.threshold
        all_logits = threshold_with_feasibility(all_logits, train_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)

    pairs = torch.tensor([(train_dataset.attr2idx[attr], train_dataset.obj2idx[obj])
                                for attr, obj in train_dataset.pairs]).cuda()
    attr2idx_CL = train_dataset_CL.attr2idx
    obj2idx_CL = train_dataset_CL.obj2idx

    train_pairs = torch.tensor([(attr2idx_CL[attr], obj2idx_CL[obj])
                                for attr, obj in train_dataset_CL.train_pairs]).cuda()

    all_pair_gt_OW = []
    #將Closed world 的gt 轉換成Open world pairs
    # for each pair in train_pairs_all_pair_gt
    for pair in train_pairs[all_pair_gt]:
        # find the index in pairs
        index = (pairs == pair).all(dim=1).nonzero().squeeze().item()
        all_pair_gt_OW.append(index)

    all_pair_gt_OW = torch.tensor(all_pair_gt_OW)  # convert to tensor if you need



    all_logits_attr = torch.zeros(all_logits.size(0), len(train_dataset.attrs))
    for i, (attr_idx, obj_idx) in enumerate(pairs):
        all_logits_attr[:, attr_idx] += all_logits[:, i]
    all_logits_attr /= len(test_dataset.objs)
    att_ped_org = torch.argmax(all_logits_attr,dim=1)
    S_attr_exp1 = F.softmax(all_logits_attr, dim=1)
    
    S_pair_exp2 = F.softmax(all_logits, dim=1)
    # 並將對應的位置將S_all_logits>S_logit_attr_ours
    
    attr_exp3 = torch.Tensor()    
    # 找到all_obj_gt中每個元素對於在pairs當中的位置
    obj_idxs = [torch.where(pairs[:, 1] == obj)[0] for obj in all_obj_gt]
    for i,obj_idx in zip(range(len(obj_idxs)),obj_idxs):
        attr_exp3 =  torch.cat([attr_exp3 ,all_logits[i,obj_idx].unsqueeze(0)], dim=0)
    S_attr_exp3 = F.softmax(attr_exp3, dim=1)
    
    probs_expert1 ,probs_expert2,probs_expert3 = S_attr_exp1,S_pair_exp2,S_attr_exp3
    global val_uce_list_ep1
    global val_uce_list_ep2
    global val_uce_list_ep3
    
    uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, prop_in_bin_values_expert1,bin_n_samples_ep1, bin_variances_ep1 = compute_uce(S_attr_exp1, all_attr_gt)
    plot_dot_UCE_diagram(uce_expert1, bin_uncertainties_expert1, bin_errors_expert1, 0, train_dataset, config, prop_in_bin_values_expert1, bin_n_samples_ep1, bin_variances_ep1)

    val_uce_list_ep1.append(uce_expert1)
    val_uce_list_ep1.append(bin_uncertainties_expert1)
    val_uce_list_ep1.append(bin_errors_expert1)
    val_uce_list_ep1.append(prop_in_bin_values_expert1)
    val_uce_list_ep1.append(bin_n_samples_ep1)
    val_uce_list_ep1.append(bin_variances_ep1)

    

    
    uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, prop_in_bin_values_expert2,bin_n_samples_ep2, bin_variances_ep2 = compute_uce(S_pair_exp2, all_pair_gt_OW)
    plot_dot_UCE_diagram(uce_expert2, bin_uncertainties_expert2, bin_errors_expert2, 1, train_dataset, config, prop_in_bin_values_expert2, bin_n_samples_ep2, bin_variances_ep2)
    val_uce_list_ep2.append(uce_expert2)
    val_uce_list_ep2.append(bin_uncertainties_expert2)
    val_uce_list_ep2.append(bin_errors_expert2)
    val_uce_list_ep2.append(prop_in_bin_values_expert2)
    val_uce_list_ep2.append(bin_n_samples_ep2)
    val_uce_list_ep2.append(bin_variances_ep2)

    uce_expert3, bin_uncertainties_expert3, bin_errors_expert3, prop_in_bin_values_expert3,bin_n_samples_ep3, bin_variances_ep3 = compute_uce(S_attr_exp3, all_attr_gt)
    plot_dot_UCE_diagram(uce_expert3, bin_uncertainties_expert3, bin_errors_expert3, 2, train_dataset, config, prop_in_bin_values_expert3, bin_n_samples_ep3, bin_variances_ep3)
    val_uce_list_ep3.append(uce_expert3)
    val_uce_list_ep3.append(bin_uncertainties_expert3)
    val_uce_list_ep3.append(bin_errors_expert3)
    val_uce_list_ep3.append(prop_in_bin_values_expert3)
    val_uce_list_ep3.append(bin_n_samples_ep3)
    val_uce_list_ep3.append(bin_variances_ep3)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

if __name__ == "__main__":
    config = parser.parse_args()
    load_args(YML_PATH[config.dataset], config)

    # set the seed value
    
    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")



    dataset_path = config.dataset_path
    
    print('loading train dataset')
    train_dataset_CL = CompositionDataset(dataset_path,
                                 phase='train',
                                 split='compositional-split-natural',
                                 open_world=config.open_world)
    
    print('loading train dataset')
    train_dataset = CompositionDataset(dataset_path,
                                 phase='train',
                                 split='compositional-split-natural',
                                 open_world=config.open_world)

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



    with open('config/' + config.dataset + '.yml') as file:
        config_dict = yaml.safe_load(file)

    class class_Config_org:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        setattr(self, sub_key, sub_value)
                else:
                    setattr(self, key, value)

    config_org = class_Config_org(config_dict)

    model_org = DFSP(config_org, attributes=attributes, classes=classes, offset=offset).cuda()
    model_org.load_state_dict(torch.load(config_org.load_model))
    
    val_uce_list_ep1 = []
    val_uce_list_ep2 = []
    val_uce_list_ep3 = []
    use_fs = False
    global_a = [] 
    val_acc_ep1 = []
    val_acc_ep2 = []
    val_acc_ep3 = []

    
    
    
       ### 使用train data建立UCE lookup table
    if config.train_look_up_table == True or config.train_look_up_table == False:
        construct_uce_table(model, train_dataset,train_dataset_CL,  config)


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
            if config.MCDP:
                print("Using MCDP")
                n_iterations = config.MCD_step
                all_logits_list, all_attr_gt_list, all_obj_gt_list, all_pair_gt_list, logits_attrs_list_list = [], [], [], [], []

                for _ in range(n_iterations):
                    all_logits__, all_attr_gt, all_obj_gt, all_pair_gt, logits_attrs_list_, loss_avg = predict_logits(model, val_dataset, config)
                    all_logits_list.append(all_logits__)
                    logits_attrs_list_list.append(logits_attrs_list_)

                # Calculate the mean for each list of results
                all_logits = torch.stack(all_logits_list).mean(dim=0)
                logits_attrs_list = [torch.stack(logits_attrs).mean(dim=0) for logits_attrs in zip(*logits_attrs_list_list)]



                
            else:    
                all_logits, all_attr_gt, all_obj_gt, all_pair_gt,logits_attrs_list ,loss_avg = predict_logits(
                    model, val_dataset, config)
            
            all_logits_org, _, _, _,_ ,_ = predict_logits(
                model_org, val_dataset, config)  
            if config.open_world:
                print('using threshold: ', best_th)
                all_logits_ = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
                
                all_logits_org_ = threshold_with_feasibility(
                    all_logits_org, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
                
            results = test(
                val_dataset,
                logits_attrs_list,   ###新增
                evaluator,
                all_logits,
                all_logits_,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config,
                all_logits_org_  #新增原本尚未改過的all_logit
            )
            
            results =test__yfs(val_dataset,
                    results,
                    evaluator,
                    all_logits,
                    all_logits_org_,    #新增原本尚未改過的all_logit
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config)
#             results =test_yfs_ep2_nfs_ep1(val_dataset,
#                     results,
#                     evaluator,
#                     all_logits,        
#                     all_logits_,
#                     all_logits_org_,    #新增原本尚未改過的all_logit
#                     all_attr_gt,
#                     all_obj_gt,
#                     all_pair_gt,
#                     config)
        print(results)
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
        if config.MCDP:
            n_iterations = config.MCD_step
            all_logits_list, all_attr_gt_list, all_obj_gt_list, all_pair_gt_list, logits_attrs_list_list = [], [], [], [], []

            for _ in range(n_iterations):
                all_logits__, all_attr_gt, all_obj_gt, all_pair_gt, logits_attrs_list_, loss_avg = predict_logits(model, test_dataset, config)
                all_logits_list.append(all_logits__)
                logits_attrs_list_list.append(logits_attrs_list_)

            # Calculate the mean for each list of results
            all_logits = torch.stack(all_logits_list).mean(dim=0)
            logits_attrs_list = [torch.stack(logits_attrs).mean(dim=0) for logits_attrs in zip(*logits_attrs_list_list)]

        else:
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt, logits_attrs_list,loss_avg = predict_logits(
                model, test_dataset, config)
            
        all_logits_org, _, _, _,_ ,_ = predict_logits(model_org, test_dataset, config)  
        if config.open_world and best_th is not None:
            print('using threshold: ', best_th)
            all_logits_ = threshold_with_feasibility(
                all_logits,
                test_dataset.seen_mask,
                threshold=best_th,
                feasiblity=unseen_scores)
            
            all_logits_org_ = threshold_with_feasibility(
                all_logits_org,
                test_dataset.seen_mask,
                threshold=best_th,
                feasiblity=unseen_scores)
        
        test_stats = test(
            test_dataset,
            logits_attrs_list,   ###新增
            evaluator,
            all_logits,
            all_logits_,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config,
            all_logits_org_  #新增原本尚未改過的all_logit
        )
        
        test_stats =test__yfs(test_dataset,
                test_stats,
                evaluator,
                all_logits,
                all_logits_org_,  #新增原本尚未改過的all_logit
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config)

#         test_stats =test_yfs_ep2_nfs_ep1(test_dataset,
#                 test_stats,
#                 evaluator,
#                 all_logits,        
#                 all_logits_,
#                 all_logits_org_,  #新增原本尚未改過的all_logit
#                 all_attr_gt,
#                 all_obj_gt,
#                 all_pair_gt,
#                 config)
        print(test_stats)
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

        
        
    if config.weighted:
        root = config.load_model[:-2] +'weighted_'
    else:
        root = config.load_model[:-2]
    if config.open_world:
        if config.train_look_up_table == True:
            result_path = root +"train_uce_use_fs"+"open.calibrated.json"
        elif config.train_look_up_table == "val_use_fs":
            result_path = root +"val_uce_use_fs" +"open.calibrated.json"
        elif config.train_look_up_table == "val_nouse_fs":
            result_path = root +"val_nouse_fs" +"open.calibrated.json"
        else:
            result_path = root + "open.calibrated.json"
    else:
        result_path = config.load_model[:-2] + "closed.json"


        
    def handle_special_values(o):
        if isinstance(o, torch.Tensor):
            if o.numel() == 1:
                return o.item()
            else:
                return o.tolist()
        elif isinstance(o, float) and math.isnan(o):
            return "NaN"
        else:
            return o

#     import ipdb 
#     ipdb.set_trace()

    with open('mit_state.json', 'w') as fp:
        json.dump(results, fp, default=handle_special_values)

        
    with open(result_path, 'w') as fp:
        json.dump(results, fp, default=handle_special_values)

        

#     print(results) 


    print("done!")
