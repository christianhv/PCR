#region import
import copy
from io import SEEK_CUR
from typing import Dict, Any
from numpy.lib.ufunclike import _dispatcher
from pydataset import data
from dataset import Dataset
from math import log
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import operator
import time
from operator import itemgetter
import logging
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

#endregion

class PCR:
    # rule dictionary keys
    # make easy to use strings as keys
    rl_antecedent = "rule"
    rl_srch_heu_value = "search_heuristic_value"
    rl_prototype = "prototype_rule"
    rl_cov_idxs = "covered indexes"
    rl_coverage = "coverage"
    rl_classify_error = "classification error"
    rl_new_weights = "new weights"
    rl_correct_cov = "correct covered"
    rl_accuracy = "accuracy"
    predicted = "predicted"

    # modification types of examples weights
    mty_add = "add"
    mty_repl = "replace"
    mty_sub = "substract"
    mty_mult = "multiplication"

    # we identify two types of attributes
    nominal = "string"
    numeric = "numeric"

    # search heuristic weights
    mhw_distance = "distance"
    mhw_coverage = "coverage"
    mhw_dissimilarity = "dissimilarity"

    #search heuristic methods
    srch_heu_CN2 = "CN2"
    srch_heu_WRAcc = "WRAcc"
    srch_heu_other = "Other"

    # examples elimination type
    mmthd_err_weight_cov = "Err-Weight-Covering"
    mmthd_std_cov = "Standard-Covering"

    #region initializing PCR
    def __init__(self,
                 training_data,
                 validation_data,
                 test_data,
                 target=[],
                 target_weight=0.0,
                 multi_direction=False,
                 maximum_targets=2,
                 search_heuristic="CN2",
                 modifying_method="Standard-Covering",
                 covering_err_weight = 0.0, #if zero the correct covered will be deleted
                 covering_weight_threshold = 0.4,
                 minimum_covered = 3,
                 coverage_weight=2.0,
                 distance_weight=3.0,
                 dissimilarity_weight=2.0,
                 maximize_heuristic_weight="coverage"  #Only used when the search_heuristic is other
                ):
        warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'

        self.training_data = training_data.sort_index()
        logging.warning(self.training_data.index)
        self.training_data = self.training_data.reset_index(drop=True)
        logging.warning("SORTED:\n{}".format(self.training_data))
        self.test_data = test_data
        logging.warning("TEST:\n{}".format(self.test_data))

        # Setting indexes of the training data
        self.all_td_indexes = self.training_data.index.values.tolist()
        self.highest_idx = self.all_td_indexes[len(self.all_td_indexes)-1]
        self.column_list = list(self.training_data.columns)
        self.td_dtypes = self.__set_data_types()
        self.global_unique_values = self.__get_unique_values(
            self.training_data)
        self.td_weights = self.__set_td_weights()



        if multi_direction:
            self.target = []
            self.no_target = 0.0
            self.max_num_targets = maximum_targets
            self.tau_target_weight = 0.0
            if target_weight > 0:
                warnings.warn('''
                            | In multi-direction all attributes have the same weight         |
                            | The weight provided is ignored                                 |''',
                              stacklevel=2)
            if (len(target)) > 0:
                warnings.warn('''
                            | If multi-direction=True, no target is taken into account       |
                            | The number of targets will be automatically generated          |
                            | Number of targets will differ by rule, until maximum           |
                            | provided default=2                                             |
                            | If you want to provide target change multi-direction to False  |''',
                              stacklevel=2)
        else:
            if type(target) != list:
                raise ValueError(
                    '| Target not a list. Target must be a list with at least one element|'
                )
            if len(target) < 1:
                raise ValueError(
                    '<<If multi-direction=False Target must be provided>>')
            if target_weight == 0:
                warnings.warn('''
                            | It is recommended to give weight to target attributes.         |
                            | Now assigning value 1 by default                               |
                            ''')
                self.tau_target_weight = 1.0
            else:
                self.tau_target_weight = target_weight

            self.target = target
            self.no_target = len(self.target)
            self.max_num_targets = 0

        #All the weights need to tune the algorithm
        self.search_heuristic = search_heuristic
        if search_heuristic == PCR.srch_heu_CN2:
            warnings.warn('''
                            | In CN2 search heuristic, all heuristic weights are set to zero |
                            | by default. Provide weight values to test their influence.     |''',
                          stacklevel=2)

            self.dispersion_off = 1.0
            self.alpha_coverage_hweight = coverage_weight
            self.beta_distance_hweight = distance_weight  #in original experiments is not used
            self.gamma_dissimilarity_hweight = dissimilarity_weight
        elif search_heuristic == PCR.srch_heu_WRAcc:
            warnings.warn('''
                            | In WRAcc search heuristic the generic dispersion is used and   |
                            | coverage weight is set to 1                                    |
                            | Provide weight values to test their influence.                 |''',
                          stacklevel=2)
            self.dispersion_off = 0.0 ####==> dispersion for WRAcc is calculated below <===###
            self.alpha_coverage_hweight = 1.0
            self.beta_distance_hweight = 0.0  #in original experiments is not used
            self.gamma_dissimilarity_hweight = 0.0
        elif search_heuristic == PCR.srch_heu_other:
            self.dispersion_off = 0.0
            self.alpha_coverage_hweight = coverage_weight
            self.beta_distance_hweight = distance_weight
            self.gamma_dissimilarity_hweight = dissimilarity_weight


        self.modifying_method = modifying_method
        self.zeta_covering_err_weight = covering_err_weight
        self.eps_cov_weight_err_thld = covering_weight_threshold

        self.modifying_method = modifying_method
        self.minimum_covered = 3
        self.maximal_number_of_rules = 100

        # Is it multi directional
        print("Multi direction:{}".format(multi_direction))
        print("Target:{} no._targets:{} max_no_targets:{}".format(
            self.target, self.no_target, self.max_num_targets))
        print("Tau:{}".format(self.tau_target_weight))

        print("disp:{} cov:{} dist:{} diss:{} ".format(
            self.dispersion_off,
            self.alpha_coverage_hweight,
            self.beta_distance_hweight,  #in original experiments is not used
            self.gamma_dissimilarity_hweight))

        print("mod_methd:{} error_weight:{}  err_thld: {}".format(
            self.modifying_method, self.zeta_covering_err_weight,
            self.eps_cov_weight_err_thld))

        ### The structures below need some of the structures above
        # setting the weights
        # attributes weights need to know the weight for the target
        self.attributes_weights = self.__set_default_attributes_weights()
        if search_heuristic == PCR.srch_heu_WRAcc:
            self.dispersion_off = self.__calculate_dispersion_of(
                self.training_data)[0]
            print("disp Off:{}".format(self.dispersion_off))

        self.selectors = self.__get_selectors(multi_direction)
        logging.warning("SELECTORS:{}".format(self.selectors))
        #setting the generic dispersion
        #we need the type of data and the column list
        #self.dispersion_off = self.__calculate_dispersion_of(self.training_data)[0]
        #setting the generic prototype
        self.td_prototype = self.__set_td_prototype()
        self.rule_set = []

        # creating arrays of covered indexes by the targets.
        # to avoid check everytime the covered examples by target values
        self.unique_values_idxs = self.__get_idxs_of_unique_values()

        logging.warning("target indexes by value")
        self.myprint_nested_dict(self.unique_values_idxs)

        logging.warning("disp Off:{}".format(self.dispersion_off))
        logging.warning("generic proto:{}".format(self.td_prototype))

#endregion
#region some tests
    def myprint_nested_dict(self, d):
        stack = list(d.items())
        visited = set()
        while stack:
            k, v = stack.pop()
            if isinstance(v, dict):
                if k not in visited:
                    stack.extend(v.items())
            else:
                logging.warning("%s: %s" % (k, v))

            visited.add(k)

    def __find_best_complex(self, learning_set, limit_best=6, limit_beam=10):
        empty_star = False
        best_candidates = []
        poorest_heuristic_value = 0.0
        star = []
        while not empty_star:
            newstar = self.__specialize_complex(star)
            logging.warning("newstar after specialization")
            logging.warning(newstar)
            newstar_as_rule = []
            if newstar:
                for komplex in newstar:
                    new_candidate = self.__calculate_search_heuristic(
                        learning_set, komplex)
                    newstar_as_rule.append(new_candidate)
                    new_heuristic_value = new_candidate[PCR.rl_srch_heu_value]
                    if new_heuristic_value > poorest_heuristic_value:
                        best_candidates.append(new_candidate)
                        best_candidates = sorted(best_candidates,
                                                 key=itemgetter(
                                                     PCR.rl_srch_heu_value),
                                                 reverse=True)

                        if len(best_candidates) > limit_best:
                            best_candidates.pop()
                logging.warning("best candidate after popping")
                for bx in best_candidates:
                    logging.warning("{}".format(bx[PCR.rl_antecedent],
                                                bx[PCR.rl_srch_heu_value]))
                newstar_as_rule = sorted(newstar_as_rule,
                                         key=itemgetter(PCR.rl_srch_heu_value),
                                         reverse=True)
                newstar = [
                    komplex[PCR.rl_antecedent] for komplex in newstar_as_rule
                ]
                newstar = newstar[0:limit_beam]
                star = newstar
                logging.warning("new star after popping")
                logging.warning(newstar)
                logging.warning("STAR")
                logging.warning(star)
                for komplex in newstar:
                    logging.warning("newstar ele:{}".format(komplex))
            else:
                empty_star = True
        return best_candidates

    def test_find_best_complex(self, limit_rule_set=5, error_threshold=0.3):
        stop_criterion = True
        poorest_ruleset_error = 1.0
        best_rule = {}
        while stop_criterion:
            best_candidates = self.__find_best_complex(self.training_data)
            poorest_iteration_error = 1.0
            #find best of best candidates
            for bc in best_candidates:
                self.__calculate_rule_classify_error(self.test_data, bc)
                if bc[PCR.rl_classify_error] < poorest_iteration_error:
                    best_rule = bc
                    poorest_iteration_error = bc[PCR.rl_classify_error]
            logging.warning("BEST CANDIDATES")
            for bc in best_candidates:
                logging.warning("{} - {} - {} - pro:{}".format(
                    bc[PCR.rl_antecedent], bc[PCR.rl_srch_heu_value],
                    bc[PCR.rl_classify_error], bc[PCR.rl_prototype]))

            logging.warning("compare: {} worst:{}".format(
                best_rule[PCR.rl_classify_error], poorest_ruleset_error))
            #Check if the best candidate is best than the worst in the rule set
            if best_rule[
                    PCR.
                    rl_classify_error] <= poorest_ruleset_error or best_rule[
                        PCR.rl_classify_error] < error_threshold:
                self.rule_set.append(best_rule)
                self.rule_set = sorted(self.rule_set,
                                       key=itemgetter(PCR.rl_classify_error),
                                       reverse=False)
                indexes_fweights = list(best_rule[PCR.rl_correct_cov].keys())
                new_weights = list(best_rule[PCR.rl_correct_cov].values())
                self.__modify_learning_set(indexes_fweights, new_weights)
                #if the rule set is bigger than the limit we take off the worst
                if len(self.rule_set) > limit_rule_set:
                    self.rule_set.pop()
                poorest_ruleset_error = self.rule_set[-1][
                    PCR.rl_classify_error]
            else:
                stop_criterion = False

            if self.training_data.empty:
                stop_criterion = False

            logging.warning("RULE SET")
            for rule in self.rule_set:
                logging.warning("{} - error:{} - sh:{} - pro:{}".format(
                    rule[PCR.rl_antecedent], rule[PCR.rl_classify_error],
                    rule[PCR.rl_srch_heu_value], rule[PCR.rl_prototype]))

    def test_find_best_complex_v02(self,
                                   limit_rule_set=100,
                                   error_threshold=0.35):
        stop_criterion = True
        logging.warning("starting v 02")
        while stop_criterion:
            best_candidates = self.__find_best_complex(self.training_data,
                                                       limit_best=8,
                                                       limit_beam=12)
            for bc in best_candidates:
                self.__calculate_rule_classify_error(self.test_data, bc)
                logging.warning("V0.2 {} - {} - {} - pro:{}".format(
                    bc[PCR.rl_antecedent], bc[PCR.rl_srch_heu_value],
                    bc[PCR.rl_classify_error], bc[PCR.rl_prototype]))

            #Check if the best candidate is best than the worst in the rule set
            self.rule_set = [*self.rule_set, *best_candidates]
            self.rule_set = sorted(self.rule_set,
                                   key=itemgetter(PCR.rl_classify_error),
                                   reverse=False)
            if len(self.rule_set) > limit_rule_set:
                self.rule_set = self.rule_set[0:limit_rule_set]
            #new_worst = self.rule_set[-1][PCR.rl_classify_error]

            for rule in best_candidates:
                to_delete = rule[PCR.rl_cov_idxs]
                self.__remove_covered_examples(to_delete)

            if self.training_data.empty:
                stop_criterion = False

            logging.warning("RULE SET")
            for rule in self.rule_set:
                logging.warning("{} - error:{} - sh:{} - pro:{}".format(
                    rule[PCR.rl_antecedent], rule[PCR.rl_classify_error],
                    rule[PCR.rl_srch_heu_value], rule[PCR.rl_prototype]))

        self.rule_set = [
            rule for rule in self.rule_set
            if rule[PCR.rl_classify_error] < error_threshold
        ]
        self.rule_set = sorted(self.rule_set,
                               key=itemgetter(PCR.rl_coverage),
                               reverse=False)

        logging.warning("After Prunning")
        for rule in self.rule_set:
            logging.warning("{} - error:{} - sh:{} - pro:{}".format(
                rule[PCR.rl_antecedent], rule[PCR.rl_classify_error],
                rule[PCR.rl_srch_heu_value], rule[PCR.rl_prototype]))

    def test_find_best_complex_v03(self,
                                   limit_rule_set=10,
                                   error_threshold=0.3):
        logging.warning("Version 03")
        stop_criterion = True
        sum_ruleset_errors = 0.0
        avg_rule_set_classify_error = 1.0
        best_rule = {}
        while stop_criterion:
            logging.warning("\n\n ### Starting or continuing #### \n")
            best_candidates = self.__find_best_complex(self.training_data)
            poorest_iteration_error = 1.0
            #find best of best candidates
            for bc in best_candidates:
                self.__calculate_rule_classify_error(self.test_data, bc)
                if bc[PCR.rl_classify_error] < poorest_iteration_error:
                    best_rule = bc
                    poorest_iteration_error = bc[PCR.rl_classify_error]
            logging.warning(
                "Best Candidates after calculating error, but still ordered by sh value:"
            )
            for bc in best_candidates:
                logging.warning('''
                 Antecedent:{} 
                 Srch_value:{} 
                 Error:{} 
                 Proto:{}'''.format(bc[PCR.rl_antecedent],
                                    bc[PCR.rl_srch_heu_value],
                                    bc[PCR.rl_classify_error],
                                    bc[PCR.rl_prototype]))

            if (len(self.rule_set) > limit_rule_set):
                last_value = self.rule_set[-1][PCR.rl_classify_error]
                temp_sum_RL = sum_ruleset_errors - last_value
                temp_sum = temp_sum_RL + best_rule[PCR.rl_classify_error]
                avg_classify_error = temp_sum / (len(self.rule_set))
            else:
                temp_sum = sum_ruleset_errors + best_rule[
                    PCR.rl_classify_error]
                avg_classify_error = temp_sum / (len(self.rule_set) + 1)

            #Check if the best candidate is best than the worst in the rule set
            if (avg_classify_error <= avg_rule_set_classify_error):
                #if the rule set is bigger than the limit we take off the worst
                if len(self.rule_set) > limit_rule_set:
                    last_value = self.rule_set[-1][PCR.rl_classify_error]
                    sum_ruleset_errors -= last_value
                    self.rule_set.pop()

                self.rule_set.append(best_rule)
                self.rule_set = sorted(self.rule_set,
                                       key=itemgetter(PCR.rl_classify_error),
                                       reverse=False)
                indexes_to_modify = list(best_rule[PCR.rl_correct_cov].keys())
                num_corrcov_targets = list(
                    best_rule[PCR.rl_correct_cov].values())
                self.__modify_learning_set(indexes_to_modify,
                                           num_corrcov_targets)

                sum_ruleset_errors += best_rule[PCR.rl_classify_error]
                avg_rule_set_classify_error = sum_ruleset_errors / len(
                    self.rule_set)

            else:
                stop_criterion = False

            if self.training_data.empty:
                stop_criterion = False

            logging.warning("RULE SET")
            for rule in self.rule_set:
                logging.warning("{} - error:{} - sh:{} - pro:{}".format(
                    rule[PCR.rl_antecedent], rule[PCR.rl_classify_error],
                    rule[PCR.rl_srch_heu_value], rule[PCR.rl_prototype]))

    def __remove_covered_examples(self, to_delete):
        current_indexes = self.training_data.index.values.tolist(
        )  # we try to find if the examples still exist in the training set and were not deleted by other rule in the iteration
        true_delete = [idx for idx in to_delete if idx in current_indexes]
        logging.warning("index to delete:\n{}".format(to_delete))
        try:
            self.training_data.drop(index=true_delete, inplace=True)
            logging.warning(self.training_data)
            logging.warning("correct deletion:\n{}".format(self.training_data))
        except IndexError:
            logging.warning("IndexError")

#region DEFAULT STRUCTURES(weights, td_prototype, dtypes)
    def __create_target_attributes(self, attributes):
        print("creating attributes")


    def __set_td_weights(self):
        return [1] * (self.highest_idx + 1)

    def __set_default_attributes_weights(self):
        dict_weights = {}
        for col in self.column_list:
            if col in self.target:
                dict_weights[col] = self.tau_target_weight
            else:
                dict_weights[col] = 1 - self.tau_target_weight
        return dict_weights

    def __set_data_types(self):
        dict_dtypes = {}
        for col in self.column_list:
            if (is_string_dtype(self.training_data[col])):
                data_type = "string"
                dict_dtypes.update({col: "string"})
            elif (is_numeric_dtype(self.training_data[col])):
                data_type = "numeric"
                dict_dtypes.update({col: "numeric"})
            #print("col: {} \n data: {}".format(col, data_type))
        return dict_dtypes

    def __set_td_prototype(self):
        td_prototype = self.__calculate_relative_freq(self.training_data)
        for col in self.column_list:
            if self.td_dtypes[col] == "numeric":
                td_prototype[col] = (self.training_data[col].mean(), self.training_data[col].std())
        return td_prototype


#endregion

#region Modify examples weights

    def __modify_learning_set(self, indexes,
                              targets_covered_by_new_rule_in_set):
        modifier = [self.zeta_covering_err_weight - 1] * len(
            targets_covered_by_new_rule_in_set
        )  #the covering weight times the length of the covered examples
        logging.warning("MLS - covering weight:{}".format(
            self.zeta_covering_err_weight))
        logging.warning("MLS - targets:{}".format(
            type(targets_covered_by_new_rule_in_set)))
        uno = [1] * len(
            targets_covered_by_new_rule_in_set
        )  # a vector with ones times the length covered examples
        no_targets = [self.no_target] * len(
            targets_covered_by_new_rule_in_set
        )  # vector with number of targets times the length of covered_examples
        targets_covered_by_new_rule_in_set = list(
            map(operator.truediv, targets_covered_by_new_rule_in_set,
                no_targets)
        )  # divide all the covered targets in each example by the number of targets
        targets_covered_by_new_rule_in_set = list(
            map(operator.mul, targets_covered_by_new_rule_in_set, modifier)
        )  #multiply the modifier by the targets covered/by number of targets
        targets_covered_by_new_rule_in_set = list(
            map(operator.add, targets_covered_by_new_rule_in_set, uno)
        )  # add 1 to the result before, most of the time lower than 1 if zero then the final modifier is zero and example is immediately deleted
        self.__modify_examples_weights(
            indexes, targets_covered_by_new_rule_in_set,
            PCR.mty_mult)  # the true modification of the value
        to_delete = [
            idx for idx in indexes
            if self.td_weights[idx] < self.eps_cov_weight_err_thld
        ]  # find the examples to delet
        self.__modify_examples_weights(
            to_delete, [0], PCR.mty_repl
        )  #since these values are lower than the threshold we set them in zero

        self.__remove_covered_examples(to_delete)  # deleting the indexes
        #self.__drop_zero_weight_examples(indexes)

    def __modify_examples_weights(self, indexes, new_weights, manner):
        #print("Modify:", indexes, new_weights)
        if len(new_weights) == 1:
            if manner == PCR.mty_add:
                for index in indexes:
                    self.td_weights[index] += new_weights[0]
            elif manner == PCR.mty_repl:
                for index in indexes:
                    #print("i:{} len:{}".format(index, len(self.td_weights)))
                    self.td_weights[index] = new_weights[0]
            elif manner == PCR.mty_sub:
                for index in indexes:
                    self.td_weights[index] -= new_weights[0]
            elif manner == PCR.mty_mult:
                for index in indexes:
                    self.td_weights[index] *= new_weights[0]
            else:
                logging.warning("Indicated manner not found")
        elif (len(new_weights) == len(indexes)):
            #print("they are")
            if manner == PCR.mty_add:
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] += replacement
            elif manner == PCR.mty_repl:
                logging.warning("Replacement")
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] = replacement
            elif manner == PCR.mty_sub:
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] -= replacement
            elif manner == PCR.mty_mult:
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] *= replacement
            else:
                logging.warning("Indicated manner not found")
        else:
            logging.warning("Arguments do not fulfill requirements")

#endregion

#region total SEARCH HEURISTIC

    def __calculate_search_heuristic(self, learning_set, rule_complex):
        rule_packed = {}
        sh_value = 0.0
        logging.warning("[rule_complex]:{}".format(rule_complex))
        covered_indexes = self.__find_covered_examples(learning_set,
                                                       rule_complex)
        if covered_indexes:
            #print("COVERED:{}".format(covered_indexes))
            rule_coverage = self.__calculate_coverage(covered_indexes)
            rule_dispersion, rule_prototype = self.__calculate_dispersion_of(
                learning_set.loc[covered_indexes])
            rule_dissimilarity = self.__calculate_dissimilarity(rule_prototype)
            if len(self.rule_set):
                rule_distance = (
                    self.__calculate_rule_distance(covered_indexes))
            else:
                rule_distance = 0
            if self.search_heuristic == PCR.srch_heu_CN2 or self.search_heuristic == PCR.srch_heu_other:
                sh_value = (
                    (self.alpha_coverage_hweight * rule_coverage) +
                    (self.dispersion_off - rule_dispersion) +
                    (self.gamma_dissimilarity_hweight * rule_dissimilarity) +
                    (self.beta_distance_hweight * rule_distance))
            elif self.search_heuristic == PCR.srch_heu_WRAcc:
                sh_value = (
                    (rule_coverage**self.alpha_coverage_hweight) *
                    (self.dispersion_off - rule_dispersion) *
                    (rule_dissimilarity**self.gamma_dissimilarity_hweight) *
                    (rule_distance**self.beta_distance_hweight))

            logging.warning("COMPLEX:{}".format(rule_complex))
            logging.warning("COVERAGE:{}".format(rule_coverage))
            logging.warning("LEN COVERAGE:{}".format(len(covered_indexes)))
            logging.warning("DISPERSION:{}".format(rule_dispersion))
            logging.warning("RULE PROTOTYPE:{}".format(rule_prototype))
            logging.warning("DISSIMILARITY:{}".format(rule_dissimilarity))
            logging.warning("DISTANCE:{}".format(rule_distance))
            logging.warning("sh_value:{}".format(sh_value))
            #cov_to_loop = copy.deepcopy(covered_indexes)

            rule_packed = {
                PCR.rl_antecedent: rule_complex,
                PCR.rl_srch_heu_value: sh_value,
                PCR.rl_classify_error: 1.0,
                PCR.rl_accuracy: 0.0,
                PCR.rl_coverage: rule_coverage,
                PCR.rl_prototype: rule_prototype,
                PCR.rl_new_weights: {},
                PCR.rl_cov_idxs: covered_indexes,
                PCR.rl_correct_cov: {k: 0
                                     for k in covered_indexes}
            }

            for target_attribute in self.target:
                value = max(rule_prototype[target_attribute],
                            key=rule_prototype[target_attribute].get)
                logging.warning("target attr:{}".format(
                    target_attribute, value))
                correct_covered = list(
                    (set(covered_indexes)
                     & set(self.target_val_idxs[target_attribute][value])))
                #print("correct", self.training_data.loc[correct_covered[0]])
                for i in correct_covered:
                    rule_packed[PCR.rl_correct_cov][i] += 1.0
        else:
            rule_packed = {
                PCR.rl_antecedent: rule_complex,
                PCR.rl_srch_heu_value: 0.0,
                PCR.rl_classify_error: 1.0,
                PCR.rl_accuracy: 0.0,
                PCR.rl_prototype: self.td_prototype,
                PCR.rl_new_weights: {},
                PCR.rl_cov_idxs: [],
                PCR.rl_correct_cov: {}
            }

            #print("correct", rule_packed[PCR.correct_cov], "\n\n")
        return rule_packed

# region Calculate Classification Error

    def __calculate_rule_classify_error(self, tdata, rule):
        #print("Calculate class error")
        rule_keys = list(
            rule[PCR.rl_antecedent].keys())  # making it more readable
        #check if target is in the rule
        rule_class_error = 0.0
        rule_accuracy = 0.0
        if not (any(item in self.target for item in rule_keys)):
            covered_by_rule = self.__find_covered_examples(
                tdata, rule[PCR.rl_antecedent])

            if covered_by_rule:
                for target_attribute in self.target:
                    predicted_dict = {}
                    key_pred = max(
                        rule[PCR.rl_prototype][target_attribute],
                        key=rule[PCR.rl_prototype][target_attribute].get)
                    predicted_dict[target_attribute] = [key_pred]
                    covered_by_predicted = self.__find_covered_examples(
                        tdata, predicted_dict)

                    true_positives = list(
                        (set(covered_by_rule) & set(covered_by_predicted)))
                    false_positives = len(covered_by_rule) - len(
                        true_positives)
                    false_negatives = len(covered_by_predicted) - len(
                        true_positives)
                    true_negatives = (len(tdata)) - (false_positives +
                                                     false_negatives +
                                                     len(true_positives))
                    logging.warning("FP:{} FN:{} TP:{} TN:{} length:{}".format(
                        false_positives, false_negatives, len(true_positives),
                        true_negatives, len(tdata)))
                    class_error = ((false_positives + false_negatives) /
                                   (len(tdata)))
                    class_error = self.attributes_weights[
                        target_attribute] * class_error
                    accuracy = (
                        (len(true_positives)) + true_negatives) / (len(tdata))
                    #class_error = self.attributes_weights[target_attribute]*class_error
                    rule_class_error += class_error
                    rule_accuracy += accuracy
                rule_class_error = rule_class_error / len(self.target)
                rule_accuracy = rule_accuracy / len(self.target)
                #print("rule_class_error:{}".format(rule_class_error))
            else:
                rule_class_error = 1.0
        else:
            #print("This is an attribute target")
            rule_class_error = 1.0
            rule_accuracy = 0.0
        rule[PCR.rl_classify_error] = rule_class_error
        rule[PCR.rl_accuracy] = rule_accuracy
        return rule_class_error

    def __calculate_classification_error(self, tdata, rule, target_attrs,
                                         threshold):
        #print("Calculate class error")

        rule_keys = list(
            rule[PCR.rl_antecedent].keys())  # making it more readable
        #check if target is in the rule
        sum_errors = 0.0
        rule_accuracy = 0.0
        avg_classify_error = 0.0
        targets_errors = {}
        targets_accuracy = {}
        consequent = {}
        classify_error_by_tar_attr = {}
        if not (any(item in target_attrs for item in rule_keys)):
            covered_by_rule = self.__find_covered_examples(
                tdata, rule[PCR.rl_antecedent])

            if covered_by_rule:
                num_targets_now = 0
                for target_attribute in target_attrs:
                    num_targets_now += 1
                    predicted_ = {}
                    predicted_value = rule[
                        PCR.rl_prototype][0][target_attribute]
                    predicted_[target_attribute] = [predicted_value]
                    covered_by_predicted = self.__find_covered_examples(
                        tdata, predicted_)

                    true_positives = list(
                        (set(covered_by_rule) & set(covered_by_predicted)))
                    false_positives = len(covered_by_rule) - len(
                        true_positives)
                    false_negatives = len(covered_by_predicted) - len(
                        true_positives)
                    true_negatives = (len(tdata)) - (false_positives +
                                                     false_negatives +
                                                     len(true_positives))
                    logging.warning("FP:{} FN:{} TP:{} TN:{} length:{}".format(
                        false_positives, false_negatives, len(true_positives),
                        true_negatives, len(tdata)))
                    class_error = ((false_positives + false_negatives) /
                                   (len(tdata)))
                    #class_error = self.attributes_weights[target_attribute] * class_error
                    accuracy = (
                        (len(true_positives)) + true_negatives) / (len(tdata))
                    sum_errors += class_error
                    avg_classify_error = sum_errors / num_targets_now
                    targets_errors[target_attribute] = class_error
                    if avg_classify_error <= threshold:
                        consequent[target_attribute] = predicted_value
                    #rule_class_error += class_error
                    #rule_accuracy += accuracy
                #print("rule_class_error:{}".format(rule_class_error))
            else:
                rule_class_error = 1.0
        else:
            #print("This is an attribute target")
            rule_class_error = 1.0
            rule_accuracy = 0.0

        return avg_classify_error, consequent, targets_errors
#endregion


# region UNIQUE values, Relative Freq and the SELECTORS
# get all the unique values by column
    def __get_idxs_of_unique_values(self):
        idxs_of_unique_values= {}
        for col in self.column_list:
            col_val = {}
            uniq_value_idxs = {}
            for value in self.global_unique_values[col]:
                col_val[col]=[value]
                idxs = self.__find_covered_examples(self.training_data, col_val)
                uniq_value_idxs[value] = idxs
                #print(value,":idxs:",idxs)
            idxs_of_unique_values[col] = uniq_value_idxs

        return idxs_of_unique_values

    def __get_unique_values(self, recv_sample):
        u_values_dict={}
        for column in recv_sample:
            if self.td_dtypes[column] == PCR.nominal:
                u_values_dict[column] = recv_sample[column].unique().tolist()
        return u_values_dict

    def __calculate_relative_freq(self, a_sample):
        freq_col = {}
        for col in self.column_list:
            if self.td_dtypes[col] == PCR.nominal:
                freq = a_sample[col].value_counts()
                rel_freq = freq / len(a_sample)
                freq_col[col] = rel_freq.to_dict()
        return freq_col

    def __get_selectors(self, multi_direction):
        attribute_value_pair = []
        if multi_direction:
            columns_to_selector = self.column_list
        else:
            columns_to_selector = [col for col in self.column_list if col not in self.target]
        print("cols:{}".format(columns_to_selector))
        for col in columns_to_selector:
            logging.warning("now with col:{}".format(col))
            if self.td_dtypes[col] == PCR.nominal:
                for attribute_value in self.global_unique_values[col]:
                    attribute_value_pair.append({col: [attribute_value]})
            elif self.td_dtypes[col] == PCR.numeric:
                column_list = self.training_data[col].values
                percentiles = np.percentile(column_list, [20,40,60,80])
                logging.warning("Before sort percentiles:{}".format(percentiles))
                percentiles.sort()
                logging.warning("After sort percentiles:{}".format(percentiles))
                res = list(zip(percentiles, percentiles[1:])) # create the pairs for ranges
                for a, b in res:
                    attribute_value_pair.append({col: (a, b)})
        return attribute_value_pair

    def define_final_prototype(self, antecedent, current_prototype):
        final_prototype = {}
        rel_freq_ptyp = {}
        ptyp_attributes = list(current_prototype.keys())
        antecdt_attributes = list(antecedent.keys())
        ptyp_attributes = [attr for attr in ptyp_attributes if attr not in antecdt_attributes]

        print(antecdt_attributes, "->", ptyp_attributes)

        for ptyp_attr in ptyp_attributes:
            value = max(current_prototype[ptyp_attr],
                        key=current_prototype[ptyp_attr].get)
            final_prototype[ptyp_attr] = value
            rel_freq_ptyp[ptyp_attr] = current_prototype[ptyp_attr][value]

        packed_ptyp = [final_prototype, rel_freq_ptyp]
        return packed_ptyp


    def get_target_attributes(self,ptyp_rel_freq, max_targets=2):
        nw_target_attrs = sorted(ptyp_rel_freq,
                                 key=ptyp_rel_freq.get,
                                 reverse=True)[:max_targets]
        return nw_target_attrs
#endregion

# region SPECIALIZE RULES

    def __specialize_complex(self, complexes):
        specialized_complexes = []
        if not complexes:
            specialized_complexes = copy.deepcopy(self.selectors)
        else:
            for condition in complexes:
                for selector in self.selectors:
                    specifying_complex = copy.deepcopy(
                        condition)  # to not modify original
                    key = list(selector.keys(
                    ))[0]  # a selector is a dict with a single key = attribute
                    if key not in specifying_complex:  # if attribute is not in complex we will try to add it
                        specifying_complex[key] = selector[key]
                        # now we check if complex already exist in complexes set
                        flag = True
                        for special_complex in specialized_complexes:
                            if specifying_complex == special_complex:  # if one is same we set flag=False
                                flag = False
                        if flag:
                            specialized_complexes.append(specifying_complex)

        return specialized_complexes

#endregion

#region SEARCH HEURISTIC - Dissimilarity
    def __calculate_dissimilarity(self, candidate_rule_prototype):
        #print("Calculate dissimilarity")
        sum_dissim = 0.0
        for column in self.column_list:
            if self.td_dtypes[column] == "string":
                nom_sum = 0.0
                for u_val in self.global_unique_values[column]:
                    #print(u_val, self.td_prototype[column][u_val] , candidate_rule_prototype[column].get(u_val, 0.0))
                    nom_sum += (abs(self.td_prototype[column][u_val] -
                                             candidate_rule_prototype[column].get(u_val, 0.0)))
                    #print("diss sum so far:{}".format(nom_sum))
                pre_nominal_distance = (nom_sum/len(self.td_prototype[column]))
                nominal_distance = self.attributes_weights[column]*pre_nominal_distance
                #print("wo weights:{} average diss: {} attr:{}".format(pre_nominal_distance,nominal_distance, column))
                sum_dissim += nominal_distance
                #print("total sum diss:{} so far".format(sum_dissim))
            elif self.td_dtypes[column] == "numeric":
                numeric_distance = self.attributes_weights[column]*((abs(self.td_prototype[column][0] - candidate_rule_prototype[column]))/
                                    self.td_prototype[column][1])
                sum_dissim += numeric_distance

        dissimilarity = sum_dissim/len(self.column_list)

        return dissimilarity
#endregion

#region SEARCH HEURISTIC - Distance
    def __calculate_rule_distance(self, candidate_rule_cov_indxes):
        sum_jd = 0.0
        for rule in self.rule_set:
            union = (set(rule[PCR.rl_cov_idxs]) | set(candidate_rule_cov_indxes))
            inter = (set(rule[PCR.rl_cov_idxs]) & set(candidate_rule_cov_indxes))
            jaccard_distance = 1 - (len(inter)/len(union))
            #jaccard_distance = (len(inter)/len(union))
            sum_jd += jaccard_distance
        total_jd = sum_jd/len(self.rule_set)
        #print("total_jd:{}".format(total_jd))
        return total_jd
#endregion

#region SEARCH HEURISTIC - Coverage
    def __find_covered_examples(self, sample_to_cover, rule_complex):
        numeric_complex = {}
        nominal_complex = {}

        numeric_cov_idxs = []
        nominal_cov_idxs = []
        covered_indexes = []

        for col in self.column_list:
            if col in rule_complex:
                #print("complex key value:{}".format(rule_complex[col]))
                if self.td_dtypes[col] == "string":
                    nominal_complex[col] = rule_complex[col]
                elif self.td_dtypes[col] == "numeric":
                    numeric_complex[col] = rule_complex[col]

        #print("Numeric Complex:{}".format(numeric_complex))
        #print("Nominal Complex:{}".format(nominal_complex))

        if (len(nominal_complex)) > 0 :
            nominal_attributes = list(nominal_complex.keys())
            nominal_mask = sample_to_cover[nominal_attributes].isin(nominal_complex).all(1)
            nominal_coverage = sample_to_cover[nominal_mask]
            nominal_cov_idxs = nominal_coverage.index.values.tolist()
            #print("Nominal indexes:{}".format(nominal_cov_idxs))
            covered_indexes = nominal_cov_idxs

        if (len(numeric_complex)) > 0 :
            for key, pair in numeric_complex.items():
                numeric_mask=sample_to_cover[key].isin(np.arange(pair[0],pair[1]))
                numeric_coverage= sample_to_cover[numeric_mask]
                numeric_idxs =numeric_coverage.index.values.tolist()
                numeric_cov_idxs.append(numeric_idxs)


        if (not covered_indexes) & len(numeric_cov_idxs):
            covered_indexes = numeric_cov_idxs[0]

        for idx_list in numeric_cov_idxs:
            covered_indexes = list(set (covered_indexes) & set(idx_list))

        #print("Covered indexes before return:{}".format(covered_indexes))
        return covered_indexes

    def __calculate_coverage(self, candidate_rule_cov_indxes):
        tdata_cov_weights = [self.td_weights[idx] for idx in self.all_td_indexes]
        crule_cov_weights = [self.td_weights[idx] for idx in candidate_rule_cov_indxes]
        total_tdata_cov_weights = np.sum(tdata_cov_weights)
        total_crule_cov_weights = np.sum(crule_cov_weights)
        relative_cov = total_crule_cov_weights/total_tdata_cov_weights
        return relative_cov

#endregion



#region MAIN  to call PCR algorithm
if __name__ == '__main__':
    ##print("Hola Python")
    logging.basicConfig(filename='Multi-dir.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
    #iris_data = data('iris')
    '''
    URL="https://www2.cs.arizona.edu/classes/cs120/fall17/ASSIGNMENTS/assg02/Pokemon.csv"
    pokemon_data = pd.read_csv(URL)
    print(type(pokemon_data))
    #pokemon_data.drop(['Name','#'], 1, inplace=True)
    pokemon_data = pokemon_data.loc[:, [
        "Type 1", "Type 2", "Generation", "Legendary", "Attack", "Defense",
        "Speed"
    ]]
    print(pokemon_data.head(5))
    print("size:{}".format(len(pokemon_data)))
    mask = pokemon_data.applymap(type) != bool
    d = {True: 'TRUE', False: 'FALSE'}
    pokemon_data = pokemon_data.where(mask, pokemon_data.replace(d))
    pokemon_data = pokemon_data.dropna()
    print("size:{}".format(len(pokemon_data)))
    training_data = pokemon_data.sample(414)
    test_data = pokemon_data.sample(200)
    #target = ["Type 1"]
    '''
    #arthritis_data = data('Arthritis')
    #training_data = arthritis_data.sample(80)
    #test_data = arthritis_data.sample(50)
    #target = ["Treatment"]
    #iris_data = data('iris')
    #training_data = iris_data.sample(80)
    #test_data = arthritis_iris_datadata.sample(50)
    #target = ["Species"]
    titanic_data = data('titanic')
    training_data = titanic_data.sample(1000)
    test_data = titanic_data.sample(500)
    validation_data = titanic_data.sample(500)
    target = ["survived"]
    logging.warning(training_data)
    PCR_instance = PCR(training_data=training_data,
                       validation_data=validation_data,
                       test_data=test_data,
                       target = target,
                       multi_direction=True)
    #PCR_instance.test_find_best_complex()
    #PCR_instance.test_find_best_complex_v02()
    #PCR_instance.test_find_best_complex_v03()



#endregion
