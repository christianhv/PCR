#region import
import copy
from io import SEEK_CUR
from typing import Dict, Any
from numpy.lib.ufunclike import _dispatcher
from pydataset import data
from math import log
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import operator
import time
from operator import itemgetter
import logging

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
    mty_rep = "replace"
    mty_sub = "substract"
    mty_mul = "multiplication"

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
    srch_heu_Other = "Other"

    # examples elimination type
    mmthd_err_weight_cov = "Err-Weight-Covering"
    mmthd_std_cov = "Standard-Covering"

    #region initializing PCR
    def __init__(self, training_data, test_data, target, target_weight=1.0):
        #logging.basicConfig(filename='test.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        self.training_data = training_data.sort_index()
        print(self.training_data.index)
        self.training_data = self.training_data.reset_index(drop=True)
        logging.warning("SORTED:\n{}".format(self.training_data))
        self.test_data = test_data
        logging.warning("TEST:\n{}".format(self.test_data))
        # self.highest_idx = np.asscalar(self.training_data.tail(1).index.values.astype(int))
        print(self.training_data.index)

        print("df:",self.training_data.index[[1,3]])
        #self.training_data.drop(self.training_data.index[[1,3]], inplace=True)

        print(self.training_data)
        #return None
        self.all_td_indexes = self.training_data.index.values.tolist()
        self.highest_idx = self.all_td_indexes[len(self.all_td_indexes)-1]

        if type(target) != list:
            print("not a list")
            self.target = [target]
        else:
            self.target = target
        self.no_target = len(self.target)

        self.column_list = list(self.training_data.columns)
        self.td_dtypes = self.__set_data_types()
        self.global_unique_values = self.__get_unique_values(self.training_data)

        #All the weights need to tune the algorithm
        self.alpha_coverage_hweight = 1.0
        self.beta_distance_hweight = 0.0  #in original experiments is not used
        self.gamma_dissimilarity_hweight = 0.0 #in original experiments is not used

        self.tau_target_weight = target_weight

        self.zeta_covering_err_weight = 0.4
        self.eps_cov_weight_err_thld = 0.5
        self.minimum_covered = 3

        self.maximal_number_of_rules = 100

        ### The structures below need some of the structures above
        # setting the weights
        self.attributes_weights = self.__set_default_attributes_weights()
        self.td_weights = self.__set_td_weights()
        print(len(self.td_weights))
        print(self.td_weights)
        self.selectors = self.__get_selectors(multi_direction=False)
        logging.warning("SELECTORS:{}".format(self.selectors))
        #setting the generic dispersion
        #we need the type of data and the column list
        #self.dispersion_off = self.__calculate_dispersion_of(self.training_data)[0]
        self.dispersion_off = 0
        #setting the generic prototype
        self.td_prototype = self.__set_td_prototype()
        self.rule_set = []
        ci = [1,1200]
        self.target_val_idxs = {}
        for k in self.target:
            temp = {}
            temp2 = {}
            for value in self.global_unique_values[k]:
                temp[k]=[value]
                idxs = self.__find_covered_examples(self.training_data, temp)
                temp2[value] = idxs
                print(value,":idxs:",idxs)
            self.target_val_idxs[k] = temp2

        logging.warning("target indexes by value")
        logging.warning(self.target_val_idxs)
        logging.warning(self.dispersion_off)
        logging.warning(self.td_prototype)

#endregion

#region some tests
    def check_rule(self):
        rule = {'sex': ['women'], 'class': ['1st class'], 'age': ['adults']}

        for column in self.column_list:
            print("column:{} value rule:{}".format(column, rule.get(column)))


    def test_modify_learning_set(self):
        print("modifying check")
        indexes = self.training_data.index.values.tolist()
        targets_covered = [1] * len(indexes)
        #print(indexes, "\n", targets_covered)
        #print(self.td_weights)
        self.__modify_learning_set(indexes, targets_covered)
        print(self.td_weights)

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
                    new_candidate = self.__calculate_search_heuristic(learning_set, komplex)
                    newstar_as_rule.append(new_candidate)
                    new_heuristic_value = new_candidate[PCR.rl_srch_heu_value]
                    if new_heuristic_value>poorest_heuristic_value:
                        best_candidates.append(new_candidate)
                        best_candidates = sorted(best_candidates, key=itemgetter(PCR.rl_srch_heu_value), reverse=True)

                        if len(best_candidates)>limit_best:
                            best_candidates.pop()
                logging.warning("best candidate after popping")
                for bx in best_candidates:
                    logging.warning("{}".format(bx[PCR.rl_antecedent], bx[PCR.rl_srch_heu_value]))
                newstar_as_rule = sorted(newstar_as_rule, key=itemgetter(PCR.rl_srch_heu_value), reverse=True)
                newstar = [komplex[PCR.rl_antecedent] for komplex in newstar_as_rule]
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

    def test_find_best_complex(self, limit_rule_set=5, error_threshold=0.4):
        stop_criterion = True
        poorest_ruleset_error = 1.0
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
                logging.warning("{} - {} - {} - pro:{}".format(bc[PCR.rl_antecedent], bc[PCR.rl_srch_heu_value], bc[PCR.rl_classify_error], bc[PCR.rl_prototype]))

            logging.warning("compare: {} worst:{}".format(
                best_rule[PCR.rl_classify_error], poorest_ruleset_error))
            #Check if the best candidate is best than the worst in the rule set
            if best_rule[PCR.rl_classify_error] < poorest_ruleset_error or best_rule[PCR.rl_classify_error]<error_threshold:
                self.rule_set.append(best_rule)
                self.rule_set = sorted(self.rule_set, key=itemgetter(PCR.rl_classify_error), reverse=False)
                indexes_fweights = list(best_rule[PCR.rl_correct_cov].keys())
                new_weights = list(best_rule[PCR.rl_correct_cov].values())
                self.__modify_learning_set(indexes_fweights, new_weights)
                #if the rule set is bigger than the limit we take off the worst
                if len(self.rule_set) > limit_rule_set:
                    self.rule_set.pop()
                poorest_ruleset_error = self.rule_set[-1][PCR.rl_classify_error]
            else:
                stop_criterion = False

            if self.training_data.empty :
                stop_criterion = False

            logging.warning("RULE SET")
            for rule in self.rule_set:
                logging.warning("{} - error:{} - sh:{} - pro:{}".format(rule[PCR.rl_antecedent], rule[PCR.rl_classify_error], rule[PCR.rl_srch_heu_value], rule[PCR.rl_prototype]))

    def test_find_best_complex_v02(self, limit_rule_set=5, error_threshold=0.4):
        stop_criterion = True
        poorest_ruleset_error = 1.0
        while stop_criterion:
            best_candidates = self.__find_best_complex(self.training_data,limit_best=8, limit_beam=12)
            for bc in best_candidates:
                self.__calculate_rule_classify_error(self.test_data, bc)
                logging.warning("V0.2 {} - {} - {} - pro:{}".format(bc[PCR.rl_antecedent], bc[PCR.rl_srch_heu_value], bc[PCR.rl_classify_error], bc[PCR.rl_prototype]))

            #Check if the best candidate is best than the worst in the rule set
            self.rule_set = [*self.rule_set, *best_candidates ]
            self.rule_set = sorted(self.rule_set, key=itemgetter(PCR.rl_classify_error), reverse=False)
            self.rule_set = self.rule_set[0:limit_rule_set]
            new_worst = self.rule_set[-1][PCR.rl_classify_error]

            for rule in self.rule_set:
                indexes_fweights = list(rule[PCR.rl_correct_cov].keys())
                new_weights = list(rule[PCR.rl_correct_cov].values())
                self.__modify_learning_set(indexes_fweights, new_weights)

            if new_worst == poorest_ruleset_error or self.training_data.empty:
                stop_criterion = False

            poorest_ruleset_error = new_worst

            logging.warning("RULE SET")
            for rule in self.rule_set:
                logging.warning("{} - error:{} - sh:{} - pro:{}".format(rule[PCR.rl_antecedent],
                                                                        rule[PCR.rl_classify_error],
                                                                        rule[PCR.rl_srch_heu_value],
                                                                        rule[PCR.rl_prototype]))

        self.rule_set = [rule for rule in self.rule_set if rule[PCR.rl_classify_error] < error_threshold]

    def test_specialization(self):
        print("test specialization")
        complexes_to_specialize = []
        empty = False
        best_five= []
        best_class_error =[]
        exist_candidate = True
        while not self.training_data.empty or exist_candidate:
            print("Out second loop")
            empty= False
            best_five= []
            best_class_error =[]
            while not empty:
                specialized = self.__specialize_complex(complexes_to_specialize)

                if not specialized:
                    empty = True
                else:
                    for sp in specialized:
                        print("sp", sp)
                        rule_packed = self.__calculate_search_heuristic(self.training_data, sp)
                        logging.warning("just info:{} - {}".format(rule_packed[PCR.rl_antecedent],rule_packed[PCR.rl_srch_heu_value] ))
                        if len(rule_packed[PCR.rl_cov_idxs]) > self.minimum_covered:
                            self.__calculate_rule_classify_error(self.test_data, rule_packed)
                            best_five.append(rule_packed)
                            best_class_error.append(rule_packed)
                        else:
                            logging.warning("Rule: {} no covered minimum".format(rule_packed[PCR.rl_antecedent]))
                        print("|", rule_packed[PCR.rl_antecedent], "\n", len(best_five))


                        best_five = sorted(best_five, key=itemgetter(PCR.rl_srch_heu_value), reverse=True)
                        best_class_error = sorted(best_class_error, key=itemgetter(PCR.rl_classify_error), reverse=False)
                        best_five = best_five[0:5]
                        best_class_error = best_class_error[0:3]




                        logging.warning("By SH")
                        for b in best_five:
                            logging.warning(b)

                        logging.warning("By CL error")
                        for bc in best_class_error:
                            indexes_fweights = list(bc[PCR.rl_correct_cov].keys())
                            new_weights = list(bc[PCR.rl_correct_cov].values())
                            if not self.training_data.empty:
                                self.__modify_learning_set(indexes_fweights, new_weights)
                            logging.warning(bc)


                complexes_to_specialize = [rule[PCR.rl_antecedent] for rule in best_five]



    def test_delete(self):
        for i in range(0, 8):
            print(i)
            #ind = (self.training_data.head(1).index.values.tolist())
            ind = (self.training_data.head(1).index.item())
            print(type(ind))
            #ind += i
            a_list = list(range(ind, ind+10))
            to = [x for x in a_list if x<(ind+5)]
            print(a_list, to)
            self.test_filtering(to)


    def test_filtering(self, to_delete):
        #new_Learning = self.training_data.loc[[3,4],:]
        #new_Learning = self.training_data.drop(to_delete, axis=0)
        logging.warning("index to delete:\n{}".format(to_delete))
        #self.training_data = self.training_data.drop(self.training_data.index[to_delete])
        #print("df:",self.training_data.index[to_delete].item())

        #rows = self.training_data.index[to_delete]
        #print(rows)

        try:
            self.training_data.drop(index = to_delete, inplace=True)
            print(self.training_data)
            logging.warning("correct deletion:\n{}".format(self.training_data))
        except IndexError:
            print("IndexError")
        #print(self.training_data)


#endregion



#region DEFAULT STRUCTURES(weights, td_prototype, dtypes)
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
    def __modify_learning_set(self, indexes, targets_covered_by_new_rule_in_set):
        modifier = [self.zeta_covering_err_weight - 1]*len(targets_covered_by_new_rule_in_set) #the covering weight times the length of the covered examples
        logging.warning("MLS - covering weight:{}".format(self.zeta_covering_err_weight))
        logging.warning("MLS - targets:{}".format(type(targets_covered_by_new_rule_in_set)))
        uno = [1]*len(targets_covered_by_new_rule_in_set) # a vector with ones times the length covered examples
        no_targets = [self.no_target] * len(targets_covered_by_new_rule_in_set) # vector with number of targets times the length of covered_examples
        targets_covered_by_new_rule_in_set = list(map(operator.truediv,targets_covered_by_new_rule_in_set, no_targets)) # divide all the covered targets in each example by the number of targets
        targets_covered_by_new_rule_in_set =list(map(operator.mul, targets_covered_by_new_rule_in_set, modifier)) #multiply the modifier by the targets covered/by number of targets
        targets_covered_by_new_rule_in_set = list(map(operator.add, targets_covered_by_new_rule_in_set, uno)) # add 1 to the result before, most of the time lower than 1 if zero then the final modifier is zero and example is immediately deleted
        self.modify_examples_weights(indexes, targets_covered_by_new_rule_in_set , PCR.mty_mul) # the true modification of the value
        to_delete = [idx for idx in indexes if self.td_weights[idx] < self.eps_cov_weight_err_thld] # find the examples to delet
        self.modify_examples_weights(to_delete, [0], PCR.mty_rep) #since these values are lower than the threshold we set them in zero

        current_indexes = self.training_data.index.values.tolist() # we try to find if the examples still exist in the training set and were not deleted by other rule in the iteration
        true_delete = [idx for idx in to_delete if idx in current_indexes] # we create a list the indexes that still exist and need to be deleted
        logging.warning("true delete:{}".format(true_delete))
        self.test_filtering(true_delete) # deleting the indexes
        #self.__drop_zero_weight_examples(indexes)


    def modify_examples_weights(self, indexes, new_weights, manner):
        #print("Modify:", indexes, new_weights)
        if len(new_weights) == 1:
            if manner ==  PCR.mty_add:
                for index in indexes:
                    self.td_weights[index] += new_weights[0]
            elif manner == PCR.mty_rep:
                for index in indexes:
                    #print("i:{} len:{}".format(index, len(self.td_weights)))
                    self.td_weights[index] = new_weights[0]
            elif manner == PCR.mty_sub:
                for index in indexes:
                    self.td_weights[index] -= new_weights[0]
            elif manner == PCR.mty_mul:
                for index in indexes:
                    self.td_weights[index] *= new_weights[0]
            else:
                print("Indicated manner not found")
        elif (len(new_weights) == len(indexes)):
            #print("they are")
            if manner ==  PCR.mty_add:
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] += replacement
            elif manner == PCR.mty_rep:
                print("Replacement")
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] = replacement
            elif manner == PCR.mty_sub:
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] -= replacement
            elif manner == PCR.mty_mul:
                for (index, replacement) in zip(indexes, new_weights):
                    self.td_weights[index] *= replacement
            else:
                print("Indicated manner not found")
        else:
            print("Arguments do not fulfill requirements")

#endregion

#region drop rows [does not work]
    def __drop_zero_weight_examples(self, indexes):
        print(indexes)
        self.training_data.drop(indexes,inplace=True)

        #df.index[[1,3]]
        print("td weights",len(self.td_weights))
#endregion

#region total SEARCH HEURISTIC


    def __calculate_search_heuristic(self,learning_set, rule_complex):

        rule_packed = {}
        print("[rule_complex]:{}".format(rule_complex))
        covered_indexes = self.__find_covered_examples(learning_set, rule_complex)
        print("|Covered|", covered_indexes)
        if covered_indexes:
            #print("COVERED:{}".format(covered_indexes))
            rule_coverage = self.__calculate_coverage(covered_indexes)
            rule_dispersion, rule_prototype = self.__calculate_dispersion_of(learning_set.loc[covered_indexes])
            rule_dissimilarity = self.__calculate_dissimilarity(rule_prototype)
            if len(self.rule_set):
                rule_distance = (self.__calculate_rule_distance(covered_indexes))
            else:
                rule_distance = 0

            sh_value = rule_coverage + (self.dispersion_off - rule_dispersion) + rule_dissimilarity + rule_distance
            logging.warning("COMPLEX:{}".format(rule_complex))
            logging.warning("COVERAGE:{}".format(rule_coverage))
            logging.warning("LEN COVERAGE:{}".format(len(covered_indexes)))
            logging.warning("DISPERSION:{}".format(rule_dispersion))
            logging.warning("RULE PROTOTYPE:{}".format(rule_prototype))
            logging.warning("DISSIMILARITY:{}".format(rule_dissimilarity))
            logging.warning("DISTANCE:{}".format(rule_distance))
            logging.warning("sh_value:{}".format(sh_value))
            #cov_to_loop = copy.deepcopy(covered_indexes)


            rule_packed = {PCR.rl_antecedent: rule_complex, PCR.rl_srch_heu_value: sh_value, PCR.rl_classify_error: 1.0,
                           PCR.rl_accuracy: 0.0, PCR.rl_coverage: rule_coverage,
                           PCR.rl_prototype: rule_prototype, PCR.rl_new_weights:{}, PCR.rl_cov_idxs: covered_indexes,
                           PCR.rl_correct_cov:{k:0 for k in covered_indexes}}

            for target_attribute in self.target:
                value = max(rule_prototype[target_attribute], key = rule_prototype[target_attribute].get)
                print(target_attribute,":",value)
                correct_covered = list((set(covered_indexes) & set(self.target_val_idxs[target_attribute][value])))
                print("correct", correct_covered)
                #print("correct", self.training_data.loc[correct_covered[0]])
                for i in correct_covered:
                    rule_packed[PCR.rl_correct_cov][i]+=1.0
        else:
            rule_packed = {PCR.rl_antecedent: rule_complex, PCR.rl_srch_heu_value: 0.0, PCR.rl_classify_error: 1.0, PCR.rl_accuracy: 0.0, PCR.rl_prototype: self.td_prototype,
                           PCR.rl_new_weights:{}, PCR.rl_cov_idxs: [], PCR.rl_correct_cov:{}}

            #print("correct", rule_packed[PCR.correct_cov], "\n\n")
        return rule_packed
    
    def __define_final_prototype(self,current_prototype):
        print("define final prototype")

# region Calculate Classification Error
    def __calculate_rule_classify_error(self, tdata, rule):
        #print("Calculate class error")

        rule_keys = list(rule[PCR.rl_antecedent].keys()) # making it more readable
        #check if target is in the rule
        rule_class_error = 0.0
        rule_accuracy = 0.0
        if not(any (item in self.target for item in rule_keys)):
            covered_by_rule = self.__find_covered_examples(tdata, rule[PCR.rl_antecedent])

            if covered_by_rule:
                for target_attribute in self.target:
                    predicted_dict={}
                    key_pred = max(rule[PCR.rl_prototype][target_attribute], key = rule[PCR.rl_prototype][target_attribute].get)
                    predicted_dict[target_attribute] = [key_pred]
                    covered_by_predicted = self.__find_covered_examples(tdata, predicted_dict)

                    true_positives = list((set(covered_by_rule) & set(covered_by_predicted)))
                    false_positives = len(covered_by_rule) - len(true_positives)
                    false_negatives = len(covered_by_predicted) - len(true_positives)
                    true_negatives = (len(tdata)) - (false_positives + false_negatives + len(true_positives))
                    print("FP:{} FN:{} TP:{} TN:{} length:{}".format(false_positives, false_negatives, len(true_positives), true_negatives,len(tdata)))
                    class_error = ((false_positives+false_negatives)/(len(tdata)))
                    class_error = self.attributes_weights[target_attribute]*class_error
                    accuracy =  ((len(true_positives))+true_negatives)/(len(tdata))
                    #class_error = self.attributes_weights[target_attribute]*class_error
                    rule_class_error += class_error
                    rule_accuracy += accuracy
                rule_class_error = rule_class_error/len(self.target)
                rule_accuracy = rule_accuracy/len(self.target)
                #print("rule_class_error:{}".format(rule_class_error))
            else:
                rule_class_error = 1.0
        else:
            #print("This is an attribute target")
            rule_class_error=1.0
            rule_accuracy=0.0
        rule[PCR.rl_classify_error] = rule_class_error
        rule[PCR.rl_accuracy] = rule_accuracy
        return rule_class_error
#endregion

# region UNIQUE values, Relative Freq and the SELECTORS
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
                freq_col[col] = rel_freq.to_dict();
        return freq_col

    def __get_selectors(self, multi_direction):
        attribute_value_pair = []
        if multi_direction:
            columns_to_selector = self.column_list
        else:
            columns_to_selector = [col for col in self.column_list if col not in self.target]
        print("cols:{}".format(columns_to_selector))
        for col in columns_to_selector:
            if self.td_dtypes[col] == PCR.nominal:
                for attribute_value in self.global_unique_values[col]:
                    attribute_value_pair.append({col: [attribute_value]})
            elif self.td_dtypes[col] == PCR.numeric:
                column_list = self.training_data[col].values
                percentiles = np.percentile(column_list, [25, 50, 75])
                percentiles.sort()
                res = list(zip(percentiles, percentiles[1:])) # create the pairs for ranges
                for a, b in res:
                    attribute_value_pair.append({col: (a, b)})
        return attribute_value_pair
#endregion

# region SPECIALIZE RULES
    def __specialize_complex(self, complexes):
        specialized_complexes = []
        if not complexes:
            specialized_complexes = copy.deepcopy(self.selectors)
        else:
            for condition in complexes:
                for selector in self.selectors:
                    specifying_complex = copy.deepcopy(condition)  # to not modify original
                    key = list(selector.keys())[0]  # a selector is a dict with a single key = attribute
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
        print("total_jd:{}".format(total_jd))
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

#region SEARCH HEURISTIC - Dispersion

    def __calculate_coefficient_of_variation(self, avg, std_deviation):
        coeff_var = std_deviation / avg
        if coeff_var > 1:
            coeff_var = 1
        return coeff_var

    def __calculate_dispersion_of(self, covered_examples):
        # calculate the prototype of the attributes
        # relative frequencies
        # 1. first the value counts for each value if attribute is nominal
        # 2. Average of the attribute if is numeric - no in fact is variance
        #   The variance is calculated at the same time that the relative frequencies
        # We need a dictionary to save the attributes prototypes

        unique_values = self.__get_unique_values(covered_examples)
        nominal_prototype = self.__calculate_relative_freq(covered_examples)
        rule_prototype = nominal_prototype

        # in fact it is the covered examples by the rule
        # relative frequency of each attribute

        # The dispersion will be a dictionary of each attributes
        # with the average of the distance of each example to the prototype
        # the prototype of each coverage examples is the relative frequency that we already calculated
        # for each attribute {att1:rel_fe1, ... attn: rel_fen}
        # first we create the variable to hold the dispersion of this rule
        attributes_total_dispersion = {}
        # here we will save the total sum of each column
        all_attributes_distances = {}
        # since the formula says that each example prototype is 0 for any
        # other possible value in the column. So, for each unique value in
        # the coverage examples we can already calculate how much will be the sum
        # then we can just search de value and sum up for all the examples in that
        # specific column and compute the average.
        # missing the weights
        #print("WOW");

        for col in self.column_list:
            if self.td_dtypes[col] == "string":
                # for this first step we need to get the relative frequencies
                # and the unique values
                attribute_freq_rel = nominal_prototype[col]
                attribute_values = unique_values[col]
                # no necessary it is always one
                # suma = sum(temp_col_dict.values())
                attribute_distances = {}
                # Going through each column
                for val in attribute_values:
                    # Here we get the column relative frequencies
                    # The if is if there is more than one unique possible value
                    # If there is just one value we do not need to calculate anything the value is 1
                    attr_value_freq_rel = attribute_freq_rel.get(val)
                    attr_distance_valk = 2 * np.abs(1 - attr_value_freq_rel)
                    attribute_distances.update({val: attr_distance_valk})
                    #print("total_per_val: {} freq:{} value:{}".format(attr_value_sum, attr_value_freq_rel, val))
                all_attributes_distances.update({col: attribute_distances})

        for col in self.column_list:
            #print("### {} @@@@".format(col))
            avg_col = 0.0
            std_col = 0.0
            attr_dispersion = 0.0
            if self.td_dtypes[col] == "string" and len(self.global_unique_values[col])>1:
                #print("sumac: {}".format(all_attributes_sums[col]))
                attr_distances_values = all_attributes_distances[col]
                col_discrete_vals = covered_examples[col].tolist()

                distances = []

                # Now we only search the result/sum/computation already done when the
                # attribute gets the value that it gets
                #for row_value in col_discrete_vals:
                #print(attribute_sums_values.get(row_value))
                #    distances.append(attr_distances_values[row_value])
                #    disper_sum += attr_distances_values.[row_value]
                distances = [ attr_distances_values[row_value ] for row_value in col_discrete_vals]
                attr_L = len(self.global_unique_values[col])
                sum_total_distances = sum(distances)
                twice_N = 2*(len(distances))
                attr_dispersion = (sum_total_distances * attr_L) / (twice_N * (attr_L-1))


            elif self.td_dtypes[col] == "numeric":
                # When is numeric we calculate the average
                # and the standard deviation
                # and it is used in the dispersion calculation
                # the variance is already normalized
                avg_col = covered_examples[col].mean()
                std_col = covered_examples[col].std()
                rule_prototype[col] = avg_col
                attr_dispersion = 0.0

                if avg_col > 0:
                    attr_dispersion = self.__calculate_coefficient_of_variation(avg_col, std_col)

            #print("attribute dispersion:{}".format(attr_dispersion))
            total_attr_dispersion = self.attributes_weights[col] * attr_dispersion
            #print("after weighted:{}".format(total_attr_dispersion))
            #print("Avg: {} coeff var: {}".format(avg_col, coeff_var_col))
            attributes_total_dispersion.update({col: total_attr_dispersion})
            # In fact the dispersion
        dispersion = (sum(attributes_total_dispersion.values())) / len(attributes_total_dispersion)
        return dispersion, rule_prototype
#endregion

#region MAIN  to call PCR algorithm
if __name__ == '__main__':
    ##print("Hola Python")
    titanic_data = data('titanic')
    #print("size::{}".format(len(titanic_data)))
    logging.basicConfig(filename='titanicTest03.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

    temp_data = data('titanic')
    #temp_data = data('Arthritis')
    #temp_data = data('iris')
    #print("size2::{}".format(len(temp_data)))
    logging.warning("TODO: \n{}".format(temp_data))
    #al = [1,2]
    #logging.warning(temp_data.loc[al])
    # training_data = titanic_data.sample(800)
    # test_data = titanic_data.sample(100)
    training_data = temp_data.sample(1000)
    #training_data = training_data.loc[:,['Treatment','Sex',  'Age', 'Improved']]

    logging.warning(training_data)
    test_data = temp_data.sample(600)

    #print("Training Data: \n{}".format(training_data))
    #print(list(training_data.head(3).index))
    target = ["survived"]
    #target = ["Treatment"]
    #target = "Species"

    PCR_instance = PCR(training_data, test_data, target)
    PCR_instance.test_find_best_complex()
    #PCR_instance.test_specialization()

    #PCR_instance.test_delete()
    #PCR_instance.train_model(3)
    #result = PCR_instance.fit_model()

    # PCR_instance.set_attribute_weight(target, 35)
    #lo = PCR_instance.target_val_idxs["survived"]["yes"][0:3]
    #PCR_instance.test_filtering(lo)
    #PCR_instance.train_model(5)
    #PCR_instance.test_modify_learning_set()


#endregion 