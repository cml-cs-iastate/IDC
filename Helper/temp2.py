# import numpy as np
# import pickle as pkl
#
# from src.WGA.wgaSolver import post_processing
#
# with open('../Output/Folds/fold_1/IF.pkl', 'rb') as f1, \
#         open('../Output/Folds/fold_2/IF.pkl', 'rb') as f2, \
#         open('../Output/Folds/fold_3/IF.pkl', 'rb') as f3, \
#         open('../Output/Folds/fold_4/IF.pkl', 'rb') as f4, \
#         open('../Output/Folds/fold_5/IF.pkl', 'rb') as f5, \
#         open('../Output/Folds/fold_6/IF.pkl', 'rb') as f6, \
#         open('../Output/Folds/fold_7/IF.pkl', 'rb') as f7, \
#         open('../Output/Folds/fold_8/IF.pkl', 'rb') as f8, \
#         open('../Output/Folds/fold_9/IF.pkl', 'rb') as f9, \
#         open('../Output/Folds/fold_10/IF.pkl', 'rb') as f10,\
#         open('../Output/Folds/fold_11/IF.pkl', 'rb') as f11, \
#         open('../Output/Folds/fold_12/IF.pkl', 'rb') as f12,\
#         open('../Output/Folds/fold_13/IF.pkl', 'rb') as f13, \
#         open('../Output/Folds/fold_14/IF.pkl', 'rb') as f14:
#
#     fold_1 = pkl.load(f1)
#     fold_2 = pkl.load(f2)
#     fold_3 = pkl.load(f3)
#     fold_4 = pkl.load(f4)
#     fold_5 = pkl.load(f5)
#     fold_6 = pkl.load(f6)
#     fold_7 = pkl.load(f7)
#     fold_8 = pkl.load(f8)
#     fold_9 = pkl.load(f9)
#     fold_10 = pkl.load(f10)
#     fold_11 = pkl.load(f11)
#     fold_12 = pkl.load(f12)
#     fold_13 = pkl.load(f13)
#     fold_14 = pkl.load(f14)
#
# with open('../Output/Folds/fold_1/count_IF.pkl', 'rb') as f1, \
#         open('../Output/Folds/fold_2/count_IF.pkl', 'rb') as f2, \
#         open('../Output/Folds/fold_3/count_IF.pkl', 'rb') as f3, \
#         open('../Output/Folds/fold_4/count_IF.pkl', 'rb') as f4, \
#         open('../Output/Folds/fold_5/count_IF.pkl', 'rb') as f5, \
#         open('../Output/Folds/fold_6/count_IF.pkl', 'rb') as f6, \
#         open('../Output/Folds/fold_7/count_IF.pkl', 'rb') as f7, \
#         open('../Output/Folds/fold_8/count_IF.pkl', 'rb') as f8, \
#         open('../Output/Folds/fold_9/count_IF.pkl', 'rb') as f9, \
#         open('../Output/Folds/fold_10/count_IF.pkl', 'rb') as f10, \
#         open('../Output/Folds/fold_11/count_IF.pkl', 'rb') as f11, \
#         open('../Output/Folds/fold_12/count_IF.pkl', 'rb') as f12,\
#         open('../Output/Folds/fold_13/count_IF.pkl', 'rb') as f13, \
#         open('../Output/Folds/fold_14/count_IF.pkl', 'rb') as f14:
#
#     fold_1_count = pkl.load(f1)
#     fold_2_count = pkl.load(f2)
#     fold_3_count = pkl.load(f3)
#     fold_4_count = pkl.load(f4)
#     fold_5_count = pkl.load(f5)
#     fold_6_count = pkl.load(f6)
#     fold_7_count = pkl.load(f7)
#     fold_8_count = pkl.load(f8)
#     fold_9_count = pkl.load(f9)
#     fold_10_count = pkl.load(f10)
#     fold_11_count = pkl.load(f11)
#     fold_12_count = pkl.load(f12)
#     fold_13_count = pkl.load(f13)
#     fold_14_count = pkl.load(f14)
#
# IF = dict()
# count_IF = dict()
# average_IF = dict()
#
# for category in {1, 2, 4, 5}:
#     IF.setdefault(category, dict())
#     count_IF.setdefault(category, dict())
#     # ------------------------------------------------------------------------------------------------------------------
#     for word, value in fold_1[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_2[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_3[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_4[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_5[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_6[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_7[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_8[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_9[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_10[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_11[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_12[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_13[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_14[category].items():
#         IF[category][word] = IF[category].setdefault(word, 0) + value
#
#     # ------------------------------------------------------------------------------------------------------------------
#     for word, value in fold_1_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_2_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_3_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_4_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_5_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_6_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_7_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_8_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_9_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_10_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_11_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_12_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_13_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
#     for word, value in fold_14_count[category].items():
#         count_IF[category][word] = count_IF[category].setdefault(word, 0) + value
#
# # get the average importance values of the interpretation_features
# for category, sub_dict in IF.items():
#     for word, value in sub_dict.items():
#         average_IF.setdefault(category, dict())
#         average_IF[category][word] = np.divide(value, count_IF[category][word])
#
#
# post_processed_if_dict = post_processing(average_IF, count_IF)
#
# for category, dic in post_processed_if_dict.items():
#     maxi = max([value for word, value in dic.items()])
#     for word, value in dic.items():
#         dic[word] = dic[word] / maxi
#
#
# for category, dic in post_processed_if_dict.items():
#     print(category, len(dic))
#
#
# with open('IF.pkl', 'wb') as f:
#     pkl.dump(IF, f, pkl.HIGHEST_PROTOCOL)
# with open('count_IF.pkl', 'wb') as f:
#     pkl.dump(count_IF, f, pkl.HIGHEST_PROTOCOL)
# with open('average_IF.pkl', 'wb') as f:
#     pkl.dump(average_IF, f, pkl.HIGHEST_PROTOCOL)
# with open('post_processed_IF.pkl', 'wb') as f:
#     pkl.dump(post_processed_if_dict, f, pkl.HIGHEST_PROTOCOL)
