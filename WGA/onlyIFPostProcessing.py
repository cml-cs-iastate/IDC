import operator
import pickle5 as pkl

import src.settings as settings
from src.WGA.wgaSolver import post_processing

# Load IF dictionary
with open(settings.average_IF, 'rb') as f:
    if_average = pkl.load(f)

with open(settings.count_IF, 'rb') as f:
    if_count = pkl.load(f)

# Post-Processing
post_processed_if_dict = post_processing(if_average, if_count, alpha_threshold=0.1, beta_threshold=0.25, min_count=5)
print(f'Post-Processing is over!')

with open(settings.post_processed_IF, 'wb') as f:
    pkl.dump(post_processed_if_dict, f, pkl.HIGHEST_PROTOCOL)

for key, value in post_processed_if_dict.items():
    print(f'The interpretation features of class: {key} is :')
    for item in sorted(value.items(), key=operator.itemgetter(1)):
        print('{} : {:.2f}'.format(item[0], item[1]))
    print(f'The length of the dictionary for class {key} is = {len(value)}')
    print()