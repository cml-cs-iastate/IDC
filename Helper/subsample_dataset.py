import csv
import collections

with open('../data/AmazonYelpCombined/AmazonYelp.csv', 'r', encoding='utf-8', errors='ignore') as f,\
        open('AmazonYelp_10K.txt', 'w', encoding='utf-8') as w:

    csv_reader = csv.reader(f)
    # get the header
    header = next(csv_reader)
    label_idx = header.index('label')
    content_idx = header.index('content')
    print(f'The label index is : {label_idx} and the content index is : {content_idx}')

    class_count = collections.defaultdict(int)
    exit_ = set()

    for i, line in enumerate(csv_reader):
        # get the sentence from the line
        sentence = line[content_idx]
        label = line[label_idx]

        if label not in exit_ and label != '3' and len(sentence.strip().split()) > 500:
            if class_count[label] < 10000:
                w.write(line[0] + ',' + line[1].strip())
                w.write('\n')
                class_count[label] += 1
            else:
                exit_.add(label)

        if exit_ == {'1', '2', '4', '5'}:
            break

    print([(i, j) for i, j in class_count.items()])
