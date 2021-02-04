import csv


# Tested
def inter_agreement(data_file):
    """
    Calculate the inter_agreement between two entities using Kohen Kappa
    :param data_file: is the csv file where the column has the interpretation features for each input document.
    See CyBox\SELFIE ...\1000 Tweets and 250 Amazon Reviews\Amazon\Amazon 250 Reviews Preprocessed Nov 6.csv
    For an data_file example.
    """
    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        data, A, B = list(), list(), list()
        for sample in csv_reader:
            data.append(sample[0])
            A.append(sample[4])
            B.append(sample[8])

        # compare between A and B (Human with Human / or Human with Machine)
        a, b, c, d, counter = 0, 0, 0, 0, 0
        for idx, sample in enumerate(data):
            list_a = A[idx].split()
            list_b = B[idx].split()
            for word in sample.split():
                counter += 1
                if word in list_a and word in list_b:
                    a += 1
                elif word in list_a and word not in list_b:
                    b += 1
                elif word not in list_a and word in list_b:
                    c += 1
                elif word not in list_a and word not in list_b:
                    d += 1

        po = (a + d) / (a + b + c + d)
        p1 = ((a + b) / (a + b + c + d)) * ((a + c) / (a + b + c + d))
        p0 = ((d + b) / (a + b + c + d)) * ((d + c) / (a + b + c + d))
        pe = p1 + p0
        kappa = (po - pe) / (1 - pe)

        print(f'a: {a}, b: {b}, c: {c}, d: {d}, sum(a,b,c,d): {a+b+c+d}, len(data): {counter}')
        print(f'po: {po}, p0: {p0}, p1: {p1}, pe: {pe}, Kappa: {kappa}')

        return kappa
