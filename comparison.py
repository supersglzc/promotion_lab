import dtree_build
import naive_bayes
import sys
import csv
from sklearn.utils import resample


def main(col_names=None):
    if len(sys.argv) < 2:
        print("Please specify input csv file name")
        return

    csv_file_name = sys.argv[1]
    data = []
    with open(csv_file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            data.append(list(row))

    train = resample(data[1:], replace=True, n_samples=int(len(data)))
    test = []
    for i in data[1:]:
        if i not in train:
            test.append(i)
    tree = dtree_build.buildtree(train, min_gain=0.01, min_samples=5)

    dtree_build.printtree(tree, '', col_names)

    result2 = naive_bayes.build(train)
    # print(result2)
    # max_tree_depth = dtree_build.max_depth(tree)
    # print("max number of questions=" + str(max_tree_depth))

    # print(test)
    out_put = [['instance', 'actual', 'predicted', 'probability']]
    total = 0
    correct = 0
    correct2 = 0
    for i in test:
        total += 1
        result = dtree_build.classify(i, tree)
        out = naive_bayes.classifier(result2, i)
        sum_probability = 0
        max_number = 0
        choice = ''
        for n, m in result.items():
            sum_probability += m
            if m >= max_number:
                max_number = m
                choice = n
        if choice == i[-1]:
            correct += 1
        if out == int(i[-1]):
            correct2 += 1
        sublist = [total, i[-1], choice, max_number/sum_probability]
        out_put.append(sublist)

        # print(result)
    with open("predicted.csv", "w") as output:
        writer = csv.writer(output)
        writer.writerows(out_put)
    print("Accuracy for decision tree is", correct / len(test))
    print("Accuracy for naive bayes is", correct2 / len(test))


if __name__ == "__main__":
    col_names = ['ZIP_CODE',
                 'TOTAL_VISITS',
                 'AVRG_SPENT_PER_VISIT',
                 'HAS_CREDIT_CARD',
                 'PKNIT_TOPS',
                 'PKNIT_DRES',
                 'PBLOUSES', 'PJACKETS',
                 'PCAR_PNTS',
                 'PSHIRTS',
                 'PFASHION',
                 'AXSPEND',
                 'SPEND_LAST_3MONTH	',
                 'SPENT_LAST_YEAR',
                 'PROMOS_ON_FILE',
                 'MARKDOWN',
                 'STYLES',
                 'STORES',
                 'MAILED',
                 'RESPONSERATE',
                 'LTFREDAY',
                 'CLUSTYPE',
                 'RESP']
    main(col_names)
