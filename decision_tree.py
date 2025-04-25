import pandas as pd
import numpy as np


def entropy(labels):
    label_counts = pd.Series(labels).value_counts(normalize=True)
    return -sum(label_counts * np.log2(label_counts + 1e-9))


def find_best_attribute_index(objects, attributes):
    best_gain = -1
    best_attr_index = -1

    base_entropy = entropy([row[-1] for row in objects])

    for i, attr in enumerate(attributes):
        values = [row[i] for row in objects]
        weighted_entropy = 0.0

        for value in set(values):
            subset = [row for row in objects if row[i] == value]
            weight = len(subset) / len(objects)
            weighted_entropy += weight * entropy([row[-1] for row in subset])

        gain = base_entropy - weighted_entropy

        if gain > best_gain:
            best_gain = gain
            best_attr_index = i

    return best_attr_index


def most_common_class(objects):
    labels = [row[-1] for row in objects]
    return pd.Series(labels).value_counts().idxmax()


def ID3(attributes, objects, depth=0, max_depth=None):
    if not objects:
        return "Brak danych"

    labels = [row[-1] for row in objects]
    if all(label == labels[0] for label in labels):
        return labels[0]

    if not attributes or (max_depth is not None and depth == max_depth):
        return most_common_class(objects)

    best_attr_index = find_best_attribute_index(objects, attributes)
    best_attr = attributes[best_attr_index]

    tree = {best_attr: {}}
    attr_values = set(row[best_attr_index] for row in objects)

    for value in attr_values:
        subset = [row[:best_attr_index] + row[best_attr_index + 1:]
                  for row in objects if row[best_attr_index] == value]
        new_attributes = (attributes[:best_attr_index] +
                          attributes[best_attr_index + 1:])
        subtree = ID3(new_attributes, subset, depth + 1, max_depth)
        tree[best_attr][value] = subtree

    return tree


def predict(tree, sample, attributes):
    if not isinstance(tree, dict):
        return 1 if tree == 1 else 0

    attribute = next(iter(tree))
    idx = attributes.index(attribute)
    value = sample.iloc[idx]

    subtree = tree[attribute].get(value)
    if value not in ['x', 'o', 'b']:
        print(f'Invalid value: {value}')
        value = 'b'

    new_attributes = attributes[:idx] + attributes[idx+1:]
    new_sample_values = list(sample[:idx]) + list(sample[idx+1:])
    new_sample = pd.Series(new_sample_values, index=new_attributes)

    return predict(subtree, new_sample, new_attributes)
