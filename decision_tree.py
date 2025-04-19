import math
import pandas as pd
import numpy as np


def ID3(attributes, objects, depth =0, max_depth=None):
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
        subset = [row[:best_attr_index] + row[best_attr_index + 1:] for row in objects if row[best_attr_index] == value]
        new_attributes = attributes[:best_attr_index] + attributes[best_attr_index + 1:]
        subtree = ID3(new_attributes, subset, depth + 1, max_depth)
        tree[best_attr][value] = subtree
    return tree