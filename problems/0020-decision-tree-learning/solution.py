import math
from collections import Counter


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    counts = Counter(labels)
    probabilities = [
        count / len(labels)
        for count in counts.values()
    ]

    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
    )


def calculate_information_gain(
    examples: list[dict],
    attr: str,
    target_attr: str
) -> float:
    """Calculate the information gain of splitting on attr."""
    # Entropy before splitting.
    parent_labels = [
        example[target_attr]
        for example in examples
    ]
    parent_entropy = calculate_entropy(parent_labels)

    # Group target labels by the selected attribute's value.
    groups = {}

    for example in examples:
        attr_value = example[attr]

        if attr_value not in groups:
            groups[attr_value] = []

        groups[attr_value].append(example[target_attr])

    # Entropy after splitting.
    weighted_child_entropy = 0.0

    for child_labels in groups.values():
        child_weight = len(child_labels) / len(examples)

        weighted_child_entropy += (
            child_weight
            * calculate_entropy(child_labels)
        )

    return parent_entropy - weighted_child_entropy


def majority_class(
    examples: list[dict],
    target_attr: str
) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(
        example[target_attr]
        for example in examples
    )

    highest_count = max(counts.values())

    tied_labels = [
        label
        for label, count in counts.items()
        if count == highest_count
    ]

    return sorted(tied_labels)[0]


def learn_decision_tree(
    examples: list[dict],
    attributes: list[str],
    target_attr: str
) -> dict | str:
    """Build a decision tree using the ID3 algorithm."""
    labels = [
        example[target_attr]
        for example in examples
    ]

    # Base case 1: the node is pure.
    if len(set(labels)) == 1:
        return labels[0]

    # Base case 2: no attributes remain.
    if not attributes:
        return majority_class(examples, target_attr)

    # Select the attribute with the largest information gain.
    # max keeps the first attribute when gains are tied.
    best_attr = max(
        attributes,
        key=lambda attr: calculate_information_gain(
            examples,
            attr,
            target_attr
        )
    )

    # ID3 does not reuse a categorical attribute.
    remaining_attributes = [
        attr
        for attr in attributes
        if attr != best_attr
    ]

    tree = {
        best_attr: {}
    }

    # The contract requires deterministic sorted branch order.
    attr_values = sorted({
        example[best_attr]
        for example in examples
    })

    for value in attr_values:
        subset = [
            example
            for example in examples
            if example[best_attr] == value
        ]

        tree[best_attr][value] = learn_decision_tree(
            subset,
            remaining_attributes,
            target_attr
        )

    return tree