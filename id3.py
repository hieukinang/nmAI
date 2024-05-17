import math
from collections import defaultdict
from typing import List, Dict

class Node:
    def __init__(self, attribute=None):
        self.attribute = attribute
        self.value = None
        self.child = {}

    def get_attribute(self):
        return self.attribute

    def set_attribute(self, attribute):
        self.attribute = attribute

    def get_child(self):
        return self.child

    def set_child(self, child):
        self.child = child

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


class Sample:
    def __init__(self, outlook, temp, humidity, wind, decision):
        self.outlook = outlook
        self.temp = temp
        self.humidity = humidity
        self.wind = wind
        self.decision = decision

    def get_value_by_attribute(self, attribute):
        if attribute == "outlook":
            return self.outlook
        elif attribute == "temp":
            return self.temp
        elif attribute == "humidity":
            return self.humidity
        elif attribute == "wind":
            return self.wind
        elif attribute == "decision":
            return self.decision
        else:
            raise ValueError("not handle this attribute")


def main():
    data = [
        Sample("Sunny", "Hot", "High", "Weak", "No"),
        Sample("Sunny", "Hot", "High", "Strong", "No"),
        Sample("Overcast", "Hot", "High", "Weak", "Yes"),
        Sample("Rain", "Mild", "High", "Weak", "Yes"),
        Sample("Rain", "Cool", "Normal", "Weak", "Yes"),
        Sample("Rain", "Cool", "Normal", "Strong", "No"),
        Sample("Overcast", "Cool", "Normal", "Strong", "Yes"),
        Sample("Sunny", "Mild", "High", "Weak", "No"),
        Sample("Sunny", "Cool", "Normal", "Weak", "Yes"),
        Sample("Rain", "Mild", "Normal", "Weak", "Yes"),
        Sample("Sunny", "Mild", "Normal", "Strong", "Yes"),
        Sample("Overcast", "Mild", "High", "Strong", "Yes"),
        Sample("Overcast", "Hot", "Normal", "Weak", "Yes"),
        Sample("Rain", "Mild", "High", "Strong", "No")
    ]

    attributes = ["outlook", "temp", "humidity", "wind"]
    decision_tree = build_decision_tree(data, attributes)
    print_decision_tree(decision_tree)


def print_decision_tree(decision_tree, level=0):
    indent = " " * (level * 4)
    if decision_tree.get_attribute():
        print(f"{indent}{decision_tree.attribute}")
    if decision_tree.get_value() or not decision_tree.get_child():
        print(f"{indent}-> {decision_tree.get_value()}")
    else:
        for key, child in decision_tree.get_child().items():
            print(f"{indent}{key}")
            print_decision_tree(child, level + 1)


def build_decision_tree(data: List[Sample], attributes: List[str]) -> Node:
    if check_constant_result(data):
        return Node()

    best_attribute = determine_best_attribute(data, attributes)
    attribute_values = get_attribute_values(data, best_attribute)
    node = Node(best_attribute)
    for value in attribute_values:
        node_child = Node()
        node.child[value] = node_child

        subset = calculate_subset(data, best_attribute, value)
        result_current_node = list(set(s.decision for s in subset))
        if len(result_current_node) <= 1:
            node_child.set_value(result_current_node[0])
        else:
            remaining_attributes = [attr for attr in attributes if attr != best_attribute]
            node.child[value] = build_decision_tree(subset, remaining_attributes)
    return node


def get_attribute_values(data: List[Sample], attribute: str) -> List[str]:
    return list(set(d.get_value_by_attribute(attribute) for d in data))


def check_constant_result(data: List[Sample]) -> bool:
    return len(set(d.decision for d in data)) <= 1


def determine_best_attribute(data: List[Sample], attributes: List[str]) -> str:
    entropy_s = entropy(data)
    best_attribute = ""
    max_gain = -float('inf')
    for attribute in attributes:
        gain = calculate_gain(data, entropy_s, attribute)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
    return best_attribute


def calculate_gain(data: List[Sample], entropy_s: float, attribute: str) -> float:
    total_data = len(data)
    values = get_value_of_attribute(data, attribute)
    result_info_node = 0.0
    for key in values.keys():
        subset = calculate_subset(data, attribute, key)
        temp = len(subset) / total_data
        result_info_node += temp * entropy(subset)
    return entropy_s - result_info_node


def calculate_subset(data: List[Sample], attribute: str, key: str) -> List[Sample]:
    return [d for d in data if d.get_value_by_attribute(attribute) == key]


def entropy(data: List[Sample]) -> float:
    values = get_value_of_attribute(data, "decision")
    total_sample = len(data)
    result = 0.0
    for key, count in values.items():
        temp = count / total_sample
        result -= temp * log2(temp)
    return result


def log2(x):
    return math.log(x) / math.log(2)


def get_value_of_attribute(data: List[Sample], attribute: str) -> Dict[str, int]:
    if not data:
        return {}
    result = defaultdict(int)
    for d in data:
        result[d.get_value_by_attribute(attribute)] += 1
    return result


if __name__ == "__main__":
    main()
