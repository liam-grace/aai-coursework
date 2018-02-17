import numpy as np

node_map = {}


class Node(object):
    def __init__(self, node_id):
        self.node_id = node_id
        node_map[node_id] = self

        self.output_weights = {}  # Leaving the note
        self.input_weights = {}  # Coming into the node

        self.value = [np.random.randint(1, 4)]

    def add_connection(self, node):
        weight = np.random.randint(1, 4)
        self.output_weights[node.node_id] = weight
        node.input_weights[self.node_id] = weight

    def recalculate_value(self):
        new_value = sum([w * node_map[n_id].value[0] for n_id, w in zip(self.input_weights.keys(), self.input_weights.values())])
        self.value = [new_value]

    def recalculate_weights(self, expected_value, actual_value):
        for n_id, w in zip(self.input_weights.keys(), self.input_weights.values()):
            current_value = self.input_weights[n_id]

            new_value = (expected_value * actual_value) + current_value  # Change to actual algorithm

            self.input_weights[n_id] = new_value
            node_map[n_id].output_weights[self.node_id] = new_value


if __name__ == '__main__':
    i0 = Node('I0')
    i1 = Node('I1')
    i2 = Node('I2')

    o = Node('O')

    i0.add_connection(o)
    i1.add_connection(o)
    i2.add_connection(o)

    def activate(i, j, k):
        i0.value = [i]
        i1.value = [j]
        i2.value = [k]

        o.recalculate_value()

        return o.value

    print(o.input_weights)

    print(activate(1, 2, 3))

    o.recalculate_weights(1, 0)

    print(o.input_weights)







