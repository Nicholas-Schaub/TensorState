import torch

from TensorState import Dependency


def test_linked_neurons():
    conv = torch.nn.Conv2d(3, 32, 3)
    adaptive = torch.nn.AdaptiveAvgPool2d(output_size=(2, 2))
    flatten = torch.nn.Flatten()
    linear = torch.nn.Linear(128, 10)

    model = torch.nn.Sequential(
        conv,
        adaptive,
        flatten,
        linear,
    )

    graph = Dependency.ModuleGraph(model)

    group = graph.groups[0]

    linked = group.linked_neurons()

    assert len(linked[0]) == 32
    assert len(linked[0][0]) == 4


# @pytest.mark.skip()
def test_capture_layers(model, data, device):
    train, test = data

    model_gen, layer = model
    m = model_gen(num_classes=len(test.dataset.classes))

    m.to(device)
    m.eval()

    graph = Dependency.ModuleGraph(m)

    # Sanity test
    for x, y in test:
        z = m(x.to(device))

        break

    print(m)

    for index, group in enumerate(graph.groups):
        # if index == 5:
        print(f"group: {index}")
        print("Before")
        for connectivity, vertex in zip(group.M(), group.V()):
            print(vertex)
            print(connectivity)
            # print(graph.groups[0].V())

        # Apoptosis
        stack = list(group.linked_neurons()[0][0])
        idxs = []
        while len(stack) > 0:
            s = stack.pop()
            if isinstance(s, int):
                idxs.append(s)
            elif isinstance(s[0], int):
                idxs.extend([i for i in s])
            else:
                stack.extend([list(i) for i in s])

        group.apoptosis(idxs=idxs)

        # if index == 5:
        print(f"idxs: {idxs}")
        print("After")
        for vertex in group.V():
            print(vertex)

        print()

    # Test modified network
    for x, y in test:
        z = m(x.to(device))

        break
