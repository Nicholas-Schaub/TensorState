import TensorState as ts


def test_capture_layers(model, data, capture_states, device, disk_path):
    train, test = data

    model_gen, layer = model
    m = model_gen(num_classes=len(test.dataset.classes))
    if capture_states:
        ts.build_efficiency_model(m, attach_to=[layer], storage_path=disk_path)

    m.to(device)
    m.eval()
    for x, y in test:
        z = m(x.to(device))

        break
