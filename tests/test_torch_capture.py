import TensorState as ts  # noqa: N813 -- deliberate package alias


def test_capture_layers(model, data, capture_states, device, disk_path, benchmark):
    _train, test = data

    model_gen, layer = model
    m = model_gen(num_classes=len(test.dataset.classes))
    if capture_states:
        ts.build_efficiency_model(m, attach_to=[layer], storage_path=disk_path)

    m.to(device)
    m.eval()

    # warmup
    for x, _y in test:
        m(x.to(device))

        break

    # benchmark
    for x, _y in test:
        benchmark(m, x.to(device))

        break
