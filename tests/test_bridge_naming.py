import importlib


def test_public_package_and_renamed_modules_are_importable():
    modules = [
        "bridge",
        "bridge.bridge_model",
        "bridge.train_bridge",
        "bridge.evaluate_bridge",
        "bridge.donor_level_average_expert",
        "bridge.sample_matching_expert",
    ]

    for module in modules:
        assert importlib.import_module(module).__name__ == module
