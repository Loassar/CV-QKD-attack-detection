from core.input_state import InputState


def test_input_shape():
    input_1 = InputState(1, 1, 1, 1)
    print(input_1.value.shape)
    assert input_1.value.shape == (4,)


def test_eq():
    input_1 = InputState(1, 1, 1, 1)
    input_2 = InputState(1, 1, 1, 1)
    assert input_1 == input_2

