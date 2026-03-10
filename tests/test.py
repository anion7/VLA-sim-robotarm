import pytest
import torch
from rt2.model import RT2


@pytest.fixture
def rt2():
    return RT2()


@pytest.fixture
def img():
    return torch.rand((1, 3, 256, 256))


@pytest.fixture
def text():
    return torch.randint(0, 20000, (1, 1024))


def test_init(rt2):
    assert isinstance(rt2, RT2)


def test_forward(rt2, img, text):
    output = rt2(img, text)
    assert isinstance(output, tuple)
    logits, loss = output
    assert logits.shape == (1, 1023, 20000)


def test_forward_different_img_shape(rt2, text):
    img = torch.rand((2, 3, 256, 256))
    text2 = torch.randint(0, 20000, (2, 1024))
    output = rt2(img, text2)
    logits, loss = output
    assert logits.shape == (2, 1023, 20000)


def test_forward_different_text_length(rt2, img):
    text = torch.randint(0, 20000, (1, 512))
    output = rt2(img, text)
    logits, loss = output
    assert logits.shape == (1, 511, 20000)


def test_forward_exception(rt2, img):
    with pytest.raises(Exception):
        rt2(img)
