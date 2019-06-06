import numpy as np
from PIL import Image
import pytest

from ecodse_funtime_alpha.data import get_dataset
from ecodse_funtime_alpha.data import get_labels_distribution


class TestDataset(object):
    @pytest.fixture(autouse=True)
    def mock_files(self, tmpdir):
        p = tmpdir.mkdir("train-jpg").join("train_v2.csv")
        p.write_text("image_name,tags\ntrain_0,haze primary\ntrain_1,agriculture clear primary water", encoding="utf-8")

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[128] = 255

        Image.fromarray(img).save(str(tmpdir.join("train-jpg").join("train_0.jpg")), quality=100)
        Image.fromarray(img).save(str(tmpdir.join("train-jpg").join("train_1.jpg")), quality=100)

    def test_image(self, tmpdir):
        image_dir = str(tmpdir.join("train-jpg"))
        labels_csv = str(tmpdir.join("train-jpg").join("train_v2.csv"))

        dataset = get_dataset(image_dir, labels_csv)

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[128] = 255
        img = img.astype(np.float32) / 255.

        for _, sample in enumerate(dataset.take(1)):
            assert np.allclose(img, sample[0].numpy())

    def test_labels(self, tmpdir):
        image_dir = str(tmpdir.join("train-jpg"))
        labels_csv = str(tmpdir.join("train-jpg").join("train_v2.csv"))

        dataset = get_dataset(image_dir, labels_csv)

        for i, sample in enumerate(dataset.take(2)):
            if i == 0:
                assert np.allclose(np.array([1, 1, 0, 0, 0]), sample[1].numpy())
            else:
                assert np.allclose(np.array([0, 1, 1, 1, 1]), sample[1].numpy())

    def test_labels_distribution(self, tmpdir):
        labels_csv = str(tmpdir.join("train-jpg").join("train_v2.csv"))

        labels = get_labels_distribution(labels_csv)

        true_labels = {
            'haze': 1,
            'primary': 2,
            'agriculture': 1,
            'clear': 1,
            'water': 1
        }

        for k, v in labels.items():
            assert k in true_labels
            assert true_labels[k] == v
