import numpy as np
from os.path import join
from PIL import Image
import pytest

from ecodse_funtime_alpha.data import get_dataset
from ecodse_funtime_alpha.data import get_labels_distribution
from ecodse_funtime_alpha.data import load_and_preprocess_image


class TestDataset(object):
    @pytest.fixture(autouse=True)
    def mock_files(self, tmpdir):
        self.img_size = 256
        self.img_channel = 3
        self.img_colorpixel = 128
        self.img_color = 255

        p = tmpdir.mkdir("train-jpg").join("train_v2.csv")
        p.write_text("image_name,tags\ntrain_0,haze primary\ntrain_1,agriculture clear primary water", encoding="utf-8")

        img = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img[self.img_colorpixel] = self.img_color

        Image.fromarray(img).save(str(tmpdir.join("train-jpg").join("train_0.jpg")), quality=100)
        Image.fromarray(img).save(str(tmpdir.join("train-jpg").join("train_1.jpg")), quality=100)

    def test_preprocess_image(self, tmpdir):
        img = load_and_preprocess_image(str(tmpdir.join("train-jpg").join("train_0.jpg")))
        assert img[self.img_colorpixel].numpy().all() == self.img_color / 255

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

    def test_missing_image(self, tmpdir, capsys):
        p = tmpdir.join("train-jpg").join("missing_image.csv")
        missing_name = "missing"
        p.write_text(f"image_name,tags\n{missing_name},haze primary\ntrain_1,agriculture clear primary water", encoding="utf-8")
        image_dir = str(tmpdir.join("train-jpg"))
        labels_csv = str(p)
        get_dataset(image_dir, labels_csv)
        full_name = join(image_dir, f"{missing_name}.jpg")
        captured = capsys.readouterr()
        assert captured.out == f"WARNING: {full_name} does not exist\n"
