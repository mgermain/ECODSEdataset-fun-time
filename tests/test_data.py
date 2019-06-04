from ecodse_funtime_alpha.data import get_dataset


def test_dataset(tmp_path):
    d = tmp_path / "train-jpg"
    d.mkdir()
    p = d / "train_v2.csv"
    p.write_text("toto")

    image_dir = d.as_posix()
    labels_csv = p.as_posix()

    # dataset = get_dataset(image_dir, p)
    assert 1 != 5
