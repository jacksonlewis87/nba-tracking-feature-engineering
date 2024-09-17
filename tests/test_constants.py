from constants import DataPaths


def test_data_paths():
    assert DataPaths.RAW.value == "raw"
    assert DataPaths.PREPROCESSED.value == "preprocessed"
    assert DataPaths.TENSORS.value == "tensors"
    assert DataPaths.PARQUET.value == "parquet"
