import pytest
import great_expectations as ge
from scripts.data import get_dataset
import pytest


@pytest.mark.fixture(scope="session")
def df_sentiment_inference() :
    pdf = get_dataset(sentiment=True).toPandas()
    context = ge.get_context()
    datasource = context.data_sources.add_pandas(name="sentiment_inference_dataset")
    data_asset = datasource.add_dataframe_asset(name="sentiment_inference_dataset_asset")
    batch_request = data_asset.build_batch_request({"dataframe":pdf})
    df = context.get_validator(
        batch_request=batch_request
    )
    return df
@pytest.mark.fixture(scope="session")
def df_sentiment_train() :
    pdf = get_dataset(train=True,sentiment=True).toPandas()
    context = ge.get_context()
    datasource = context.data_sources.add_pandas(name="sentiment_train_dataset")
    data_asset = datasource.add_dataframe_asset(name="sentiment_train_dataset_asset")
    batch_request = data_asset.build_batch_request({"dataframe":pdf})
    df = context.get_validator(
        batch_request=batch_request
    )
    return df
@pytest.mark.fixture(scope="session")
def df_cc_inference() :
    pdf = get_dataset().toPandas()
    context = ge.get_context()
    datasource = context.data_sources.add_pandas(name="cc_inference_dataset")
    data_asset = datasource.add_dataframe_asset(name="cc_inference_dataset_asset")
    batch_request = data_asset.build_batch_request({"dataframe":pdf})
    df = context.get_validator(
        batch_request=batch_request
    )
    return df

@pytest.mark.fixture(scope="session")
def df_cc_train() :
    pdf = get_dataset(train=True).toPandas()
    context = ge.get_context()
    datasource = context.data_sources.add_pandas(name="cc_train_dataset")
    data_asset = datasource.add_dataframe_asset(name="cc_train_dataset_asset")
    batch_request = data_asset.build_batch_request({"dataframe":pdf})
    df = context.get_validator(
        batch_request=batch_request
    )
    return df
