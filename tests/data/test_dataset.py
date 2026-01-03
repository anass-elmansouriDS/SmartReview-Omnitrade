import great_expectations as ge
import pandas as pd
from scripts.data import get_dataset

def test_sentiment_inference_dataset(df_sentiment_inference):
    """Test sentiment inference dataset quality and integrity."""
    column_list = ["review_id", "review", "predicted_sentiment"]
    df_sentiment_inference.expect_table_columns_to_match_ordered_list(column_list=column_list)  # schema adherence
    df_sentiment_inference.expect_column_values_to_be_null(column="predicted_sentiment")  # eligible reviews to be predicted on
    df_sentiment_inference.expect_column_values_to_be_unique(column="review_id")  # unique values
    df_sentiment_inference.expect_column_values_to_be_unique(column="review")  # unique values
    df_sentiment_inference.expect_column_values_to_be_of_type(column="review", type_="str")  # type adherence
    df_sentiment_inference.expect_column_values_to_be_of_type(column="review_id", type_="int")  # type adherence
    # Expectation suite
    results = df_sentiment_inference.validate()
    assert results["success"]

def test_sentiment_train_dataset(df_sentiment_train):
    """Test sentiment training dataset quality and integrity."""
    column_list = ["review_id", "review", "sentiment"]
    df_sentiment_train.expect_table_columns_to_match_ordered_list(column_list=column_list)  # schema adherence
    sentiments = ["Positive","Negative"]
    df_sentiment_train.expect_column_values_to_be_in_set(column="sentiment", value_set=sentiments)  # expected labels
    df_sentiment_train.expect_column_values_to_not_be_null(column="sentiment")  # missing values
    df_sentiment_train.expect_column_values_to_be_unique(column="review_id")  # unique values
    df_sentiment_train.expect_column_values_to_be_unique(column="review")  # unique values
    df_sentiment_train.expect_column_values_to_be_of_type(column="review", type_="str")  # type adherence
    df_sentiment_train.expect_column_values_to_be_of_type(column="review_id", type_="int")  # type adherence
    df_sentiment_train.expect_column_values_to_be_of_type(column="sentiment", type_="str") # type adherence
    # Expectation suite
    results = df_sentiment_train.validate()
    assert results["success"]

def test_cc_inference_dataset(df_cc_inference):
    """Test C&C inference dataset quality and integrity."""
    column_list = ["review_id", "review", "predicted_compliments","predicted_complaints"]
    df_cc_inference.expect_table_columns_to_match_ordered_list(column_list=column_list)  # schema adherence
    df_cc_inference.expect_column_values_to_be_null(column="predicted_compliments")  # eligible reviews to be predicted on
    df_cc_inference.expect_column_values_to_be_null(column="predicted_complaints")  # eligible reviews to be predicted on
    df_cc_inference.expect_column_values_to_be_unique(column="review_id")  # unique values
    df_cc_inference.expect_column_values_to_be_unique(column="review")  # unique values
    df_cc_inference.expect_column_values_to_be_of_type(column="review", type_="str")  # type adherence
    df_cc_inference.expect_column_values_to_be_of_type(column="review_id", type_="int")  # type adherence
    results = df_cc_inference.validate()
    assert results["success"]

def test_cc_train_dataset(df_cc_train):
    """Test C&C training dataset quality and integrity."""
    column_list = ["review_id", "review", "label"]
    df_cc_train.expect_table_columns_to_match_ordered_list(column_list=column_list)  # schema adherence
    df_cc_train.expect_column_values_to_not_be_null(column="label")  # missing values
    df_cc_train.expect_column_values_to_be_unique(column="review_id")  # unique values
    df_cc_train.expect_column_values_to_be_unique(column="review")  # unique values
    df_cc_train.expect_column_values_to_be_of_type(column="review", type_="str")  # type adherence
    df_cc_train.expect_column_values_to_be_of_type(column="review_id", type_="int")  # type adherence
    df_cc_train.expect_column_values_to_be_of_type(column="label", type_="str") # type adherence
    df_cc_train.expect_column_values_to_match_regex(column="label",regex=r"^\{(?=.*plaintes)(?=.*compliments).*}$")
    # Expectation suite
    results = df_cc_train.validate()
    assert results["success"]