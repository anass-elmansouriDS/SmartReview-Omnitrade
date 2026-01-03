import pytest
import pandas as pd
@pytest.mark.parametrize(
    "review",
    [("This is a great product"),("Bon produit"),("mantoj 3yan"),("هاذ المنتوج مقزّز وبزّاف.")],
)
def test_sentiment_predict(review,sentiment_system_prompt,sentiment_model) :
    pred=sentiment_model.predict(system_prompt=sentiment_system_prompt,review=review)
    assert isinstance(pred, str)
    assert pred in ["Positive","Negative"]

def test_sentiment_predict_batch(mock_reviews,sentiment_system_prompt,sentiment_model) :
    preds=sentiment_model.predict_batch(system_prompt=sentiment_system_prompt,reviews=mock_reviews)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape==(len(mock_reviews),3)

@pytest.mark.parametrize(
    "review",
    [("This is a great product"),("Bon produit"),("mantoj 3yan"),("هاذ المنتوج مقزّز وبزّاف.")],
)
def test_cc_predict(review,cc_system_prompt,cc_model) :
    pred=cc_model.predict(system_prompt=cc_system_prompt,review=review)
    assert isinstance(pred, str)
    assert pred.startswith("{")
    assert pred.endswith("}")
    assert "plaintes" in pred
    assert "compliments" in pred

def test_cc_predict_batch(mock_reviews,cc_system_prompt,cc_model) :
    preds=cc_model.predict_batch(system_prompt=cc_system_prompt,reviews=mock_reviews)
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape==(len(mock_review),5)