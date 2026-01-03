import pytest
import pandas as pd
from scripts.models import SmartReview_Sentiment_Classifier, SmartReview_CC_Extractor

@pytest.fixture(scope="module")
def sentiment_model():
  return SmartReview_Sentiment_Classifier()

@pytest.fixture(scope="module")
def cc_model():
  return SmartReview_CC_Extractor()

@pytest.fixture(scope="module")
def sentiment_system_prompt() :
    return """You are a Customer Review Analyst that predicts the sentiment of a customer's review concerning a product.
Customer reviews will come in one of four languages/dialects: English, French, Arabic written Moroccan darija, and Latin written moroccan darija.
You must choose between one of the following labels for each review: ["Positive","Negative"].
Only respond with the label name and nothing else.
"""

@pytest.fixture(scope="module")
def cc_system_prompt() :
    return """You are a customer review analyst whose job is to extract customer complaints and compliments from the input customer review.
Return the complaints and compliments in a string written in French shaped like this: {"plaintes" : [(list of complaints you extracted seperated by a comma)], "compliments" : [(list of compliments you extracted seperated by a comma)]}.​
Always output only the string and do not wrap it in JSON markdown or add any extra text before or after the string.​
Always write the compliments and complaints in a professional and formal register in French.​

Additional rules:
Always extract only complaints or compliments that are specific to the main product being reviewed; ignore any compliments or complaints that target another product, even if it is from the same brand as the main product.​
Always include any compliments or complaints related to the delivery or shipping of the product (speed, packaging, condition on arrival, courier experience, etc.), because delivery is considered an important part of the overall product experience.​
If the review contains no relevant complaint or compliment about the main product or its delivery, return empty lists: {"plaintes" : [], "compliments" : []}
"""