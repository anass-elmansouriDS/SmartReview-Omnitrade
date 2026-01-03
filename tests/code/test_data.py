import pytest
from scripts.data import *
from pyspark.sql.connect.dataframe import DataFrame

@pytest.mark.parametrize(
    "train, sentiment",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_get_dataset(train,sentiment) :
    assert isinstance(get_dataset(train,sentiment),DataFrame)
@pytest.mark.parametrize(
    "example, system_prompt, sentiment, expected_output",
    [
        ({"review" : "bon produit", "sentiment" : "Positive"}, "Classify the input review as positive or negative.\n Customer Review : ", True, {
            "messages": [
                {"role": "user", "content": "Classify the input review as positive or negative.\n Customer Review : bon produit"},
                {"role": "assistant", "content": "Positive"}
            ]
        }),
        ({"review" : "bon produit", "label" : '{"plaintes": [], "compliments": ["bon produit"]}'}, "Extract the compliments and complaints from the input review and put them in a string that resembles a python dictionary with keys 'plaintes' and 'compliments'.\n Customer Review : ", False, {
            "messages": [
                {"role": "user", "content": "Extract the compliments and complaints from the input review and put them in a string that resembles a python dictionary with keys 'plaintes' and 'compliments'.\n Customer Review : bon produit"},
                {"role": "assistant", "content": '{"plaintes": [], "compliments": ["bon produit"]}'}
            ]
        }),
    ],
)
def test_formatting_prompts_func(example, system_prompt, sentiment, expected_output) :
    assert formatting_prompts_func(example, system_prompt, sentiment) == expected_output

def test_format_conversation() :
    eos_token="<|EOS|>"
    messages={
            "messages": [
                {"role": "user", "content": "Extract the compliments and complaints from the input review and put them in a string that resembles a python dictionary with keys 'plaintes' and 'compliments'.\n Customer Review : bon produit"},
                {"role": "assistant", "content": '{"plaintes": [], "compliments": ["bon produit"]}'}
            ]
        }
    expected='<|user|>\nExtract the compliments and complaints from the input review and put them in a string that resembles a python dictionary with keys \'plaintes\' and \'compliments\'.\n Customer Review : bon produit<|assistant|>\n{"plaintes": [], "compliments": ["bon produit"]}<|EOS|>'
    assert format_conversation(messages["messages"],eos_token)==expected