import pytest

@pytest.mark.parametrize(
    "input, expected_output",
    [
        ("This is a great product, has all the things i need and feels good too!","Positive"),
        ("This is a good product, has all the stuff i need and feels great too!","Positive"),
        ("Ce produit n'est pas vraiment bon, il n'a pas tout ce dont j’ai besoin et il est désagréable à utiliser !","Negative"),
        ("Ce produit n'est pas bon, il n'a pas du tout ce que dont j’ai besoin et il donne pas une bonne sensation !","Negative"),
        ("المنتوج زوين بزّاف، فيه كُلشي اللي كنحتاج وكيعطي إحساس مزيان حتى هو!","Positive"),
        ("المنتوج مزيان، فيه كُلشي اللي كنحتاج وكيعطي إحساس زوين بزّاف","Positive"),
        ("Had lproduit khayb bzzaaf, mafih ta7aja mn dkchi li kan7taj w 3yan bzaf!","Negative"),
        ("Had lproduit khayb bzf, mafih ta7aja mn dkchi li kan7taj w makay3tich i7ssas zwine!","Negative"),
    ]
)
def test_sentiment_behavioral_invariance(input, expected_output, sentiment_system_prompt,sentiment_model):
    pred=sentiment_model.predict(system_prompt=sentiment_system_prompt,review=input)
    assert pred == expected_output

@pytest.mark.parametrize(
    "input, expected_output",
    [
        ("This is a great product, has all the things i need and feels good too!","Positive"),
        ("This is not a good product, it does not have all the stuff i need and feels awful too!","Negative"),
        ("Ce produit n'est pas vraiment bon, il n'a pas tout ce dont j’ai besoin et il est désagréable à utiliser !","Negative"),
        ("Ce produit est vraiment bon, il a tout ce que dont j’ai besoin et il donne une trés bonne sensation !","Positive"),
        ("المنتوج زوين بزّاف، فيه كُلشي اللي كنحتاج وكيعطي إحساس مزيان حتى هو!","Positive"),
        ("المنتوج خايب، ما فيه حتى حاجة من اللي كنحتاج وكيعطي إحساس خايب بزّاف","Negative"),
        ("Had lproduit khayb bzzaaf, mafih ta7aja mn dkchi li kan7taj w 3yan bzaf!","Negative"),
        ("Had lproduit zwin bzf, fih kolchi dkchi li kan7taj w kay3ti i7ssas zwine!","Positive"),
    ]
)
def test_sentiment_behavioral_directional(input, expected_output, sentiment_system_prompt,sentiment_model):
    pred=sentiment_model.predict(system_prompt=sentiment_system_prompt,review=input)
    assert pred == expected_output

