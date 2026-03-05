from app import train_and_predict

def test_prediction():
    prediction = train_and_predict(6)
    assert round(prediction, 1) == 13.0