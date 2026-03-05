from sklearn.linear_model import LinearRegression
import numpy as np

def train_and_predict(years_input):
    years = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    salary = np.array([3, 5, 7, 9, 11])

    model = LinearRegression()
    model.fit(years, salary)

    prediction = model.predict([[years_input]])
    return prediction[0]

if __name__ == "__main__":
    result = train_and_predict(6)
    print(f"Predicted salary for 6 years experience: {result}")