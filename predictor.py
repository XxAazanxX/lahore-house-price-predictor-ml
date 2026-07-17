import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class LahoreHousePricePredictor:
    """
    An Object-Oriented Machine Learning pipeline to clean data,
    train a Linear Regression model, and predict property prices in Lahore.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def load_data(self, file_path: str):
        """Loads and cleans dataset."""
        print("[*] Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Isolate features and target variable
        required_columns = ["Size_sqm", "Bedrooms", "Total_Price_Local"]
        df = df[required_columns].dropna()
        
        X = df[['Size_sqm', 'Bedrooms']]
        Y = df['Total_Price_Local']
        
        # Splitting data into training and testing sets
        return train_test_split(X, Y, test_size=0.2, random_state=42)

    def train(self, X_train, Y_train):
        """Trains the Linear Regression algorithm on the training dataset."""
        print("[*] Training the Linear Regression model...")
        self.model.fit(X_train, Y_train)
        self.is_trained = True
        print("[+] Model trained successfully!")

    def evaluate(self, X_test, Y_test):
        """Evaluates model performance using R-squared metric."""
        if not self.is_trained:
            raise Exception("[-] Model must be trained before evaluation.")
        score = self.model.score(X_test, Y_test)
        print(f"[+] Model Accuracy (R² Score): {score * 100:.2f}%")
        return score
    
    def predict_price(self, size_sqm: float, bedrooms: int):
        """Predicts the price of a property based on its size and number of bedrooms."""
        if not self.is_trained:
            raise Exception("[-] Model must be trained before calculation.")
        
        # FIXED: Wrap in a DataFrame with matching column names to keep scikit-learn happy
        input_data = pd.DataFrame([[size_sqm, bedrooms]], columns=['Size_sqm', 'Bedrooms'])
        prediction = self.model.predict(input_data)
        
        # Extract and return the single scalar value
        return max(0.0, float(prediction[0]))
    
# ==========================================
# Execution Block 
# ==========================================
if __name__ == "__main__":
    predictor = LahoreHousePricePredictor()
    dataset_name = "lahore_housing_prices.csv"
    
    try:
        X_train, X_test, Y_train, Y_test = predictor.load_data(dataset_name)

        predictor.train(X_train, Y_train)
        predictor.evaluate(X_test, Y_test)

        # Test sample calculation
        print("\n--- Running Live Prediction Test ---")
        test_size = 125.0  # 125 sqm (~5 Marla house)
        test_beds = 3      # 3 Bedrooms
        
        # FIXED: Passed size first, then beds, to match function definition order
        estimated_pkr = predictor.predict_price(test_size, test_beds)
        
        print(f"Input: {test_beds} Bed, {test_size} sqm Size House in Lahore")
        print(f"Estimated Market Value: PKR {estimated_pkr:,.2f}\n")
        
    except FileNotFoundError:
        print(f"[-] Error: '{dataset_name}' not found. Please place your dataset file in the same directory.")
