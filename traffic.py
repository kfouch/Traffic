import pandas as pd
import streamlit as st
import xgboost as xgb
from mapie.regression import MapieRegressor
from sklearn.model_selection import train_test_split

# Function to load data (ensure this is implemented according to your file's structure)
def load_data():
    # Load your dataset (adjust the path as needed)
    df = pd.read_csv("Traffic_Volume.csv")  # Replace with your data path
    return df

# Function to preprocess data
def preprocess_data(df):
    # Convert 'date_time' to datetime and extract useful features
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['weekday'] = df['date_time'].dt.weekday

    # Handle the 'holiday' column by encoding it as binary
    df['holiday'] = df['holiday'].apply(lambda x: 0 if x == 'None' else 1)

    # One-hot encode 'weather_main' column
    weather_encoded = pd.get_dummies(df['weather_main'], prefix='weather')
    df = pd.concat([df, weather_encoded], axis=1)

    # Drop unnecessary columns
    df.drop(['weather_main', 'date_time'], axis=1, inplace=True)

    return df

# Load and preprocess data
def main():
    st.title("Traffic Volume Prediction")
    st.image("traffic_image.gif", width=700)

    # Load and preprocess data
    df = preprocess_data(load_data())

    st.sidebar.image("traffic_sidebar.jpg", use_container_width=True)
    st.write("## Adjust Alpha Value")
    alpha = st.slider("Significance Level (Alpha)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # Reduce the dataset size if memory issues occur (use only the first 1000 rows for training)
    df = df.head(1000)

    # Separate features and target
    X = df.drop(columns=["traffic_volume"])
    y = df["traffic_volume"]

    # Load models
    xgb_model, mapie = load_model()

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model (make sure to fit your model)
    xgb_model.fit(X_train, y_train)
    mapie.fit(X_train, y_train)

    # Predictions
    y_pred = mapie.predict(X_test, alpha=alpha)
    y_pred_mean = y_pred[0]
    y_pred_low, y_pred_high = y_pred[1][:, 0], y_pred[1][:, 1]

    # Ensure consistent dimensions
    y_test = y_test.reset_index(drop=True)
    y_pred_mean = pd.Series(y_pred_mean.flatten())
    y_pred_low = pd.Series(y_pred_low.flatten())
    y_pred_high = pd.Series(y_pred_high.flatten())

    # Create Results DataFrame
    test_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_mean,
        "Lower Prediction Limit": y_pred_low,
        "Upper Prediction Limit": y_pred_high
    })
    st.write("### Predictions on Test Data")
    st.dataframe(test_results)

    # Sidebar: Select file input method
    option = st.sidebar.selectbox(
        "Choose Input Method",
        ["Upload CSV File", "Create Your Own Form"]
    )

    if option == "Upload CSV File":
        # CSV file upload for prediction data
        st.sidebar.write("Upload a CSV file containing the prediction data with the following columns: 'hour', 'day', 'month', 'weekday', 'holiday', and 'weather_main'.")

        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            st.sidebar.write("File uploaded successfully.")  # Debugging step to ensure file upload
            
            # Read the uploaded CSV file
            csv_data = pd.read_csv(uploaded_file)
            st.sidebar.write(csv_data.head())  # Display a preview of the file

            # Ensure the uploaded CSV contains all the features (including weather columns)
            all_weather_columns = ['weather_Fog', 'weather_Snow', 'temp', 'weather_Clouds', 'clouds_all', 'weather_Haze', 'weather_Thunderstorm', 'weather_Drizzle', 'rain_1h', 'weather_Smoke', 'weather_Mist', 'weather_Rain', 'weather_Squall', 'snow_1h']
            missing_columns = [col for col in all_weather_columns if col not in csv_data.columns]
            for col in missing_columns:
                csv_data[col] = 0  # Add missing columns with default value 0

            # Reorder columns to match the model's expectations (same order as training data)
            csv_data = csv_data[all_weather_columns + ['hour', 'day', 'month', 'weekday', 'holiday']]

            # Preprocess the uploaded data
            csv_data = preprocess_data(csv_data)

            # Make predictions on the uploaded data
            predictions = mapie.predict(csv_data, alpha=alpha)

            # Display the predictions
            prediction_df = pd.DataFrame({
                "Predicted Traffic Volume": predictions[0].flatten(),
                "Lower Prediction Limit": predictions[1][:, 0].flatten(),
                "Upper Prediction Limit": predictions[1][:, 1].flatten()
            })
            st.sidebar.write("### Predictions on Uploaded Data")
            st.sidebar.dataframe(prediction_df)

    elif option == "Create Your Own Form":
        # Form for users to input their own data
        st.sidebar.write("Enter the following values:")

        hour = st.sidebar.number_input("Hour (0-23)", min_value=0, max_value=23)
        day = st.sidebar.number_input("Day (1-31)", min_value=1, max_value=31)
        month = st.sidebar.number_input("Month (1-12)", min_value=1, max_value=12)
        weekday = st.sidebar.number_input("Weekday (0=Monday, 6=Sunday)", min_value=0, max_value=6)
        holiday = st.sidebar.selectbox("Holiday", ["None", "Holiday"])
        weather_main = st.sidebar.selectbox("Weather", ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Snow", "Thunderstorm"])

        # Submit form
        submit_button = st.sidebar.button(label="Submit")
        
        if submit_button:
            # Create a DataFrame from user input
            user_data = {
                "hour": [hour],
                "day": [day],
                "month": [month],
                "weekday": [weekday],
                "holiday": [holiday],
                "weather_main": [weather_main],
            }
            user_df = pd.DataFrame(user_data)

            # Preprocess the input data
            user_df = preprocess_data(user_df)

            # Make predictions on the user data
            predictions = mapie.predict(user_df, alpha=alpha)

            # Display the predictions
            prediction_df = pd.DataFrame({
                "Predicted Traffic Volume": predictions[0].flatten(),
                "Lower Prediction Limit": predictions[1][:, 0].flatten(),
                "Upper Prediction Limit": predictions[1][:, 1].flatten()
            })
            st.sidebar.write("### Predictions for Your Input Data")
            st.sidebar.dataframe(prediction_df)

    # Main tab for visualizations
    visualization_tab = st.radio("Select Visualization", ["Feature Importance", "Residuals", "Predicted vs Actual", "Coverage"])

    if visualization_tab == "Feature Importance":
        st.subheader("Feature Importance")
        st.image("feature_importance.png", use_container_width=True)

    elif visualization_tab == "Residuals":
        st.subheader("Residuals")
        st.image("residuals.png", use_container_width=True)

    elif visualization_tab == "Predicted vs Actual":
        st.subheader("Predicted vs Actual")
        st.image("pred_vs_actual.png", use_container_width=True)

    elif visualization_tab == "Coverage":
        st.subheader("Coverage")
        st.image("coverage.png", use_container_width=True)

# Load model function (ensure you have your model loading logic here)
def load_model():
    # Replace this with your actual model loading code (this is a placeholder)
    xgb_model = xgb.XGBRegressor()  # Placeholder model
    mapie = MapieRegressor(xgb_model)
    return xgb_model, mapie

if __name__ == "__main__":
    main()
