
# USA Travel Airfare Prediction App

## Overview
This project hosts a Streamlit application designed to predict air travel fares within the USA. Users can input details about their trip, such as origin and destination airports, departure date and time, and cabin type, to receive an estimated fare for their journey.

Streamlit URL - [https://advmlaat3.streamlit.app/)](https://advmlaat3.streamlit.app/)

## Models
The application integrates four different predictive models, each developed by a member of our team. These models have been trained on a comprehensive dataset of flight fares and are capable of providing quick and accurate fare predictions.


|Model Name|Team Member|Student ID|
|--|--|--|
|TensorFlow|Kritika Dhawale|24587661|
|CatBoost|Sahil Kotak|24707592|
|Gradient Boost|Varun Chhetri|24711703|
|XGBoost|Anika Chauhan|14188775|



## Repository Structure
- `notebooks/`: Contains all the Jupyter notebooks with naming convention `<lastname>_<firstname>-<student_id>_<description>.ipynb`.
- `models/`: Stores the serialized versions of our best predictive models.
- `src/models/`: Includes scripts for training and loading the predictive models.
- `reports/`: Contains experiment reports detailing the modeling process and results.

## Getting Started

### Prerequisites
- scikit-learn==1.2.1
- pandas==2.1.1
- numpy>=1.23.4
- joblib==1.2.0
- tensorflow==2.14.0
- streamlit==1.14.0

### Installation
Clone the repository and navigate to the project directory. Install the required dependencies using pip:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### Running the Streamlit App
To start the Streamlit app, navigate to the project directory in your terminal and run:

```bash
streamlit run app/streamlit_app.py
```
Once the app is running, it will be accessible in your web browser at the address indicated by Streamlit, typically http://localhost:8501.

## Usage

To use the application:

1.  Open the Streamlit app - [https://advmlaat3.streamlit.app/](https://advmlaat3.streamlit.app/) in your web browser.
2.  Enter the details of your trip into the provided input fields.
3.  Click the 'Predict' button to receive fare estimates from each of the four models.

## Contributing

We encourage contributions from all members of the GitHub community. If you would like to contribute to this project, please take the following steps:

1.  Fork the project repository.
2.  Create a new feature branch (`git checkout -b feature/YourFeature`).
3.  Make your changes to the codebase.
4.  Commit your changes (`git commit -m 'Add some YourFeature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a new Pull Request.


## License

This project is licensed under the MIT License.

## Authors

-   **Anika Chauhan** - [GitHub](https://github.com/anika)
-   **Kritika Dhawale** - [GitHub](https://github.com/Kritz23)
-   **Sahil Kotak** - sahil.kotak@student.uts.edu - [GitHub](https://github.com/sahilkotak)
-   **Varun Chhetri** - [GitHub](https://github.com/varun)
