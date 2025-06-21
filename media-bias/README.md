Article-Bias checker
---

### Screenshot

![image](https://github.com/user-attachments/assets/0f14c221-3db8-4267-b883-c702f4874697)

![image](https://github.com/user-attachments/assets/7493c473-8f17-424e-9309-f58784ad3ec8)



---

## Key Features

*   **Bias Prediction**: Analyzes input text for left-leaning or right-leaning bias using a trained logistic regression model.
*   **Confidence Score**: Displays the model's confidence in its prediction.
*   **Contrasting Viewpoints**: Automatically finds and displays relevant left-leaning and right-leaning articles from the dataset to provide a balanced view.
*   **Relevance Matching**: Uses TF-IDF and cosine similarity, boosted by keyword matching, to find the most relevant contrasting articles.
*   **Interactive UI**: A clean, modern, and responsive user interface built with Flask and vanilla JavaScript.

---

## Technology Stack

*   **Backend**: Python, Flask
*   **Machine Learning**: Scikit-learn, Pandas, NumPy
*   **Frontend**: HTML5, CSS3, JavaScript (no frameworks)
*   **Data Source**: A `scraped_articles.csv` file containing phrases, outlets, and bias labels.

---

## Setup and Installation

To run this project locally, follow these steps:

**1. Clone the repository:**
git clone https://github.com/gayathri2210/article-bias-check.git
cd article-bias-check

text

**2. Create and activate a virtual environment:**

*   **On macOS/Linux:**
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
*   **On Windows:**
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```

**3. Install the required dependencies:**
   
pip install Flask scikit-learn pandas newspaper3k flask-cors



**4. Run the application:**
python app.py

text

**5. Open in your browser:**
Navigate to `http://127.0.0.1:5000` to use the application.

---

## How It Works

1.  **Model Training**: At startup, the application loads the `scraped_articles.csv` dataset. It then trains a Logistic Regression classifier using `TfidfVectorizer` on the text phrases to learn the patterns associated with left and right-leaning text.

2.  **User Input**: When a user submits text, the Flask backend receives it via an API endpoint.

3.  **Prediction**: The input text is transformed using the same `TfidfVectorizer` and fed into the trained model, which predicts the bias and its confidence level.

4.  **Article Matching**: To find contrasting articles, the system calculates the cosine similarity between the user's input vector and all article vectors in the dataset. The similarity scores are boosted for articles that contain key terms from the input text, ensuring high relevance.

5.  **Display**: The prediction and the top-matching left and right articles are sent back to the frontend and displayed dynamically to the user.

---


