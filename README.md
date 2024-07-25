
# Backend ML Pipeline

## Setup

To set up your local environment:
1. Clone the repository

```bash
git clone https://repositorylink.git
cd your-project-directory
```

2. Install Python dependencies
```bash
pip install mysql-connector-python pandas scikit-learn xgboost
```
3. Configure database connection. Modify the database connection settings to match MySQL server credentials and database details.

4. The application can be executed by running the main Python script:
```python
python main.py
```

## Procedure

1. Establishes a Database Connection: Connects to MySQL database using credentials specified in the configuration settings
2. Fetches Document Data: Retrieves documents from the documents table that require category predictions. Specifically, it selects documents where predicted_category_id is null, indicating that these documents have not yet been categorized.
3. Joins with AnnotatorLabels: Performs an INNER JOIN with the annotatorLabels table on document_id. This join brings together each document's text with associated category IDs assigned by annotators.
4. Joins with Users Table: Includes another INNER JOIN with the users table on annotator_user_id to fetch the reliability scores of annotators.
5. Computes Weighted Averages: Computing of weighted averages of category labels based on reliability scores.
6. Trains the Model: Utilizes the XGBoost classifier with a TF-IDF vectorizer to train on fetched documents. The training data includes document texts and their associated weighted average category labels
7. Makes Predictions: After training, the model predicts categories for each document.
8. Updates the Database: Updates the documents table with the predicted category IDs.

# Additional Details:

If we want to predict on one specific dataset, fetch_documents_for_training should be modified to:

```python

def fetch_documents_for_training(conn, dataset_id):
    """
    Fetches documents from the database that need category predictions, 
    along with associated annotator labels and reliability scores for training,
    filtered by dataset ID.
    
    Parameters:
    conn (mysql.connector.connection_cext.CMySQLConnection): Active database connection.
    dataset_id (int): ID of the dataset to filter the documents by.
    
    Returns:
    pandas.DataFrame: DataFrame containing columns for document_id, text, category_id, and reliability.
    """
    query = """
    SELECT d.id, d.document_text, al.category_id, u.reliability
    FROM documents d
    JOIN annotatorLabels al ON d.id = al.document_id
    JOIN users u ON al.annotator_user_id = u.id
    WHERE d.predicted_category_id IS NULL AND d.dataset_id = %s
    """
    cursor = conn.cursor()
    cursor.execute(query, (dataset_id,))
    result = cursor.fetchall()
    return pd.DataFrame(result, columns=['document_id', 'text', 'category_id', 'reliability'])

```

If we want to update the database with predicted category names along with predicted category ids, use this function for prediction instead. This function assumes that
an additional column predicted_category_name currently exists in documents and needs to be populated in addition to predicted category ids.

```python

def predict_and_update(conn, model, vectorizer, docs_df):
    """
    Predicts categories for documents using the trained model and updates the database
    with both the category ID and the category name.

    Parameters:
    conn (mysql.connector.connection_cext.CMySQLConnection): Active database connection.
    model (xgboost.XGBClassifier): Trained XGBoost classifier.
    vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): Fitted TF-IDF vectorizer.
    docs_df (pandas.DataFrame): DataFrame containing document IDs and text.
    
    Returns:
    None
    """
    # predict categories
    X_pred = vectorizer.transform(docs_df['text'])
    predictions = model.predict(X_pred)
    
    update_query = """
    UPDATE documents d
    INNER JOIN categories c ON c.id = %s
    SET d.predicted_category_id = %s, d.predicted_category_name = c.category_name
    WHERE d.id = %s
    """
    cursor = conn.cursor()
    try:
        update_data = [(pred, pred, doc_id) for pred, doc_id in zip(predictions, docs_df['document_id'])]
        cursor.executemany(update_query, update_data)
        conn.commit()
        print(f"Updated {cursor.rowcount} documents with predicted categories.")
    except mysql.connector.Error as err:
        print("Failed to update documents:", err)
    finally:
        if cursor:
            cursor.close()

```
