import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

def get_db_connection():
    """
    Establishes a connection to a MySQL database.
    
    Returns:
    mysql.connector.connection_cext.CMySQLConnection: Database connection object.
    """
    return mysql.connector.connect(
        host='your_host',      
        user='your_username',   
        password='your_password',
        database='your_database'
    )

# the below function fetches all documents from the database. For fetching documents from a specific dataset, refer to the alternative function in the README.
def fetch_documents_for_training(conn):
    """
    Fetches documents from the database that need category predictions, 
    along with associated annotator labels and reliability scores for training.
    
    Parameters:
    conn (mysql.connector.connection_cext.CMySQLConnection): Active database connection.
    
    Returns:
    pandas.DataFrame: DataFrame containing columns for document_id, text, category_id, and reliability.
    """
    query = """
    SELECT d.id, d.document_text, al.category_id, u.reliability
    FROM documents d
    JOIN annotatorLabels al ON d.id = al.document_id
    JOIN users u ON al.annotator_user_id = u.id
    WHERE d.predicted_category_id IS NULL
    """
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    return pd.DataFrame(result, columns=['document_id', 'text', 'category_id', 'reliability'])

def compute_weighted_labels(docs_df):
    """
    Computes weighted labels for each document based on annotator reliability scores.
    
    Parameters:
    docs_df (pandas.DataFrame): DataFrame containing document IDs, category IDs, and reliability scores.
    
    Returns:
    pandas.DataFrame: DataFrame containing document_id and predicted_category_id with weighted category labels
    """
    weighted_scores = docs_df.assign(weighted_score=docs_df['reliability'] * docs_df['category_id'])
    weighted_labels = weighted_scores.groupby(['document_id']).apply(
        lambda x: (x['weighted_score'].sum() / x['reliability'].sum()).round().astype(int)
    ).reset_index(name='predicted_category_id')
    return weighted_labels

def train_model(docs_df, weighted_labels):
    """
    Trains an XGBoost classifier on the provided document text and categories. Used TfidfVectorizer to vectorize text.
    
    Parameters:
    docs_df (pandas.DataFrame): DataFrame containing document texts.
    weighted_labels (pandas.DataFrame): DataFrame containing document IDs and weighted categories.
    
    Returns:
    tuple: Tuple containing the trained XGBoost model and TfidfVectorizer instance.
    """
    training_data = docs_df.merge(weighted_labels, on='document_id')
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(training_data['text'])
    y_train = training_data['predicted_category_id']
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model, vectorizer

def predict_and_update(conn, model, vectorizer, docs_df):
    """
    Predicts categories for documents using the trained model and updates the database.
    
    Parameters:
    conn (mysql.connector.connection_cext.CMySQLConnection): Active database connection.
    model (xgboost.XGBClassifier): Trained XGBoost classifier.
    vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): Fitted TF-IDF vectorizer.
    docs_df (pandas.DataFrame): DataFrame containing document IDs and text.
    
    Returns:
    None
    """
    X_pred = vectorizer.transform(docs_df['text'])
    predictions = model.predict(X_pred)
    cursor = conn.cursor()
    update_query = "UPDATE documents SET predicted_category_id = %s WHERE id = %s"
    update_data = [(pred, doc_id) for pred, doc_id in zip(predictions, docs_df['document_id'])]
    cursor.executemany(update_query, update_data)
    conn.commit()

def train_and_predict(conn):
    """
    Manages the process of fetching, training, predicting, and updating document categories.
    
    Parameters:
    conn (mysql.connector.connection_cext.CMySQLConnection): Active database connection.
    
    Returns:
    str: Message indicating the number of documents updated with predictions.
    """
    docs_df = fetch_documents_for_training(conn)
    if docs_df.empty:
        return "No documents to process."

    weighted_labels = compute_weighted_labels(docs_df)
    model, vectorizer = train_model(docs_df, weighted_labels)
    predict_and_update(conn, model, vectorizer, docs_df)
    return f"{len(docs_df)} documents updated with predictions."

def main():
    """
    Main function to initialize database connection and execute training and prediction process.
    """
    conn = get_db_connection()
    try:
        result = train_and_predict(conn)
        print(result)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
