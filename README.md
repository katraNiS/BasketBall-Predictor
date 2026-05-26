# Basketball Predictor

Εφαρμογή Machine Learning για ανάλυση απόδοσης παικτών NBA, κατασκευασμένη με Streamlit και deployed στο Streamlit Community Cloud.

## Περιγραφή

Η εφαρμογή αναπτύχθηκε στο πλαίσιο του μαθήματος **Πλατφόρμες και Αρχιτεκτονικές Νέφους**. Επιτρέπει στον χρήστη να ανεβάσει dataset στατιστικών NBA, να το εξερευνήσει οπτικά και να εφαρμόσει μεθόδους Machine Learning για πρόβλεψη και ανάλυση απόδοσης παικτών.

## Live Εφαρμογή

https://bj7enbteflyrvpnpujlbkl.streamlit.app

## Λειτουργίες

- **Φόρτωση & Προεπεξεργασία** — Ανέβασμα CSV, χειρισμός missing values, αφαίρεση duplicates, scaling
- **EDA** — Histogram, Box Plot, Scatter Plot, Correlation Heatmap, PCA
- **Regression** — Linear Regression vs Random Forest για πρόβλεψη assists παίκτη
- **Classification** — Decision Tree vs KNN για κατηγοριοποίηση παικτών σε Guard / Forward / Center
- **Clustering** — K-Means vs DBSCAN για ανακάλυψη τύπων παικτών

## Dataset

NBA Player Stats Regular Season 2023-24 από το Kaggle.

## Τεχνολογίες

| Εργαλείο | Χρήση |
|----------|-------|
| Python | Γλώσσα προγραμματισμού |
| Streamlit | Web app framework |
| Pandas / NumPy | Επεξεργασία δεδομένων |
| Scikit-learn | Machine Learning |
| Plotly | Διαδραστικά γραφήματα |
| Streamlit Community Cloud | Deployment |

## Τοπική Εκτέλεση

```bash
git clone https://github.com/katraNiS/BasketBall-Predictor
cd BasketBall-Predictor
pip install -r requirements.txt
streamlit run Home.py
```
