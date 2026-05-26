import pandas as pd 
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#-------- Get the CSV from the session_state --------

df_clean = st.session_state.get("new_df_clean")
df_model = st.session_state.get("new_df_model")

if df_clean is None: # guard for the df_clean
    st.warning("Please complete Data Loading first.")
    st.stop()

if df_model is None: # guard for the df_model
    st.warning("Please complete Data Loading first.")
    st.stop()
    

#-------- Title --------

st.title("ML Pipeline")
st.caption("Build and evaluate your machine learning models with ease.")

# Target Features and Target Variables (user picks)
st.info(
    "Target variables are restricted to avoid feature leakage and unrealistic model performance."
)

allowed_targets = ["AST", "TRB", "STL", "BLK", "TOV", "PTS"]

target_variable = st.selectbox(
    "Select Target Variable",
    options=allowed_targets,
    key="target_variable"
)
default_features = [col for col in df_clean.columns if col != target_variable] # set default 
target_features = st.multiselect(
    "Select Target Feature",
    options=[col for col in df_clean.columns if col != target_variable],
    default=default_features,
    key="target_features"
)


# Save the target variable and features in session state for EDA page

if not target_features: # guard 
    st.warning("Please select atleast one feature to continue.")
    st.stop()


# Saving with "saved_" prefix to avoid conflicts with widget keys — widget keys can reset during page navigation

st.session_state["saved_target_variable"] = target_variable 
st.session_state["saved_target_features"] = target_features 

#-------- Get targets from the session_state --------   
    
target_variable = st.session_state.get("saved_target_variable") # get the target_variable from the "Data_Loading" page
target_features = st.session_state.get("saved_target_features") # get the target_features from the "Data_Loading" page

#-------- Guards for the targets --------

if target_variable is None: 
    st.warning("Please select your Target Variable first.")
    st.stop()
    
if target_features is None: 
    st.warning("Please select your Target Features first.")
    st.stop()    
    
if target_variable not in df_model.columns:
    st.warning("Target variable not found in the model data. Please select a numeric target.")
    st.stop()

#----------------------------
#-------- Regression --------
#----------------------------

st.title("Regression Model Comparison")


leakage_cols = ["Player", "Pos", "Tm"]

target_features = [
    col for col in target_features
    if col not in leakage_cols
]

# Use selected features from the clean dataframe
x = df_clean[target_features].copy()
y = df_clean[target_variable].copy()

# Convert categorical selected features into numeric columns
x = pd.get_dummies(x, drop_first=True)



st.subheader("Model Selection")
user_regression_model = st.selectbox("Select a Regression Model", options=["Linear Regression", "Random Forest"], key="reg_model")

#-------- User's Parameters --------

# for linear both models
test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# defaults
user_n_estimators = 100
user_max_depth = 10
if user_regression_model == "Random Forest":
    # for random forest
    st.text("For Random Forest:")
    user_n_estimators = st.slider("Select how many trees you want:", min_value = 10, max_value = 200, value = 100, step = 10)
    user_max_depth = st.slider("Select the depth of the trees:", min_value = 2, max_value = 20, value = 10, step = 1)

# random_state=42 ensures reproducible results — same split every time the code runs
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

#-------- Linear Regression --------

lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
lr_y_pred = lr_model.predict(x_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))
lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)


#-------- Random Forest --------

# oob_score=True enables out-of-bag evaluation — uses data not seen by each tree as a built-in validation method
rf_model = RandomForestRegressor(
    n_estimators = user_n_estimators,
    random_state=42,
    oob_score = True,
    max_depth=user_max_depth
    )

rf_model.fit(x_train, y_train)
importance_df = pd.DataFrame({
    "Feature": x.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.subheader("Feature Importance")
st.dataframe(importance_df.head(10))

fig = px.bar(
    importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig)
rf_y_pred = rf_model.predict(x_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)


#-------- RMSE, R2 & Visualizations -------- 

if user_regression_model == "Linear Regression":
    st.subheader(" Linear Regression Measures")
    st.write("Root Mean Squared Error:", lr_rmse)
    st.write("Mean Absolute Error:", lr_mae)
    st.write("R2 score:", lr_r2)
    fig = px.scatter(x=y_test, y=lr_y_pred,
                    labels={'x': 'Actual', 'y':'Predicted'})
    st.plotly_chart(fig)
    
elif user_regression_model == "Random Forest":
    st.subheader("Random Forest Measures")
    st.write("Root Mean Squared Error:", rf_rmse)
    st.write("Mean Absolute Error:", rf_mae)
    st.write("R2 score:", rf_r2)
    fig = px.scatter(x=y_test, y=rf_y_pred,
                    labels={'x': 'Actual', 'y':'Predicted'})
    st.plotly_chart(fig)
    

#-------- Model Comparison --------    

st.subheader("Regression Model Comparison")
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [lr_rmse, rf_rmse],
    'MAE': [lr_mae, rf_mae],
    'R2': [lr_r2, rf_r2]
})
st.dataframe(comparison)

#----------------------------
#----- Classification -------
#----------------------------

st.divider()

st.title("Classification Model Comparison")
st.caption("Predict player role based on selected statistical features.")

classification_features = [
    "AST",
    "TRB",
    "STL",
    "BLK",
    "FG%",
    "3P%",
    "FT%",
    "MP",
    "PTS"
]

# Keep only features that exist in the dataset
classification_features = [
    col for col in classification_features
    if col in df_clean.columns
]

x_class = df_clean[classification_features].copy()

# One-hot encode if needed
x_class = pd.get_dummies(x_class, drop_first=True)


# Use real player position as the classification target
position_col = "Pos"

if position_col not in df_clean.columns:
    st.warning("Position column was not found in the dataset.")
    st.stop()

y_class = df_clean[position_col].copy()

# Convert detailed NBA positions into 3 broader player roles
def map_position(pos):
    pos = str(pos)

    if "PG" in pos or "SG" in pos:
        return "Guard"
    elif "SF" in pos or "PF" in pos:
        return "Forward"
    elif "C" in pos:
        return "Center"
    else:
        return "Other"

y_class = y_class.apply(map_position)

# Remove unknown/unsupported positions
valid_rows = y_class != "Other"

x_class = x_class.loc[valid_rows]
y_class = y_class.loc[valid_rows]

# Guard against very small classes
class_counts = y_class.value_counts()
valid_classes = class_counts[class_counts >= 2].index

x_class = x_class[y_class.isin(valid_classes)]
y_class = y_class[y_class.isin(valid_classes)]

st.write("Classification target distribution:")
st.dataframe(y_class.value_counts())

st.subheader("Model Selection")

user_classification_model = st.selectbox(
    "Select a Classification Model",
    options=["Decision Tree", "K-Nearest Neighbors"],
    key="class_model"
)

#-------- User's Parameters --------

test_size = st.slider(
    "Test size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    key="class_test_size"
)

user_max_depth_class = 10
user_n_neighbors = 5

if user_classification_model == "Decision Tree":
    st.text("For Decision Tree:")
    user_max_depth_class = st.slider(
        "Select the depth of the tree:",
        min_value=2,
        max_value=20,
        value=10,
        step=1,
        key="max_depth_class"
    )

elif user_classification_model == "K-Nearest Neighbors":
    st.text("For K-Nearest Neighbors:")
    user_n_neighbors = st.slider(
        "Select the number of neighbors:",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="n_neighbors"
    )

x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
    x_class,
    y_class,
    test_size=test_size,
    random_state=42,
    stratify=y_class
)

#-------- Decision Tree --------

dt_model = DecisionTreeClassifier(
    max_depth=user_max_depth_class,
    random_state=42
)

dt_model.fit(x_train_class, y_train_class)
dt_y_pred = dt_model.predict(x_test_class)

dt_accuracy = accuracy_score(y_test_class, dt_y_pred)
dt_f1 = f1_score(y_test_class, dt_y_pred, average="weighted")

#-------- K-Nearest Neighbors --------

knn_model = KNeighborsClassifier(n_neighbors=user_n_neighbors)

knn_model.fit(x_train_class, y_train_class)
knn_y_pred = knn_model.predict(x_test_class)

knn_accuracy = accuracy_score(y_test_class, knn_y_pred)
knn_f1 = f1_score(y_test_class, knn_y_pred, average="weighted")

#-------- Accuracy & Visualizations --------

labels = ["Guard", "Forward", "Center"]

if user_classification_model == "Decision Tree":
    st.subheader("Decision Tree Measures")
    st.write("Accuracy:", dt_accuracy)
    st.write("F1 Score:", dt_f1)

    cm = confusion_matrix(y_test_class, dt_y_pred, labels=labels)

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        labels={"x": "Predicted", "y": "Actual"}
    )

    st.plotly_chart(fig)

elif user_classification_model == "K-Nearest Neighbors":
    st.subheader("K-Nearest Neighbors Measures")
    st.write("Accuracy:", knn_accuracy)
    st.write("F1 Score:", knn_f1)

    cm = confusion_matrix(y_test_class, knn_y_pred, labels=labels)

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        labels={"x": "Predicted", "y": "Actual"}
    )

    st.plotly_chart(fig)

#-------- Model Comparison --------

st.subheader("Classification Model Comparison")

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "K-Nearest Neighbors"],
    "Accuracy": [dt_accuracy, knn_accuracy],
    "F1 Score": [dt_f1, knn_f1]
})

st.dataframe(comparison)


#----------------------------
#-------- Clustering --------
#----------------------------

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.divider()
st.title("Clustering Analysis")
st.caption("Group players into types based on their statistical profile.")

# Features για clustering
clustering_features = ["PTS", "AST", "TRB", "STL", "BLK", "FG%", "MP"]
clustering_features = [col for col in clustering_features if col in df_clean.columns]

x_cluster = df_clean[clustering_features].dropna()

# Scaling — απαραίτητο για K-Means και DBSCAN
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_cluster)

st.subheader("K-Means Clustering")
n_clusters = st.slider("Number of clusters (K):", min_value=2, max_value=8, value=3, step=1)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(x_scaled)

silhouette_kmeans = silhouette_score(x_scaled, kmeans_labels)
st.write("Silhouette Score:", silhouette_kmeans)

# Προσθήκη cluster label στα δεδομένα
x_cluster = x_cluster.copy()
x_cluster["KMeans_Cluster"] = kmeans_labels
x_cluster["Player"] = df_clean.loc[x_cluster.index, "Player"]

fig = px.scatter(
    x_cluster,
    x="PTS",
    y="AST",
    color=x_cluster["KMeans_Cluster"].astype(str),
    hover_data=["Player"],
    title="K-Means: Points vs Assists"
)
st.plotly_chart(fig)

st.subheader("DBSCAN Clustering")
eps = st.slider("Epsilon (neighborhood size):", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
min_samples = st.slider("Min samples:", min_value=2, max_value=20, value=5, step=1)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(x_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

st.write("Clusters found:", n_clusters_dbscan)
st.write("Noise points:", n_noise)

if n_clusters_dbscan > 1:
    silhouette_dbscan = silhouette_score(x_scaled, dbscan_labels)
    st.write("Silhouette Score:", silhouette_dbscan)
else:
    silhouette_dbscan = -1
    st.warning("DBSCAN found only 1 cluster — try adjusting epsilon or min samples.")

x_cluster["DBSCAN_Cluster"] = dbscan_labels

fig = px.scatter(
    x_cluster,
    x="PTS",
    y="AST",
    color=x_cluster["DBSCAN_Cluster"].astype(str),
    hover_data=["Player"],
    title="DBSCAN: Points vs Assists"
)
st.plotly_chart(fig)

#-------- Clustering Comparison --------

st.subheader("Clustering Model Comparison")
clustering_comparison = pd.DataFrame({
    "Model": ["K-Means", "DBSCAN"],
    "Clusters Found": [n_clusters, n_clusters_dbscan],
    "Silhouette Score": [silhouette_kmeans, silhouette_dbscan],
    "Noise Points": [0, n_noise]
})
st.dataframe(clustering_comparison)