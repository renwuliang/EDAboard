# Streamlit Packages
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    st.title("AUTO ML")

    activites= ["EDA", "Plot", "Model building", "About"]

    choice= st.sidebar.selectbox("select activity", activites)
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data=  st.file_uploader("Upload dataset", type=["csv", "txt"])
        if data is not None:
            # Perform EDA
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Show shape"):
                st.write(df.shape)

            if st.checkbox("Show colums"):
                all_columns= df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Select columns to show"):
                all_columns= df.columns.to_list()
                selected_columns= st.multiselect("Select columns", all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Show summary"):
                st.write(df.describe())

            if st.checkbox("Show value count"):
                st.write(df.iloc[:,-1].value_counts())



    elif choice=='Plot':
        st.subheader("Data visualization")

        data=  st.file_uploader("Upload dataset", type=["csv", "txt"])
        if data is not None:
            # Perform EDA
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox("Correlation with seaborn"):
                # st.write(sns.heatmap(df.corr(), annot=True))
                # st.pyplot()

                fig, ax = plt.subplots(figsize=(12, 12))

                # Plot the correlation heatmap
                correlation_matrix = df.corr()
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

                # Show the Matplotlib figure in Streamlit
                st.pyplot(fig)

            if st.checkbox("Pie chat"):
                all_columns= df.columns.to_list()
                columns_to_plot= st.selectbox("Select 1 columns", all_columns)
                # Create a Matplotlib figure explicitly
                fig, ax = plt.subplots()
                df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                
                # Show the Matplotlib figure in Streamlit
                st.pyplot(fig)
                # pie_plot= df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                # st.write(pie_plot)
                # st.pyplot()

            # all_column_names= df.columns.to_list()
            # type_of_plot = st.selectbox








            if st.checkbox("Area Plot"):
                all_columns_names = df.columns.to_list()
                columns_to_plot = st.selectbox("Select X-axis for Area Plot", all_columns_names)
                fig, ax = plt.subplots(figsize=(10, 6))
                area_plot = df.plot.area(x=columns_to_plot, ax=ax)
                st.write(area_plot)
                st.pyplot(fig)

            if st.checkbox("Line Plot"):
                all_columns_names = df.columns.to_list()
                columns_to_plot = st.multiselect("Select columns for Line Plot", all_columns_names)
                fig, ax = plt.subplots(figsize=(10, 6))
                line_plot = df[columns_to_plot].plot.line()
                st.write(line_plot)
                st.pyplot(fig)

            if st.checkbox("Bar Plot"):
                all_columns_names = df.columns.to_list()
                columns_to_plot = st.selectbox("Select X-axis for Bar Plot", all_columns_names)
                bar_plot = df.plot.bar(x=columns_to_plot)
                st.write(bar_plot)
                st.pyplot()




            if st.checkbox("Scatter Plot"):
                x_column_scatter = st.selectbox("Select X-axis for Scatter Plot", df.columns)
                y_column_scatter = st.selectbox("Select Y-axis for Scatter Plot", df.columns)
                scatter_plot = df.plot.scatter(x=x_column_scatter, y=y_column_scatter)
                st.write(scatter_plot)
                st.pyplot()

            if st.checkbox("Histogram"):
                column_for_hist = st.selectbox("Select column for Histogram", df.columns)
                hist_plot = df[column_for_hist].plot.hist()
                st.write(hist_plot)
                st.pyplot()

            if st.checkbox("Box Plot"):
                column_for_boxplot = st.selectbox("Select column for Box Plot", df.columns)
                box_plot = df.boxplot(column=column_for_boxplot)
                st.write(box_plot)
                st.pyplot()






            if st.checkbox("Custom Plot"):
                st.subheader("Custom Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                x_column = st.selectbox("Select X-axis", df.columns)
                y_column = st.selectbox("Select Y-axis", df.columns)
                custom_plot = df.plot(x=x_column, y=y_column, ax=ax)
                st.write(custom_plot)
                st.pyplot(fig)








           
        
    elif choice=='Model building':
        st.subheader("Machine Learning")

        data=  st.file_uploader("Upload dataset", type=["csv", "txt"])
        if data is not None:
            # Perform EDA
            df = pd.read_csv(data)
            st.dataframe(df.head())












                # Allow user to check for missing values
            if st.checkbox("Check for missing values"):
                st.write(df.isnull().sum())

            # Allow user to select features (X) and target variable (Y)
            st.sidebar.subheader("Select Features and Target Variable")
            features_selected = st.sidebar.multiselect("Select features for X", df.columns)
            target_variable = st.sidebar.selectbox("Select target variable for Y", df.columns)

            X = df[features_selected]
            Y = df[target_variable]

            # Allow user to select models
            st.sidebar.subheader("Select Models")
            selected_models = st.sidebar.multiselect("Select models", ["LR", "LDA", "KNN", "CART", "NB", "SVM"])

            # Allow user to enter seed value
            seed = st.sidebar.number_input("Enter Seed Value", min_value=1, step=1, value=42)

            # Model Building
            models = {
                "LR": LogisticRegression(),
                "LDA": LinearDiscriminantAnalysis(),
                "KNN": KNeighborsClassifier(),
                "CART": DecisionTreeClassifier(),
                "NB": GaussianNB(),
                "SVM": SVC()
            }

            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'

            for name in selected_models:
                model = models[name]
                kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())

                accuracy_results = {"Model": name, "Accuracy": cv_results.mean(), "Standard Deviation": cv_results.std()}
                all_models.append(accuracy_results)

            # Display model results as a table
            st.subheader("Model Performance")
            st.dataframe(pd.DataFrame(all_models))

            # Allow user to download results as CSV
            # if st.button("Download Model Results as CSV"):
            #     download_link = generate_download_link(pd.DataFrame(all_models), filename="model_results.csv", key="model_results")
            #     st.markdown(download_link, unsafe_allow_html=True)

            


            def train_selected_models(X, Y, selected_models, seed):
                models = {
                    "LR": LogisticRegression(),
                    "LDA": LinearDiscriminantAnalysis(),
                    "KNN": KNeighborsClassifier(),
                    "CART": DecisionTreeClassifier(),
                    "NB": GaussianNB(),
                    "SVM": SVC()
                }

                training_results = []

                for name in selected_models:
                    model = models[name]
                    model.fit(X, Y)
                    accuracy = model.score(X, Y)

                    # Store training results
                    training_result = {"Model": name, "Accuracy": accuracy}
                    training_results.append(training_result)

                    # Display training results
                    st.write(f"{name} trained successfully. Accuracy: {accuracy:.2f}")

                # Display results as a table
                st.subheader("Training Results")
                st.dataframe(pd.DataFrame(training_results))

            # Allow user to choose models for training
            st.subheader("Train Selected Models")
            train_selected_models(X, Y, selected_models, seed)

                # Allow user to download training results as CSV
#                 if st.button("Download Training Results as CSV"):
#                     download_link = generate_download_link(pd.DataFrame(training_results), filename="training_results.csv", key="training_results")
#                     st.markdown(download_link, unsafe_allow_html=True)

# def generate_download_link(df, filename="dataframe.csv", key="download_csv"):
#      csv = df.to_csv(index=False)


            














            # X = df.iloc[:,0:-1]
            # Y = df.iloc[:, -1]

            # st.sidebar.subheader("Model Parameters")
            # seed = st.sidebar.number_input("Enter Seed Value", min_value=1, step=1, value=42)


            # models = []
            # models.append(('LR', LogisticRegression()))
            # models.append(('LDA', LinearDiscriminantAnalysis()))
            # models.append(('KNN', KNeighborsClassifier()))
            # models.append(('CART', DecisionTreeClassifier()))
            # models.append(('NB', GaussianNB()))
            # models.append(('SVM', SVC()))

            
            # model_names = []
            # model_mean = []
            # model_std = []
            # all_models = []
            # scoring = 'accuracy'

            # for name, model in models:
            #     kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
            #     cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            #     model_names.append(name)
            #     model_mean.append(cv_results.mean())
            #     model_std.append(cv_results.std())

            #     accuracy_results = {"model_name": name, "model_accuracy": cv_results.mean(), "standard_deviation": cv_results.std()}
              
            #     all_models.append(accuracy_results)

            # if st.checkbox("metrics as table"):
            #     st.dataframe(pd.DataFrame(zip(model_names, model_mean, model_std), columns=["Model_name", "model_mean", "model_std"]))

    
                    






#             X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#             st.sidebar.subheader("Model Parameters")
#             seed = st.sidebar.number_input("Enter Seed Value", min_value=1, step=1, value=42)

#             model_results = build_models(X_train, Y_train, X_test, Y_test, seed)

#             st.subheader("Model Accuracy:")
#             for model, accuracy in model_results.items():
#                 st.write(f"{model}: {accuracy}")










#     elif choice=='About':
#         st.subheader("About")



#         df_size, df_shape, df_columns, df_targetname, df_featurenames, df_Xfeatures, df_Ylabels = perform_eda(df)

        


# # EDA function
# def perform_eda(df):
#     df_size = df.size
#     df_shape = df.shape
#     df_columns = list(df.columns)
#     df_targetname = df[df.columns[-1]].name
#     df_featurenames = df_columns[0:-1]
#     df_Xfeatures = df.iloc[:, 0:-1]
#     df_Ylabels = df[df.columns[-1]]

#     return df_size, df_shape, df_columns, df_targetname, df_featurenames, df_Xfeatures, df_Ylabels


# # Model Building function



# # Streamlit App
# def main():
#     st.title("Data Upload and Analysis with Streamlit")

#     # File Upload
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

#     if uploaded_file is not None:
#         # Perform EDA
#         df = pd.read_csv(uploaded_file)
#         df_size, df_shape, df_columns, df_targetname, df_featurenames, df_Xfeatures, df_Ylabels = perform_eda(df)

#         # Model Building
#         model_results, model_names, allmodels = build_models(df_Xfeatures, df_Ylabels)

#         # Display EDA and Model Results
#         st.subheader("Exploratory Data Analysis:")
#         st.write(f"Data Size: {df_size}")
#         st.write(f"Data Shape: {df_shape}")
#         st.write(f"Columns: {df_columns}")
#         st.write(f"Target Column Name: {df_targetname}")

#         st.subheader("Model Results:")
#         for model_result in allmodels:
#             st.write(model_result)

#         # Save Data to Database
#         st.subheader("Save Data to Database:")
#         if st.button("Save to Database"):
#             # You can add your database saving logic here
#             st.success("Data saved to database!")

#         # Display the DataFrame
#         st.subheader("Display DataFrame:")
#         st.write(df)

if __name__ == "__main__":
    main()






# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn import metrics

# def main():
#     st.title("Machine Learning App")

#     menu = ["Exploratory Data Analysis", "Model Building", "Predictions"]
#     choice = st.sidebar.selectbox("Select Task", menu)

#     if choice == "Exploratory Data Analysis":
#         eda()
#     elif choice == "Model Building":
#         model_building()
#     elif choice == "Predictions":
#         st.subheader("Make Predictions")

        
#             # Allow the user to make predictions using a pre-trained model
#             if st.checkbox("Use pre-trained model for predictions"):
#                 # Load the pre-trained model
#                 model = load_pretrained_model()  # Define this function to load your pre-trained model

#                 # Select features for predictions
#                 features = st.multiselect("Select features for predictions", df.columns)

#                 # Make predictions
#                 predictions = model.predict(df[features])

#                 # Display predictions
#                 st.write("Predictions:")
#                 st.write(predictions)

# def model_building():
#     st.subheader("Model Building")

#     data = st.file_uploader("Upload dataset", type=["csv", "txt"])
#     if data is not None:
#         df = pd.read_csv(data)
#         st.dataframe(df.head())

#         # Allow user to check for missing values
#         if st.checkbox("Check for missing values"):
#             st.write(df.isnull().sum())

#         # Allow user to select features (X) and target variable (Y)
#         st.sidebar.subheader("Select Features and Target Variable")
#         features_selected = st.sidebar.multiselect("Select features for X", df.columns)
#         target_variable = st.sidebar.selectbox("Select target variable for Y", df.columns)

#         X = df[features_selected]
#         Y = df[target_variable]

#         # Allow user to select models
#         st.sidebar.subheader("Select Models")
#         selected_models = st.sidebar.multiselect("Select models", ["LR", "LDA", "KNN", "CART", "NB", "SVM"])

#         # Allow user to enter seed value
#         seed = st.sidebar.number_input("Enter Seed Value", min_value=1, step=1, value=42)

#         # Model Building
#         models = {
#             "LR": LogisticRegression(),
#             "LDA": LinearDiscriminantAnalysis(),
#             "KNN": KNeighborsClassifier(),
#             "CART": DecisionTreeClassifier(),
#             "NB": GaussianNB(),
#             "SVM": SVC()
#         }

#         model_names = []
#         model_mean = []
#         model_std = []
#         all_models = []
#         scoring = 'accuracy'

#         for name in selected_models:
#             model = models[name]
#             kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
#             cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#             model_names.append(name)
#             model_mean.append(cv_results.mean())
#             model_std.append(cv_results.std())

#             accuracy_results = {"Model": name, "Accuracy": cv_results.mean(), "Standard Deviation": cv_results.std()}
#             all_models.append(accuracy_results)

#         # Display model results as a table
#         st.subheader("Model Performance")
#         st.dataframe(pd.DataFrame(all_models))

#         # Allow user to download results as CSV
#         if st.button("Download Model Results as CSV"):
#             download_link = generate_download_link(pd.DataFrame(all_models), filename="model_results.csv", key="model_results")
#             st.markdown(download_link, unsafe_allow_html=True)

#         # Allow user to choose models for training
#         st.subheader("Train Selected Models")
#         train_selected_models(X, Y, selected_models, seed)

# def train_selected_models(X, Y, selected_models, seed):
#     models = {
#         "LR": LogisticRegression(),
#         "LDA": LinearDiscriminantAnalysis(),
#         "KNN": KNeighborsClassifier(),
#         "CART": DecisionTreeClassifier(),
#         "NB": GaussianNB(),
#         "SVM": SVC()
#     }

#     for name in selected_models:
#         model = models[name]
#         model.fit(X, Y)
#         st.write(f"{name} trained successfully.")

# def generate_download_link(df, filename="dataframe.csv", key="download_csv"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}" key="{key}">Download CSV</a>'

# if __name__ == "__main__":
#     main()
