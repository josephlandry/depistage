import pandas as pd                                   # pour l'importation et le conditionnement des données en dataframe
import numpy as np                                    # pour les calculs numériques
import seaborn as sns                                 # pour l'affichage des graphiques
import matplotlib.pyplot as plt
import tensorflow as tf                               # pour l'entrainement du réseau des neurones artificiels
import streamlit as st
import plotly.express as px  
import joblib                                         # pour la sauvegarde des modèles                        
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler    # pour la standardisation des données
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from tensorflow import keras
from keras.models import  Sequential                 # modèle de deep learning
from keras.layers import Dense, Flatten              # importation des différents modèles des couches de neurones
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, silhouette_score  # pour l'évaluation des modèles
from sklearn.model_selection import cross_val_score
from colorama import Fore, Back, Style, init
import time


st.sidebar.title("Prédiction du dépistage du VIH/SIDA chez les hommes en Côte d'Ivoire")
pages = ["Resumé du projet", "Déscription des données", "Apprentissage des modèles", "Prédictions"]


page=st.sidebar.selectbox('Choisir la page', pages)

@st.cache_data
def get_data(file):
    data = pd.read_excel(file)

    return data

@st.cache_data
def get_data(file1):
    data1 = pd.read_excel(file1)

    return data1
    
#df = pd.read_excel('base_codifiee.xlsx', 'Sheet1')

data = get_data('base_codifiee.xlsx')
data1 = get_data('Dictionnaire_variables.xlsx')

df = data
df_var=data1

df1 = df.fillna(df.mean())
#.astype(int)

df2 = df1[['mv001', 'mv013', 'mv024', 'mv025', 'mv035', 'mv106', 'mv107', 'mv130','mv131', 
        'mv133', 'mv149', 'mv150', 'mv155', 'mv157', 'mv158', 'mv159',
        'mv169a', 'mv169b', 'mv169c', 'mv170', 'mv171a', 'mv171b', 'mv176',
        'mv177', 'mv190', 'mv191', 'mv190a', 'mv191a', 'mv201', 'mv213',
        'mv217', 'mv218', 'mv750', 'mv751', 'mv754cp', 'mv754dp', 'mv754jp',
        'mv754wp', 'mv756', 'mv761', 'mv761b', 'mv762', 'mv762a', 'mv763a',
        'mv763b', 'mv763c', 'mv766a', 'mv766b', 'mv767a', 'mv767b', 'mv781',
        'mv785', 'mv822', 'mv824', 'mv825', 'mv836', 'mv837', 'mv857a',
        'mv859']]


#st.write(df2.isna().sum())


if page == "Resumé du projet" :

    st.title("Resumé")
    st.markdown("Ce travail est fait dans le but de construire un modèle de prédiction du dépistage du VIH/SIDA chez les hommes en Côte d'ivoire. Il s'agira ici de prédire, en tenant compte d'un certain nombre d'informations, si un homme faisant l'objet de l'enquête est susceptible d'accepter de se faire dépister ou pas au VIH/SIDA. Le modèle sorti de ce travail s'est basé sur les données d'enquête effectuée en 2021 par le programme DHS en Côte d ivoire")

    st.image("hiv_test1.jpg")
    st.image("hiv_test2.jpg")

    st.write('Par :')
    st.write("Landry KINTEBA") 
    #/ C. GNONGOUE / A. SAWADOGO / E. EBOUE / O. TUNDE")
    st.write('Master Data Engineer /Intelligence Artificielle')


elif page == "Déscription des données" :

    st.title("Déscription des données")
    st.subheader('Echantillon des données')
    
    st.write(df2.head(7591))    
    st.write("La base des données a", df2.shape[0], "observations")
    st.write("La base des données a", df2.shape[1], "variables")
    st.write("--------------------------------------------")

    st.subheader('Liste des variables')
    st.dataframe(df_var)


    st.write("---------------------------------------------")
    
    st.subheader('Déscription statistique des données')
        
    st.dataframe(df2.describe())
    
    st.write("--------------------------------------------")
    
    st.subheader("Nuage des données")

    cols2 = df2.drop(["mv781"], axis = 1).columns
    var_x = st.selectbox('Choisir la variable en abscisse', cols2)
    var_y = st.selectbox('Choisir la variable en ordonnée', cols2)
    mv781 = df2["mv781"].to_list()

    fig1 = px.scatter(
        df2,
        x=var_x,
        y=var_y,
        color=mv781,
        title=str(var_y) + " / " + str(var_x)
        )
    st.plotly_chart(fig1)
    
    st.write("--------------------------------------------")
    
    st.subheader('Histogrammes')

    cols3 = df2.columns
    var = st.selectbox('Choisir la variable', cols3)
    fig2, ax2 = plt.subplots()
    ax2 = sns.histplot(df2[var])
    plt.xlabel(var)
    
    st.pyplot(fig2)

    st.write("--------------------------------------------")
    
    st.subheader('Matrice de correlation de Pearson')
    
    clicked1 = st.button('Afficher')
    
    clicked2 = st.button('Reduire')
    
    if clicked1 :
        
        fig3, ax3 = plt.subplots()
        ax3 = sns.heatmap(df2.corr())
        st.pyplot(fig3)

    



elif page == "Apprentissage des modèles" :

    Modeles = ['LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier', "Artificial Neuronal Network"]

    Modele=st.sidebar.selectbox('Selectionner le modèle', Modeles)
    
    X = df2.drop(["mv781"], axis=1)
    y = df2["mv781"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    if Modele == "LogisticRegression" :

        st.header("Entrainement du modèle LogisticRegression")
        
        st.write("--------------------------------------------")

        model_lr = LogisticRegression()
        
        clicked11 = st.button('Entrainer')
    
        clicked12 = st.button('Arrêter')
    
        if clicked11 :
            
            model_lr.fit(X_train_scaled, y_train)
            y_pred_lr = model_lr.predict(X_test_scaled)

            st.subheader('Evaluation du modèle')
            st.write("accuracy score :", accuracy_score(y_test, y_pred_lr))
            st.write("F1 score :", f1_score(y_test, y_pred_lr))
            st.write(classification_report(y_test, y_pred_lr))
            
            st.subheader('Matrice de confusion')
            st.write(confusion_matrix(y_test, y_pred_lr))
            fig4, ax4 = plt.subplots()
            ax4 = sns.heatmap(confusion_matrix(y_test, y_pred_lr))
            st.pyplot(fig4)

    elif Modele == "RandomForestClassifier" :      
        
        st.header("Entrainement du modèle RandomForestClassifier")

        st.write("--------------------------------------------")

        model_rf = RandomForestClassifier()

        clicked13 = st.button('Entrainer')
    
        clicked14 = st.button('Arrêter')
    
        if clicked13 :

            model_rf.fit(X_train_scaled, y_train)
            y_pred_rf = model_rf.predict(X_test_scaled)

            st.subheader('Evaluation du modèle')
            st.write("accuracy score :", accuracy_score(y_test, y_pred_rf))
            st.write("F1 score :", f1_score(y_test, y_pred_rf))
            st.write(classification_report(y_test, y_pred_rf))

            st.subheader('Matrice de confusion')
            st.write(confusion_matrix(y_test, y_pred_rf))
            
            fig5, ax5 = plt.subplots()
            ax5 = sns.heatmap(confusion_matrix(y_test, y_pred_rf))
            st.pyplot(fig5)

            st.subheader('Feature importances')

            fig6, ax6 = plt.subplots()
            ax6 = plt.bar(X.columns, model_rf.feature_importances_)
            st.pyplot(fig6)
            
            

    elif Modele == "KNeighborsClassifier" :

        st.header("Entrainement du modèle KNeighborsClassifier")

        st.write("--------------------------------------------")
    
        model_kn = KNeighborsClassifier()

        clicked15 = st.button('Entrainer')
    
        clicked16 = st.button('Arrêter')
    
        if clicked15 :  
            
            model_kn.fit(X_train_scaled, y_train)
            y_pred_kn = model_kn.predict(X_test_scaled)

            st.subheader('Evaluation du modèle')
            st.write("accuracy score :", accuracy_score(y_test, y_pred_kn))
            st.write("F1 score :", f1_score(y_test, y_pred_kn))
            st.write(classification_report(y_test, y_pred_kn))
            
            st.subheader('Matrice de confusion')
            st.write(confusion_matrix(y_test, y_pred_kn))

            fig7, ax7 = plt.subplots()
            ax7 = sns.heatmap(confusion_matrix(y_test, y_pred_kn))
            st.pyplot(fig7)

    
    elif Modele == "Artificial Neuronal Network" :      
        
        st.header("Entrainement du modèle ANN")

        st.write("--------------------------------------------")


        model_ann = Sequential()

        model_ann.add(keras.layers.Flatten(input_shape=(1,58)))
        model_ann.add(keras.layers.Dense(4, activation="relu"))
        model_ann.add(keras.layers.Dense(8, activation="relu"))
        model_ann.add(keras.layers.Dense(4, activation="relu"))
        model_ann.add(keras.layers.Dense(1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model_ann.compile(loss="binary_crossentropy", optimizer=optimizer , metrics=["accuracy"])

        X_train_scaled_ann = X_train_scaled.reshape(-1, 1, 58)

        X_test_scaled_ann = X_test_scaled.reshape(-1, 1, 58)


        clicked17 = st.button('Entrainer')
    
        clicked18 = st.button('Arrêter')
    
        if clicked17 :

            history = model_ann.fit(X_train_scaled_ann, y_train, epochs=20, batch_size = 100 )

            acc = model_ann.evaluate(X_test_scaled_ann, y_test)

            print('accuracy :', acc)
            
            predictions = model_ann.predict(X_test_scaled_ann)
            prediction_labels =[1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]

            st.subheader('Evaluation du modèle')
            st.write("accuracy score :", accuracy_score(y_test, prediction_labels))
            st.write("F1 score :", f1_score(y_test, prediction_labels))
            st.write(classification_report(y_test, prediction_labels))

            st.subheader('Matrice de confusion')
            
            

            st.write(confusion_matrix(y_test, prediction_labels))
            
            
            fig8, ax8 = plt.subplots()
            ax8 = sns.heatmap(confusion_matrix(y_test, prediction_labels))
            st.pyplot(fig8)

            st.subheader('Courbes loss-accuracy')
            fig9, ax9 = plt.subplots()
            
            ax9 = plt.plot(range(1,21), history.history['loss'], label='loss', color='red')                   
            ax9 = plt.plot(range(1,21), history.history['accuracy'], label='acc', color='blue')
            ax9 = plt.xlabel("epochs")
            ax9 = plt.legend()

            st.pyplot(fig9)           

elif page == "Prédictions" :

    st.title("La Prédiction du dépistage")

    st.write('-----------------------------------')

    cols3 = ['mv001', 'mv013', 'mv024', 'mv106', 'mv107', 'mv130', 'mv131', 'mv133',
       'mv149', 'mv158', 'mv170', 'mv177', 'mv191', 'mv190a', 'mv191a',
       'mv201', 'mv217', 'mv218', 'mv836', 'mv837']
    
    X1=df3 = df2[cols3]

    y1 = df2["mv781"]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

    scaler = MinMaxScaler()
    X1_train_scaled = scaler.fit_transform(X1_train)
    X1_test_scaled = scaler.fit_transform(X1_test)

    st.subheader('Fiche des renseignements du candidat')

    a=[]
    
    a0 = st.number_input("***cluster number***", value=0, key=1)
    a1 = st.number_input("***age in 5-year groups***", value=0, key=2)
    st.write("15-19=1 / " , "20-24=2 / "  , "25-29=3 / "  , "30-34=4 / "  , "35-39=5 / "  , "40-44=6 / "  , "45-49=7 / "  , "50-54=8 / "  , "55-59=9 / " , "60-64=10")
    a2 = st.number_input("***region***", value = 0, key=3)
    st.write("abidjan=1 /", "yamoussoukro=2 / ", "bas sassandrra=3 / ", "comoe=4 / ", "denguele=5 /", "goh-djiboua=6 / ", "lacs=7")
    st.write("lagunes=8 / ", "montagnes=9 / ", "sassandra-marahoue=10 / ", "savanes=11 / ", "vallee du bandama = 12 / ", "woroba=13 / ", "zanzan=14")
    a3 = st.number_input("***educational level***", value = 0, key=4)
    st.write("no education=0  / ", "primary=1  / ", "secondary=2  / ", "higher=3")
    a4 = st.number_input("***highest year of education_at level in mv106***", value = 0, key=5)
    st.write("no years completed at level mv106=0")
    a5 = st.number_input("***religion***", value = 0, key=6)
    st.write("musilm=1/", "catholic=2/", "methodist=3/", "evangelical=4/", "other christian religion=5/", "animist=6/", "other=96/", "no religion=97")
    a6 = st.number_input("***ethnicity***", value = 0, key=7)
    st.write("abbey=101/", "abidji=102/", "aboure=103/", "abron=104/", "abjoukrou=105/", "agni=106/", "ahizi=107", "alladian=108/", "....", "worodougouka=189/", "other ivorian=994/", "not-ivorian=995/", "other ethnicity to be specified=996")
    a7 = st.number_input("***total number of years of education***", value=0, key=8)
    st.write("inconsistent=97")
    a8 = st.number_input("***educational attainment***", value=0, key=9)
    st.write("no education=0/", "incomplete primary=1/", "complete primary=2/", "incomplete secondary=3/", "complete secondary=4/", "higher=5")
    a9 = st.number_input("***frequency of listening to radio***", value=0, key=10)
    st.write("not at all=0/", "less than once a week=1/", "at least once a week=2/", "almost every day=3")
    a10 = st.number_input("***has an account in a bank or other financial in...***", 0, key=11)
    st.write("no=0/", "yes=1")
    a11 = st.number_input("***money put/taken from bank account last 12 months***", 0, key=12)
    st.write("no=0/", "yes=1")
    a12 = st.number_input("***wealth index factor score combined_5 decimals***", 0, key=13)
    a13 = st.number_input("***wealth index for urban/rural***", 0, key=14)
    st.write("poorest=1/", "poorer=2/", "middle=3/", "richer=4/", "richest=5")
    a14 = st.number_input("***wealth index factor score for urban/rural***", 0, key=15)
    a15 = st.number_input("***total children ever born***", 0, key=16)
    a16 = st.number_input("***knowledge of ovulatory cycle***", 0, key=17)
    st.write("during her period=1/", "after period ended=2/", "middle of the cycle=3/", "before period begins=4/", "at any time=5/", "other=6/", "don't know=8")
    a17 = st.number_input("***number of living children***", 0, key=18)
    a18 = st.number_input("***total lifetime number of sex partners***", 0, key=19)
    st.write("95+ = 95/", "don't know=98")
    a19 = st.number_input("***heard of drugs to help hiv infected people liv***", 0, key=20)
    st.write("no=0/", "yes=1/", "don't know=8")

    st.write("------------------------------------------------------")

    st.write('Liste des variables')
    st.dataframe(df_var)

    a = np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19])

    st.write("------------------------------------------------------")

    b = np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19]).reshape(1,-1)

    st.subheader("Les valeurs saisies sont :")
    st.write(b)

    st.write("------------------------------------------------------")

    st.subheader("Prédiction")
    
    scaler1 = MinMaxScaler()
    b_scaled = scaler1.fit_transform(b)

    #st.write(b_scaled)


    Modeles = ['LogisticRegression', 'RandomForestClassifier', 'KNeighborsClassifier', "Artificial Neuronal Network"]

    Modele1=st.selectbox('Selectionner le modèle', Modeles)

    if Modele1 == "LogisticRegression":

        model_lr1 = LogisticRegression()

        clicked19 = st.button('Prédire')
    
        clicked20 = st.button('Effacer')
    
        if clicked19 :

            model_lr1.fit(X1_train_scaled, y1_train)
            y_pred_lr1 = model_lr1.predict(X1_test_scaled)

            predict_lr = model_lr1.predict(b_scaled)
            
            if predict_lr == 0 :
                             
                           
                st.subheader('*** *Le candidat va refuser de se faire dépister au VIH/SIDA***')
                
                                
            elif predict_lr == 1 :

                st.subheader('*** *Le candidat va accepter de se faire dépister au VIH/SIDA***')
            
            precision_lr = accuracy_score(y1_test, y_pred_lr1)*100

            st.write("***Avec une précision de :***", precision_lr, "%")


    elif Modele1 == "RandomForestClassifier":

        model_rf1 = RandomForestClassifier()

        clicked21 = st.button('Prédire')
    
        clicked22 = st.button('Effacer')
    
        if clicked21 :

            model_rf1.fit(X1_train_scaled, y1_train)
            y_pred_rf1 = model_rf1.predict(X1_test_scaled)

            predict_rf = model_rf1.predict(b_scaled)

            if predict_rf == 0 :
                
                st.subheader('*** *Le candidat va refuser de se faire dépister au VIH/SIDA***')
            
            elif predict_rf == 1 :

                st.subheader('*** *Le candidat va accepter de se faire dépister au VIH/SIDA***')
            
            precision_rf = accuracy_score(y1_test, y_pred_rf1)*100

            st.write("***Avec une précision de :***", precision_rf, "%")

    elif Modele1 == "KNeighborsClassifier":

        model_kn1 = KNeighborsClassifier()

        clicked23 = st.button('Prédire')
    
        clicked24 = st.button('Effacer')
    
        if clicked23 :

            model_kn1.fit(X1_train_scaled, y1_train)
            y_pred_kn1 = model_kn1.predict(X1_test_scaled)

            predict_kn = model_kn1.predict(b_scaled)

            if predict_kn == 0 :
                
                st.subheader('*** *Le candidat va refuser de se faire dépister au VIH/SIDA***')
            
            elif predict_kn == 1 :

                st.subheader('*** *Le candidat va accepter de se faire dépister au VIH/SIDA***')
            
            precision_kn = accuracy_score(y1_test, y_pred_kn1)*100

            st.write("***Avec une précision de :***", precision_kn, "%")

    if Modele1 == "Artificial Neuronal Network":

        model_ann1 = Sequential()

        model_ann1 = Sequential()

        model_ann1.add(keras.layers.Flatten(input_shape=(1,20)))
        model_ann1.add(keras.layers.Dense(4, activation="relu"))
        model_ann1.add(keras.layers.Dense(8, activation="relu"))
        model_ann1.add(keras.layers.Dense(4, activation="relu"))
        model_ann1.add(keras.layers.Dense(1))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model_ann1.compile(loss="binary_crossentropy", optimizer=optimizer , metrics=["accuracy"])

        X1_train_scaled_ann = X1_train_scaled.reshape(-1, 1, 20)

        X1_test_scaled_ann = X1_test_scaled.reshape(-1, 1, 20)


        clicked25 = st.button('Prédire')
    
        clicked26 = st.button('Effacer')
    
        if clicked25 :

            history = model_ann1.fit(X1_train_scaled_ann, y1_train, epochs=20, batch_size = 100 )

            acc1 = model_ann1.evaluate(X1_test_scaled_ann, y1_test)

            print('accuracy :', acc1)
            
            predictions1 = model_ann1.predict(X1_test_scaled_ann)
            prediction_labels1 =[1 if prob1 > 0.5 else 0 for prob1 in np.ravel(predictions1)]
            

            b_scaled_ann = b_scaled.reshape(-1, 1, 20)

            predict_ann1 = model_ann1.predict(b_scaled_ann)
            prediction_labels_ann1 =[1 if prob_ann1 > 0.5 else 0 for prob_ann1 in np.ravel(predict_ann1)]

            #st.write("prediction :", prediction_labels_ann1[0])

            if prediction_labels_ann1[0] == 0 :
                
                st.subheader('*** *Le candidat va refuser de se faire dépister au VIH/SIDA***')
            
            elif prediction_labels_ann1[0] == 1 :

                st.subheader('*** *Le candidat va accepter de se faire dépister au VIH/SIDA***')
            
            precision_ann = accuracy_score(y1_test, prediction_labels1)*100

            st.write("***Avec une précision de :***", precision_ann, "%")

            st.write("-------------------------------------------------------------")

            st.write('Courbes loss-accuracy')
            fig10, ax10 = plt.subplots()
            
            ax10 = plt.plot(range(1,21), history.history['loss'], label='loss', color='red')                   
            ax10 = plt.plot(range(1,21), history.history['accuracy'], label='acc', color='blue')
            ax10 = plt.xlabel("epochs")
            ax10 = plt.legend()
            st.pyplot(fig10, width=200)
            



    #scaler = MinMaxScaler()
    #a_scaled = scaler.fit_transform(a)

    #a_pred = model_rf.predict(a_scaled)

    #st.write(a_pred)


