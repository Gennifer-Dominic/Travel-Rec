from flask import Flask, render_template,request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np


#app
app=Flask(__name__)

#load Models
features=['UserId','CityName','Attraction','VisitMode','AttractionType']
RegClassfeatures=['UserId','CityName','Rating','Attraction','AttractionType']
model=pickle.load(open('rf_regressor.pkl','rb'))
rc=pickle.load(open('recomClassifier.pkl','rb'))
label_encoder=pickle.load(open('label_encoder.pkl','rb'))
#rclabel_encoder=pickle.load(open('rclabel_encoder.pkl','rb'))

#Load datasets
user_df=pd.read_excel("Tourism dataset\\User.xlsx")
city_df=pd.read_excel("Tourism dataset\\City.xlsx")
continent_df=pd.read_excel("Tourism dataset\\Continent.xlsx")
country_df=pd.read_excel("Tourism dataset\\Country.xlsx")
item_df=pd.read_excel("Tourism dataset\\Item.xlsx")
mode_df=pd.read_excel("Tourism dataset\\Mode.xlsx")
region_df=pd.read_excel("Tourism dataset\\Region.xlsx")
transaction_df=pd.read_excel("Tourism dataset\\Transaction.xlsx")
type_df=pd.read_excel("Tourism dataset\\Type.xlsx")

#EDA
citydetail_df=pd.merge(user_df,city_df,on='CityId',how='inner')
citydetail_df.drop("CountryId_y",axis=1,inplace=True)
citydetail_df.rename(columns={'ContenentId':'ContentId','CountryId_x':'CountryId'}, inplace=True)
countrydetail_df=pd.merge(citydetail_df,country_df,on='CountryId',how='inner')
countrydetail_df.drop("RegionId_y",axis=1,inplace=True)
countrydetail_df.rename(columns={'RegionId_x':'RegionId'}, inplace=True)
countrydetail_df=pd.merge(countrydetail_df,region_df,on='RegionId',how='inner')
countrydetail_df.drop("ContentId_y",axis=1,inplace=True)
countrydetail_df.rename(columns={'ContentId_x':'ContentId'}, inplace=True)
continent_df.rename(columns={'ContenentId':'ContentId'}, inplace=True)
userdetails_df=pd.merge(countrydetail_df,continent_df,on='ContentId',how='inner')
transactiondetails_df=pd.merge(transaction_df,item_df,on='AttractionId',how='inner')
transactiondetails_df.rename(columns={'VisitMode':'VisitModeId'}, inplace=True)
transactiondetails_df=pd.merge(transactiondetails_df,mode_df,on='VisitModeId',how='inner')
transactiondetails_df=pd.merge(transactiondetails_df,type_df,on='AttractionTypeId',how='inner')

user_recom=transactiondetails_df
user_recom=user_recom.sample(10000)

user_item_matrix=user_recom.pivot_table('Rating',['UserId'],'AttractionId')
user_item_matrix.fillna(0,inplace=True)
user_mat_sim=cosine_similarity(user_item_matrix)

df=pd.merge(userdetails_df,transactiondetails_df,on='UserId',how='inner')
df=df.sample(10000)
df.drop("ContentId",axis=1,inplace=True)
df.drop("RegionId",axis=1,inplace=True)
df.drop("CountryId",axis=1,inplace=True)
df.drop("CityId",axis=1,inplace=True)
df.drop("Country",axis=1,inplace=True)
df.drop("Region",axis=1,inplace=True)
df.drop("Contenent",axis=1,inplace=True)
df.drop("TransactionId",axis=1,inplace=True)
df.drop("VisitYear",axis=1,inplace=True)
df.drop("VisitMonth",axis=1,inplace=True)
df.drop("VisitModeId",axis=1,inplace=True)
df.drop("AttractionId",axis=1,inplace=True)
df.drop("AttractionAddress",axis=1,inplace=True)
df.drop("AttractionCityId",axis=1,inplace=True)
df.drop("AttractionTypeId",axis=1,inplace=True)

col=['UserId', 'CityName', 'Attraction', 'VisitMode','AttractionType', 'Rating']
df=df[col]

col1=['UserId', 'CityName','Rating','Attraction','AttractionType', 'VisitMode']
data=df[col1]



#Collaborative Fillter
def collabfill(user_id,user_mat_sim,user_item_matrix,user_recom):
    #find user similarity
    simuser=user_mat_sim[user_id-1]
    #Get similar users
    sim_user_ids=np.argsort(simuser)[::-1][1:6]
    # destination liked by sim users
    sim_user_rating=user_item_matrix.iloc[sim_user_ids].mean(axis=0)
    # recommend top 5 dest
    rec_dest_id=sim_user_rating.sort_values(ascending=False).head(5).index
    rec=user_recom[user_recom['AttractionId'].isin(rec_dest_id)][['Attraction','VisitMode','Rating']].drop_duplicates().head(5)

    return rec


#Recommendation function using regression model
def recom_dest(user_input,model,label_encoder,features,data):
    #encode use i/p
    encodeddata={}
    for i in features:
        if i in label_encoder:
            encodeddata[i]=label_encoder[i].transform([user_input[i]])[0]
        else:
            encodeddata[i]=user_input[i]

    input_df=pd.DataFrame([encodeddata])
    pred_rate=model.predict(input_df)[0]
    return int(pred_rate)

#Recommendation function using Classifier model
def recomClassifier_dest(user_input, model, label_encoder, features, data):

    encodeddata = {}
    for i in features:
        if i in label_encoder:
            encodeddata[i] = label_encoder[i].transform([user_input[i]])[0]
        else:
            encodeddata[i] = user_input[i]

    input_df = pd.DataFrame([encodeddata])
    pred_rate = model.predict(input_df)[0]
    return pred_rate

#route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        user_input = {'UserId': int(request.form['cuserid']),
                      'CityName': request.form['ccity'],
                      'Rating': int(request.form['cmode']),
                      'Attraction': request.form['cattraction'],
                      'AttractionType': request.form['cattractiontype']
                      }

        pred_visitmode = recomClassifier_dest(user_input, rc, label_encoder, RegClassfeatures, data)

        return render_template('classification.html', pred_visitmode=pred_visitmode)


@app.route('/recommend', methods=['POST','GET'])
def recommend():
    if request.method == 'POST':
        user_id= int(request.form['userid'])
        user_input = {'UserId': int(request.form['userid']),
                      'CityName': request.form['city'],
                      'Attraction': request.form['attraction'],
                      'VisitMode': request.form['mode'],
                      'AttractionType': request.form['attractiontype']
                      }

        recom_data = collabfill(user_id, user_mat_sim, user_item_matrix, user_recom)
        rec=recom_data.to_dict()
        pred_rating = recom_dest(user_input, model, label_encoder, features, df)

        return render_template('index.html', recom_data=rec,pred_rating=pred_rating,tables=[recom_data.to_html()], titles=[''])

#python main
if __name__=="__main__":
    app.run(debug=True)
