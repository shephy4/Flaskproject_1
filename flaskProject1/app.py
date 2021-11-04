import pandas as pd
import numpy as np
import nltk, re, pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from flask_bootstrap import Bootstrap
from sklearn.model_selection import cross_val_score
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from flask import Flask,render_template,url_for,request
from flask.views import MethodView
import joblib
from joblib import dump
app = Flask(__name__)
bootstrap = Bootstrap(app)
@app.route('/')
def home():
	return render_template('result.html')
@app.route('/predict', methods=['POST'])
def predict():
    file_name = r'C:\Users\DELL\Downloads\transactions_unclean.csv'
    data = pd.read_csv(file_name)
    data = data.drop(['sub_category', 'label', 'account', 'user', 'status', 'id', 'category',
                        'trans_location', 'updated', 'created', 'date_of_trans', 'trans_note',
                        'trans_number', 'service', 'is_subscription', 'channel', 'mode_of_payment'],
                       axis=1
                       )
    data = data.set_index('kind')
    data = data.drop('manual', axis=0)
    data = data.drop_duplicates()
    data['desc_trans_type'] = data['trans_type'] + ' ' + data['trans_desc']
    others_list = ['Airtime/Data', 'Withdrawal',
                'Transfer Payments', 'Bill Payments',
                'Commission', 'Gifts/Grants',
                'Salary/Wages', 'Transfers',
                ]

    def replace_categories(text):
        text_new = (str(text)).lower()
        text_new = (str(text_new)).split('\n')
        text_new = re.sub('^.*(credit.transfer).*$', 'Transfers', str(text_new))
        text_new = re.sub('^.*(debit.transfer|trf|from|to|mbanking).*$', 'Transfer Payments', str(text_new))
        text_new = re.sub('^.*(commission|tax|charge).*$', 'Commission', str(text_new))
        text_new = re.sub('^.*(wdl|withdrawal|cash|csh|wd).*$', 'Withdrawal', str(text_new))
        text_new = re.sub(
        '^.*(nibss|payment|bills|pos|bet|order|restaurant|cinemas|supermarket|chicken|food|tfare|meat|cloth|birthday).*$',
        'Bill Payments', str(text_new))
        text_new = re.sub('^.*(airtime|data|mtn|glo|airtel|9mobile).*$', 'Airtime/Data', str(text_new))
        text_new = re.sub('^.*(allowance|bonus).*$', 'Gifts/Grants', str(text_new))
        text_new = re.sub('^.*salary.*$', 'Salary/Wages', str(text_new))
        return text_new

    def others(x):
        if x in others_list:
            return x
        else:
            return 'others'

    def clean_data(text):
        new_text = re.sub(r'\b(\w+)(\1\b)+', r'\1', str(text))
        new_text = re.sub(r'[^\w\s]', '', str(new_text))
        new_text = re.sub('\s+', ' ', str(new_text))
        new_text = re.sub('\d+', '', str(new_text))
        return new_text.strip().lower()

    data['categories_n'] = data['desc_trans_type'].apply(replace_categories)
    data['categories'] = data['categories_n'].apply(others)
    data = data.drop('trans_amount', axis=1)
    data = data.set_index('categories')
    data = data.drop('others', axis=0)
    data = data.reset_index()
    data['desc_remarks'] = data['trans_desc'] + ' ' + data['trans_remarks']
    data = data.rename(columns={'categories': 'Category', 'trans_type': 'Transaction_Type'})
    data.drop(['categories_n', 'desc_trans_type', 'trans_desc'], axis=1, inplace=True)
    y = data['Category']
    encodeTarget = LabelEncoder()
    Y = encodeTarget.fit_transform(y)
    x = data[["Transaction_Type", "desc_remarks"]]
    x["Transaction_Type"] = x["Transaction_Type"].apply(clean_data)
    x["desc_remarks"] = x["desc_remarks"].apply(clean_data)
    descRemarks = x['desc_remarks']
    countVecDescRemarks = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    descRemarksEncd = countVecDescRemarks.fit_transform(descRemarks)
    transactionType = data['Transaction_Type'].to_numpy().reshape(-1, 1)
    OneHotEnctransType = OneHotEncoder(handle_unknown = 'ignore').fit(transactionType)
    transactionTypeEncd = OneHotEnctransType.transform(transactionType)
    transactionTypeEncdArr = transactionTypeEncd.toarray()
    descRemarksEncdArr = descRemarksEncd.toarray()
    X = np.hstack((descRemarksEncdArr, transactionTypeEncdArr))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    #y_pred = model.predict(X_test)
    #acc = accuracy_score(Y_test, y_pred)
    #pred_class = encodeTarget.classes_[y_pred]
    #joblib.dump(model, 'transaction_model.pkl')
    #transaction_model = open('transaction_model.pkl','rb')
    #clf = joblib.load(transaction_model)
    if request.method == 'POST':
        message = request.form['message']
        raw_desc = [message]
        raw_desc = pd.DataFrame(data=raw_desc)
        messages = request.form['messages']
        raw_trans = [messages]
        raw_trans = pd.DataFrame(data=raw_trans)
        #n = np.hstack((raw_desc, raw_trans))
        #n = pd.DataFrame(data=n)
        #n.columns = ['desc_remarks', 'Transaction_Type']
        raw_desc = raw_desc.apply(clean_data)
        descRemarksEncd = countVecDescRemarks.transform(raw_desc)
        transactionType = raw_trans.to_numpy().reshape(-1, 1)
        transactionTypeEncd = OneHotEnctransType.transform(transactionType)
        transactionTypeEncdArr = transactionTypeEncd.toarray()
        descRemarksEncdArr = descRemarksEncd.toarray()
        xTest = np.hstack((descRemarksEncdArr, transactionTypeEncdArr))
        pred = model.predict(xTest)
        pred_class = encodeTarget.classes_[pred]
    return render_template('result.html', prediction = pred_class)
if __name__ == "__main__":
    app.run(debug=True)
