import pandas as pd
import numpy as np
import nltk, re, pprint
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Transaction:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(self.file_name)
        self.data = self.data.drop(['sub_category', 'label', 'account', 'user', 'status', 'id', 'category',
                                    'trans_location', 'updated', 'created', 'date_of_trans', 'trans_note',
                                    'trans_number', 'service', 'is_subscription', 'channel', 'mode_of_payment'],
                                   axis=1
                                   )
        self.data = self.data.set_index('kind')
        self.data = self.data.drop('manual', axis=0)
        self.data = self.data.drop_duplicates()
        self.data['desc_trans_type'] = self.data['trans_type'] + ' ' + self.data['trans_desc']
        # self.data = self.data.rename(columns = {'categories': 'Category', 'trans_type': 'Transaction_Type'})
        self.others_list = ['Airtime/Data', 'Withdrawal',
                            'Transfer Payments', 'Bill Payments',
                            'Commission', 'Gifts/Grants',
                            'Salary/Wages', 'Transfers',
                            ]

    def replace_categories(self, text):
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

    def others(self, x):
        if x in self.others_list:
            return x
        else:
            return 'others'

    def clean_data(self, text):
        new_text = re.sub(r'\b(\w+)(\1\b)+', r'\1', str(text))
        new_text = re.sub(r'[^\w\s]', '', str(new_text))
        new_text = re.sub('\s+', ' ', str(new_text))
        new_text = re.sub('\d+', '', str(new_text))
        return new_text.strip().lower()

    def predict_data(self, ):
        self.data['categories_n'] = self.data['desc_trans_type'].apply(self.replace_categories)
        self.data['categories'] = self.data['categories_n'].apply(self.others)
        self.data = self.data.drop('trans_amount', axis=1)
        self.data = self.data.set_index('categories')
        self.data = self.data.drop('others', axis=0)
        self.data = self.data.reset_index()
        self.data['desc_remarks'] = self.data['trans_desc'] + ' ' + self.data['trans_remarks']
        self.data = self.data.rename(columns={'categories': 'Category', 'trans_type': 'Transaction_Type'})
        self.data.drop(['categories_n', 'desc_trans_type', 'trans_desc'], axis=1, inplace=True)
        self.y = self.data['Category']
        encodeTarget = LabelEncoder()
        Y = encodeTarget.fit_transform(self.y)
        self.x = self.data[["Transaction_Type", "desc_remarks"]]

        self.x["Transaction_Type"] = self.x["Transaction_Type"].apply(self.clean_data)
        self.x["desc_remarks"] = self.x["desc_remarks"].apply(self.clean_data)
        descRemarks = self.x['desc_remarks']
        countVecDescRemarks = CountVectorizer(ngram_range=(1, 1), stop_words='english')
        descRemarksEncd = countVecDescRemarks.fit_transform(descRemarks)
        transactionType = self.data['Transaction_Type'].to_numpy().reshape(-1, 1)
        OneHotEnctransType = OneHotEncoder()
        OneHotEnctransType.fit(transactionType)
        transactionTypeEncd = OneHotEnctransType.transform(transactionType)
        transactionTypeEncdArr = transactionTypeEncd.toarray()
        descRemarksEncdArr = descRemarksEncd.toarray()
        X = np.hstack((descRemarksEncdArr, transactionTypeEncdArr))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(Y_test, y_pred)
        pred_class = encodeTarget.classes_[y_pred]
        return [acc, pred_class]