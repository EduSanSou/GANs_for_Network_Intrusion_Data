from sklearn.preprocessing import LabelEncoder
from category_encoders import OrdinalEncoder
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from models.classifiers import SVC
import sys, warnings
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings('ignore')

def get_data():
    data_folder = "../Data/UNSW-NB15"

    #drop id feature
    train = pd.read_csv(data_folder+"/UNSW_NB15_testing-set.csv",usecols=range(1,45))


    le = LabelEncoder()
    le.fit(train.attack_cat)
    label_mapping = {l: i for i, l in enumerate(le.classes_)}

    train['attack_cat'] = le.transform(train.attack_cat)

    return train, label_mapping

def subsample(x, y, size=25000):
    """Sample data from x """
    new_counts = np.ceil(np.bincount(y)/y.shape[0] * size)
    data = np.hstack((x, y.reshape((y.shape[0], 1))))
    class_wise_arrays = []

    for cls in range(10):
        class_wise_arrays.append(data[y == cls])

    class_wise_random_indices = []
    for array, count in zip(class_wise_arrays, new_counts):
        class_wise_random_indices.append(np.random.choice(array.shape[0], int(count), replace=False))

    data_small = np.vstack(tuple([array[index, :] for array, index in zip(class_wise_arrays, class_wise_random_indices)]))
    x_small = data_small[:, :-1]
    y_small = data_small[:, -1]

    return x_small, y_small.astype("int")

### === main code === ###

train, label_mapping = get_data()
print("Label mapping:\n", label_mapping)
x_train, y_train = train.drop(["label","attack_cat"],axis=1), train.attack_cat.values

categorical_features = ["proto","state","service"]
enc = OrdinalEncoder(cols=categorical_features)
x_train = enc.fit_transform(x_train,train.label)
train = None

data_cols = list(x_train.columns)

#filter out normal data points
att_ind = np.where(y_train != label_mapping["Normal"])[0]

#remove Normal netwok traffic from data
del label_mapping["Normal"]

x = x_train[data_cols].values[att_ind]
y = y_train[att_ind]

a = {'Analysis':6000, 'Backdoor':6000, 'DoS':3000, 'Exploits':0,'Fuzzers':2000,\
    'Generic':0, 'Reconnaissance':5000, 'Shellcode':2000, 'Worms':300}
val, count = np.unique(y,return_counts =True)
current_gt = dict(zip(val , count))
up_sampling_strategy = {label_mapping.get(j) : i+current_gt.get(label_mapping.get(j)) for j, i in a.items() if i > 0}

sm = SVMSMOTE(sampling_strategy = up_sampling_strategy,svm_estimator=SVC(C=10, cache_size=1500, class_weight='balanced'))
sm.fit(x,y)

rus = RandomUnderSampler(sampling_strategy = {label_mapping["Generic"]:20000,label_mapping["Exploits"]:20000},random_state=42)
x, y = rus.fit_resample(x,y)

sub_x, sub_y = subsample(x=x,y=y,size=25000)

new_trainx, new_y = sm.fit_resample(sub_x, sub_y)

new_trainx_dataframe = pd.DataFrame(new_trainx, columns = ['dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports'])
decoded_generated_samples = enc.inverse_transform(new_trainx_dataframe)

with open("original_samples.txt","w") as e:
    print(sub_x, file=e)

with open("svmsmote_generated_samples.txt","w") as f:
    print(new_trainx, file=f)

with open("svmsmote_generated_labels.txt","w") as g:
    print(new_y, file=g)

with open("svmsmote_decoded_generated_samples.txt","w") as h:
    print(decoded_generated_samples.to_string(), file=h)
