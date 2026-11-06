from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import sys, warnings
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings('ignore')

def get_data():
    data_folder = "../../Data/UNSW-NB15"

    #drop id feature
    train = pd.read_csv(data_folder+"/UNSW_NB15_testing-set.csv",usecols=range(1,45))

    le = LabelEncoder()
    le.fit(train.attack_cat)
    label_mapping = {l: i for i, l in enumerate(le.classes_)}

    train['attack_cat'] = le.transform(train.attack_cat)

    return train, label_mapping

### === main code === ###

#get train data and label mapping
train, label_mapping = get_data()
print("Label mapping:\n", label_mapping)
x_train, y_train = train.drop(["label","attack_cat"],axis=1), train.attack_cat.values

#filter out normal data points
#att_ind = np.where(y_train != label_mapping["Normal"])[0]
#x = x_train[data_cols].values[att_ind]
#y = y_train[att_ind]

#remove Normal netwok traffic from data
#del label_mapping["Normal"]

#assign features and labels
x = x_train.values
y = y_train

#create up sampling strategy
a = {'Analysis':6000, 'Backdoor':6000, 'DoS':3000, 'Exploits':0, 'Fuzzers':2000, 'Generic':0, 'Normal':0, 'Reconnaissance':5000, 'Shellcode':2000, 'Worms':300}
print("Number of samples to be added for each class:\n", a)
val, count = np.unique(y,return_counts =True)
current_gt = dict(zip(val , count))
print("Current number of samples for each class:\n", current_gt)
up_sampling_strategy = {label_mapping.get(j) : i+current_gt.get(label_mapping.get(j)) for j, i in a.items() if i > 0}
print("Up sampling strategy:\n", up_sampling_strategy)
print("Initial x shape:\n", x.shape)
print("Initial y shape:\n", y.shape)

#undersampling major classes
rus = RandomUnderSampler(sampling_strategy = {label_mapping["Generic"]:20000,label_mapping["Exploits"]:20000,label_mapping["Normal"]:20000},random_state=42)
x, y = rus.fit_resample(x,y)
print("x shape after undersampling:\n", x.shape)
print("y shape after undersampling:\n", y.shape)

#oversampling minor classes
smnc = SMOTENC(categorical_features = [1,2,3], sampling_strategy = up_sampling_strategy, random_state = None, k_neighbors = 5)
new_trainx, new_y = smnc.fit_resample(x, y)
print("x shape after oversampling:\n", new_trainx.shape)
print("y shape after oversampling:\n", new_y.shape)

#converting numpy array to pandas dataframe
new_trainx_dataframe = pd.DataFrame(new_trainx, columns = ['dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports'])
new_y_dataframe = pd.DataFrame(new_y, columns = ['attack_cat'])

#concatenating dataframes
new_concatenated_dataframe = pd.concat([new_trainx_dataframe, new_y_dataframe], axis=1)

#save data
with open("rus_generated_samples.txt","w") as e:
    print(x, file=e)

with open("smotenc_generated_samples.txt","w") as f:
    print(new_trainx_dataframe.to_string(), file=f)

with open("smotenc_generated_labels.txt","w") as g:
    print(new_y_dataframe.to_string(), file=g)

with open("smotenc_generated_samples_labels_concatenated.txt","w") as h:
    print(new_concatenated_dataframe.to_string(), file=h)
