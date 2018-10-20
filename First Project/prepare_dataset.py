import os

import codecs
import numpy as np
import json

data_features = np.array([])
data_label = np.array([])

file = codecs.open("all_data 3.json", "r", "utf-8")
json_str = file.read()
json_data = json.loads(json_str)

features = ["education", "job", "marital_status",
            "birt_year", "gender", "user_city"]
usersIDdic = {}


def create_feature(data_features, data_label, file):
    each_row = []
    label = []
    each_row.append(str(file['viewer_education']))
    each_row.append(str(file['viewer_job']))
    each_row.append(str(file['viewer_marital_status']))
    each_row.append(str(file['viewer_birt_year']))
    each_row.append(str(file['viewer_gender']))
    each_row.append(str(file['viewer_user_city']))
    if (file['event_category'] is None) or (file['event_category'] == ''):
        label.append(None)
    else:
        label.append(file['event_category'])

    each_row = np.array(each_row)
    return each_row, label


# the clicks that has clicked more than 1 from a unique user

for id in json_data:
    if str(id['event_type']) == "CLICK":

        if id['viewer_user_id'] not in usersIDdic.keys():
            usersIDdic[id['viewer_user_id']] = []
            usersIDdic[id['viewer_user_id']].append(id['ad_id'])  # append the ad

            each_row, label = create_feature(data_features, data_label, id)
            data_features = np.append(data_features, each_row)
            data_label = np.append(data_label, label)

        elif id['ad_id'] not in usersIDdic[id['viewer_user_id']]:  # for unique user different ads
            usersIDdic[id['viewer_user_id']].append(id['ad_id'])  # append the ad

            each_row, label = create_feature(data_features, data_label, id)
            data_features = np.append(data_features, each_row)
            data_label = np.append(data_label, label)

if not os.path.exists("Dataset"):
    os.makedirs("Dataset")
np.save("Dataset//features.npy", data_features)
np.save("Dataset//labels.npy", data_label)