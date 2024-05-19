import joblib
import numpy as np

from dmage_size_detect import get_damage_size
from repository import features , input

def predict_damage_cost(datapoint):
    load_model = joblib.load('damage_cost_prediction.h5')
    predictions = load_model.predict(datapoint)
    print(predictions)
    return predictions[0]

def detect_cost(body):
    datapoint = []
    keys = input.keys()
    keys_list = list(keys)
    print(keys_list)
    keys_feature = features.keys()
    keys_list_feature = list(keys_feature)
    print(keys_list_feature)
    for key in keys_list:
        print(key)
        if key == "damage_size":
            try:
                damage_size = get_damage_size()
                datapoint.append(damage_size)
            except:
                datapoint.append(0)
        else:
            try:
                if not key in keys_list_feature:
                    print(body[key])
                    datapoint.append(float(body[key]))
                else:
                    print(features[key][body[key]])
                    datapoint.append(features[key][body[key]])
            except:
                try:
                    print(features[key]['nan'])
                    datapoint.append(features[key]['nan'])
                except:
                    return False
    print(datapoint)
    prediction = predict_damage_cost([datapoint])
    return prediction

# datapoint = [[0,10.0,0,700000.0,0.76]]
# numpy_data = np.array(datapoint)
# np.set_printoptions(suppress=True)
# print(numpy_data)
# predict_damage_cost(numpy_data)
#body = {
#    "vehicle_type" : "Toyota_AQUA",
#   "distance_to_location":10,
#    "vehicle_damage_part" : "Front Bumper",
#    "standard_current_price_of_the_part_of_vehicle":13000,
#     "damage_size":0.60,
#}
#print("cost of total: ", detect_cost(body))