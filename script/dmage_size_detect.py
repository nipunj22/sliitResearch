import cv2
import joblib
from rembg import remove
from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt

def check_contrast(a,b,x,image):
  matrix = [[0 for _ in range(256)] for _ in range(256)]
  total = 0
  boundry1 = a+x
  boundry2 = b+x
  if boundry1 == 200:
    boundry1-=1
  if boundry2 == 200:
    boundry2-=1
  for i in range(a , boundry1):
    for j in range(b , boundry2):
      matrix[image[i][j]][image[i][j+1]]+=1
      total+=1
  contrast = 0
  for i in range(0,256):
    for j in range(0 , 256):
      contrast+=matrix[i][j]/total * pow(i-j, 2)
  print(contrast)
  return contrast , a , b

def get_lbp_feature(gray_image):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    # print(len(lbp.ravel()))
    return lbp

def convert_image(file_path):
    try:
        image = cv2.imread(file_path)
        image = remove(image)
        image = cv2.resize(image, (200, 200))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {
            'state':True,
            'image':gray_image
        }
    except:
        return {
            'state': False
        }

def get_damage_size():
    image_path = "uploaded.png"
    y = []
    radius = 1
    n_points = 8 * radius
    image_no = "783"
    file_path = image_path
    gray_image = convert_image(file_path=file_path)['image']
    lbp = get_lbp_feature(gray_image=gray_image)
    h = np.array([[]])
    h = np.append(h, lbp.ravel())
    file_path = image_path
    gray_image = convert_image(file_path=file_path)['image']
    lbp = get_lbp_feature(gray_image=gray_image)
    h3 = np.array([[]])
    h3 = np.append(h3, lbp.ravel())
    num_rows = 1
    num_columns = len(h) // num_rows
    two_dimensional_array = h.reshape((num_rows, num_columns))
    print(two_dimensional_array)
    print(two_dimensional_array)
    rf_regressor = joblib.load('damage_size_prediction.h5')
    y_pred = rf_regressor.predict(two_dimensional_array)
    print(y_pred)
    image_state = convert_image(image_path)
    image = image_state['image']
    image_ = image
    print(image[125][140])
    print(image[125][75])
    print(image[100][100])
    print(len(image))
    rough_array = []
    box_size = 25
    for i in range(0, 200, box_size):
        for j in range(0, 200, box_size):
            contrast, a, b = check_contrast(i, j, box_size,image)
            if contrast > 1500:
                object_ = {
                    'a': a,
                    'b': b
                }
                rough_array.append(object_)
    print(rough_array)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))
    overlay = image.copy()
    alpha = 0.4  # Set the transparency level (0.0 - fully transparent, 1.0 - fully opaque)
    margin = 20
    # Define the coordinates of the box
    for i in rough_array:
        print(i)
        a = i['a'] - margin
        b = i['b'] - margin
        c = i['a'] + box_size + margin
        d = i['b'] + box_size + margin
        x, y, w, h = 100, 100, 200, 150  # Change these values according to your requirements
        cv2.rectangle(overlay, (a, b), (c, d), (255, 0, 0), thickness=cv2.FILLED)

    # Combine the image and the overlay with transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Display the result
    plt.imshow(image)

    overlay = image.copy()
    alpha = 0.4  # Set the transparency level (0.0 - fully transparent, 1.0 - fully opaque)
    number_text = str("{:.1f}".format(y_pred[0] * 100)) + "%"  # Use (i + 1) to start numbering from 1
    text_position = ((a + 5), (b + d) // 2)  # Center of the rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    cv2.putText(overlay, number_text, text_position, font, font_scale, (0, 0, 255), font_thickness)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    plt.savefig('image_highlighted.png')
    return y_pred[0]
    #plt.imshow(image)
    #plt.show()

#get_damage_size()
#print("damage size: ", get_damage_size())