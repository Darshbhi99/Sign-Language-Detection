import cv2, os
# import matplotlib.pyplot as plt
import pickle as pkl
import mediapipe as mp

# Initializing the hands class
mp_hands = mp.solutions.hands
# Initializing the points on the hands
mp_drawing = mp.solutions.drawing_utils
# Initializing the connection and style of points on the hand
mp_drawing_styles = mp.solutions.drawing_styles

# Object for detecting hands in the image
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data = []
labels = []
# Getting the list of labels dir in the images dir
for dir in os.listdir('images'):
    # Taking image from each directory
    for img in os.listdir(os.path.join('images',dir)):
        # Default color after reading in cv2 will be BGR
        img = cv2.imread(os.path.join('images', dir, img))
        # Converting it into RGB to further preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detecting the hands in the image
        results = hands.process(img_rgb)
        # Creating list for storing the x and y co-ordinates of the landmarks
        data_aux = []
        # If Hands Detected in the Image
        if results.multi_hand_landmarks:
            # Getting each point from the detected hand (Around 7 points in single hand)
            for hand_landmarks in results.multi_hand_landmarks:
                # Drawing the connections and style on the image 
                # mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                #                             mp_drawing_styles.get_default_hand_landmarks_style(),
                #                             mp_drawing_styles.get_default_hand_connections_style())
                
                # Getting the each landmark point and Saving the x and y co-ordinates of it 
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x)
                    data_aux.append(hand_landmarks.landmark[i].y)
            data.append(data_aux)
            labels.append(dir)
                
        # plt.figure()
        # plt.imshow(img_rgb)

# plt.show()
# Saving the preproccessed data
with open('data.pkl', 'wb') as f:
    pkl.dump({'data': data, 'label': labels}, f)
f.close()