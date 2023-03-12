import cv2 ,os
from time import sleep

# url = "https://192.168.0.103:8080/video"
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)


no_of_classes = 3
for i in range(no_of_classes):
    if not os.path.exists('images'):
        os.mkdir('images')
    
    if not os.path.exists(f'images/{i}'):
        os.mkdir(f'images/{i}')
    
    while True:
        success, frame = cap.read()
        if success:
            cv2.putText(frame, "Ready!!! Press Q !!", (100 ,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25)==ord('q'):
                break
    
    print(f"Collecting images for Class {i}")
    
    counter = 0
    while counter<100:
        success, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(f'images/{i}/{counter}.jpg', frame)
        counter+=1


cap.release()
cv2.destroyAllWindows()