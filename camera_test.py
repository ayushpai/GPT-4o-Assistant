import cv2


while True:
        cap = cv2.VideoCapture(1)
        # display camera feed
        ret, frame = cap.read()

        #flip camera feed 180
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)


        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()

