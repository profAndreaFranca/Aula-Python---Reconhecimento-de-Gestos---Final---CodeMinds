import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

tipIds = [4, 8, 12, 16, 20]

def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        landmarks = hand_landmarks[handNo].landmark
        fingers = []

        for lm_index in tipIds:
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index - 2].y

            thumb_tip_x = landmarks[lm_index].x
            thumb_bottom_x = landmarks[lm_index - 2].x

            if lm_index != 4:
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if thumb_tip_x > thumb_bottom_x:
                    fingers.append(1)
                else:
                    fingers.append(0)

        totalFingers = fingers.count(1)

        cv2.putText(image, f'Dedos: {totalFingers}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def drawHandLandmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_hands.HAND_CONNECTIONS
            )

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    results = hands.process(image)
    hand_landmarks = results.multi_hand_landmarks

    drawHandLandmarks(image, hand_landmarks)
    countFingers(image, hand_landmarks)

    cv2.putText(image, "Contador de Dedos IA", (30, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, "Espaco = sair", (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Contador de Dedos IA", image)

    key = cv2.waitKey(1)
    if key == 32:
        break

cap.release()
cv2.destroyAllWindows()
