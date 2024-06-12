import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 32
dataset_size = 100

cap = cv2.VideoCapture(0)


# Function to calculate the center position for the text
def get_centered_text_position(text, font, scale, thickness, frame_width, line_number, total_lines):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = 50 + (line_number - 1) * (text_size[1] + 10) + ((total_lines - 1) * (text_size[1] + 10) // 2)
    return text_x, text_y


for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Display initial text until 's' is pressed
    while True:
        ret, frame = cap.read()
        frame_width = frame.shape[1]
        text = 'Ready? Press "s" to start!'
        text_x, text_y = get_centered_text_position(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3, frame_width, 1, 2)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
            print('Captured image {}/{} for class {}'.format(counter + 1, dataset_size, j))
            counter += 1

    # Display text again after data for the current class is collected
    while True:
        ret, frame = cap.read()
        frame_width = frame.shape[1]
        text1 = 'Data for class {} collected.'.format(j + 1)
        text2 = 'Ready? Press "s" to start next class.'

        text1_x, text1_y = get_centered_text_position(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, frame_width, 1, 2)
        text2_x, text2_y = get_centered_text_position(text2, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, frame_width, 2, 2)

        cv2.putText(frame, text1, (text1_x, text1_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text2, (text2_x, text2_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break

cap.release()
cv2.destroyAllWindows()
