import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model('nnh.h5')

run = False
ix, iy = -1, -1
follow = 25
img = np.zeros((512, 512, 3), np.uint8)  # Ensure a 3-channel image (BGR)

# Function for drawing and prediction
def draw(event, x, y, flags, params):
    global run, ix, iy, img, follow

    if event == cv2.EVENT_LBUTTONDOWN:
        run = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if run:
            cv2.circle(img, (x, y), 20, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        run = False
        cv2.circle(img, (x, y), 20, (255, 255, 255), -1)
        
        # Preprocess the drawn image to match the input shape expected by the model
        resized_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray = np.expand_dims(gray, axis=-1)  # Add channel dimension
        gray = np.expand_dims(gray, axis=0)   # Add batch dimension

        result = np.argmax(model.predict(gray))
        result_text = 'CNN Prediction: {}'.format(result)
        cv2.putText(img, org=(25, follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text=result_text, color=(255, 0, 0), thickness=1)
        follow += 25

    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512, 512, 3), np.uint8)  # Reset image
        follow = 25

# Parameters and GUI window
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while True:
    cv2.imshow("image", img)
   
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

