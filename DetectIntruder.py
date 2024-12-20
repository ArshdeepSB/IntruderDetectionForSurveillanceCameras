from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from win10toast import ToastNotifier
import numpy as np
import imutils
import time
import cv2

# ---------Initialize Variables---------
toaster = ToastNotifier()
notification_sent = False
intruder_start_time = None
vs = VideoStream(src=0).start()
time.sleep(2.0)
model = load_model("intruder_detector_resnet.keras")

# --------Run Model on Video Frames---------
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Predict whether there is an intruder
    resized_frame = cv2.resize(frame, (224, 224))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    preprocessed_frame = preprocess_input(img_to_array(rgb_frame))
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

    probability = model.predict(preprocessed_frame)
    print(probability)
    # Classify if intruder or permitted
    if probability >= 0.1:
        label = "Permitted"
        color = (0, 255, 0)
        intruder_start_time = None
        notification_sent = False
    else:
        label = "Intruder"
        color = (0, 0, 255)
        # Start timing if intruder detected
        if intruder_start_time is None:
            intruder_start_time = time.time()

        # Check if intruder is detected for 5 seconds
        elif (time.time() - intruder_start_time) >= 5:
                # Send Windows notification
                toaster.show_toast(
                    "Intruder Alert!",
                    "An intruder has been detected on your property.",
                    icon_path=None,
                    duration=10,
                    threaded=True, 
                )
                notification_sent = True
                
    # Display label
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Frame", frame)

    # Press Escape Key to exit or close window
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

# ---------Cleanup----------
cv2.destroyAllWindows()
vs.stop()