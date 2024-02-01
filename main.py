import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Specify the path to the image you want to overlay on the face
overlay_image_path = 'icon.png'

# Try loading the overlay image
try:
    overlay_image = cv2.imread(overlay_image_path, -1)
    if overlay_image is None:
        raise Exception(f"Error loading image: {overlay_image_path}")
except Exception as e:
    print(f"An exception occurred: {e}")
    exit()

# Check the loaded image
if overlay_image is None:
    print(f"Error: Unable to load the image from path: {overlay_image_path}")
    exit()

# Print information about the loaded image
print(f"Overlay Image Shape: {overlay_image.shape}")
print(f"Overlay Image Channels: {overlay_image.shape[2] if len(overlay_image.shape) == 3 else 1}")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterate through the detected faces
    for (x, y, w, h) in faces:
        try:
            # Resize the overlay image to fit the face
            resized_overlay = cv2.resize(overlay_image, (w, h))
        except Exception as resize_error:
            print(f"Error resizing image: {resize_error}")
            continue

        # Extract the alpha channel from the overlay image
        alpha_channel = resized_overlay[:, :, 3] / 255.0

        # Blend the face region with the overlay image
        for c in range(0, 3):
            frame[y:y + h, x:x + w, c] = (1 - alpha_channel) * frame[y:y + h, x:x + w, c] + alpha_channel * resized_overlay[:, :, c]

    # Display the resulting frame
    cv2.imshow('Face Overlay', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
