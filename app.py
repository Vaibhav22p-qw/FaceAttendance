from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)

# Load known face encodings and IDs
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds

# Initialize webcam capture
cap = cv2.VideoCapture(0)

def detect_faces(frame):
    # Resize frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert BGR to RGB (OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Initialize variables for face detection results
    face_ids = []
    
    for face_encoding in face_encodings:
        # Compare face encoding with known encodings
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        
        # Find the best match
        match_index = np.argmin(face_distances)
        
        if matches[match_index]:
            face_ids.append(studentIds[match_index])
    
    return face_locations, face_ids

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect faces in the frame
        face_locations, face_ids = detect_faces(frame)
        
        # Draw rectangles around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
        
        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if not ret:
            break
        
        # Convert buffer to bytes
        frame_bytes = buffer.tobytes()
        
        # Yield frame in bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
