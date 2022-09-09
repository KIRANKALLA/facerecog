import streamlit as st
import pickle 
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2


pickle_in=open('/content/drive/MyDrive/secondyearencodings.pkl','rb')
dd=pickle.load(pickle_in)
pickle_in1=open('/content/drive/MyDrive/secondyearnames.pkl','rb')
ee=pickle.load(pickle_in1)
unknown_image=None
st.header('RCEE AI&DS THIRD YEAR FACE RECOGNITION')
unknown_image=st.file_uploader('Take a Picture')
if unknown_image is not None:
    unknown_image=face_recognition.load_image_file(unknown_image)
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)
name=''
# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(dd, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(dd, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = ee[best_match_index]
        

    # Draw a box around the face using the Pillow module
    #pil_image=cv2.rectangle(unknown_image,pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255),thickness=2)
    pil_image=cv2.rectangle(unknown_image,pt1=(left, top),pt2=(right, bottom),color=(255,0,0),thickness=2)

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    cv2.putText(pil_image, text=name, org=(left + 6, bottom - text_height - 5), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)

    #cv2.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    #cv2.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    st.success(name)


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
#st.image(pil_image,caption='PREDICTED')
st.image(pil_image,output_format='PNG')
st.success(name)
