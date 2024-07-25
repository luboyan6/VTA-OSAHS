import os
import csv
import cv2
import numpy as np
import pickle as pic
import mediapipe as mp
import torch
from utils.feature_selector import FeatureSelector
from utils.multimodal_fusion import data_fusion

def save_dataset(dataset, save_path):
    pic.dump(dataset, open(save_path, 'wb'))

def read_dataset(dataset_path):
    return pic.load(open(dataset_path, 'rb'))


def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def get_keypoints(key_points):
    column2 = list(set(list(np.array(key_points).T[1])))
    temp = insertion_sort(column2)
    top_points = temp[0:256]
    points_list = []

    for i in top_points:
        for j in key_points:
            arr = np.array(j)
            if int(i) == int(arr[1]):
                points_list.append(j)
                break
    return points_list

def data_normalized(key_points):
    x_axis = [i[0] for i in key_points]
    y_axis = [i[1] for i in key_points]
    z_axis = [i[2] for i in key_points]

    normalized_x = np.array(x_axis) / np.linalg.norm(x_axis)
    normalized_y = np.array(y_axis) / np.linalg.norm(y_axis)
    normalized_z = np.array(z_axis) / np.linalg.norm(z_axis)

    normalized_v = np.concatenate((normalized_x, normalized_y, normalized_z))
    FeaSec_model = FeatureSelector(normalized_v.shape[0])
    return FeaSec_model(torch.as_tensor(normalized_v))

def semantic_text(patient_info):

    gender = "he" if patient_info[0] == 1 else "she"
    n_gender = "male" if patient_info[0] == 1 else "female"
    bmi_info = float(patient_info[2])
    is_obesity = ", indicating that " + gender

    ill = ""
    conditions = [
        int(patient_info[3]), int(patient_info[4]),
        int(patient_info[5]), int(patient_info[6])
    ]

    if all(condition == 0 for condition in conditions):
        ill = "and not history of hypertension, diabetes, heart disease, and hyperlipidemia."
    elif all(condition == 1 for condition in conditions):
        ill = "and a medical history of hypertension, diabetes, heart disease, and hyperlipidemia."
    else:
        condition_texts = [
            "hypertension", "diabetes", "heart disease", "hyperlipidemia"
        ]
        present_conditions = [cond_text for cond_text, cond_value in zip(condition_texts, conditions) if cond_value == 1]
        absent_conditions = [cond_text for cond_text, cond_value in zip(condition_texts, conditions) if cond_value == 0]

        present_conditions_str = ", ".join(present_conditions)
        absent_conditions_str = ", ".join(absent_conditions)

        # Handle conjunctions properly
        if len(present_conditions) > 1:
            present_conditions_str = ", and ".join(present_conditions_str.rsplit(", ", 1))  # Add "and" before the last item

        if len(absent_conditions) > 1:
            absent_conditions_str = ", and ".join(absent_conditions_str.rsplit(", ", 1))    # Add "and" before the last item

        ill = f"and a medical history of {present_conditions_str}, but not {absent_conditions_str}."

    if bmi_info < 18.5:
        is_obesity += " is underweight"
    elif 18.5 <= bmi_info < 24:
        is_obesity += " is healthy"
    elif 24 <= bmi_info < 28:
        is_obesity += " is overweight"
    elif 28 <= bmi_info < 32:
        is_obesity += " is obesity"
    elif bmi_info >= 32:
        is_obesity += " is morbid obesity"

    info = [
        f"This {patient_info[7]}-year-old {n_gender} has a neck circumference of {patient_info[1]}cm",
        f"a waist to hip ratio of {patient_info[8]}",
        f"a body mass index of {bmi_info}{is_obesity}",
        ill
    ]

    return ", ".join(info)

def handle_dataset():

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)

    image_path = r'yourpath'
    text_data_path = r'yourpath'

    file_list = os.listdir(image_path)

    image_files = [os.path.join(image_path, i) for i in file_list if i.lower().endswith(('.jpg', '.jpeg', '.png'))]

    dataset = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

      for idx, file in enumerate(image_files):
        file_name = os.path.basename(file)
        file_id = file_name.split(".")[0]

        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
          continue
        annotated_image = image.copy()
        face_info = []

        for face_landmarks in results.multi_face_landmarks:
          for id, lm in enumerate(face_landmarks.landmark):
              ih, iw, ic = image.shape
              x, y, z = int(lm.x * iw), int(lm.y * ih),int(lm.z*((iw+ih)/2))
              face_info.append([x, y ,z])
          #Facial feature markers
          mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())

          mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=drawing_spec,
              connection_drawing_spec=drawing_spec)

          mp_drawing.draw_landmarks(
              image=annotated_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())

        key_points = get_keypoints(face_info)
        id_no = str(file_id)

        with open(text_data_path, encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for ii in reader:
                 id = str(ii['\ufeffid'])
                 if id == id_no:
                    patient_info = [
                        int(ii['Gender']),
                        round(float(ii['Neck_c'].strip()), 1),
                        round(float(ii['BMI'].strip()), 1),
                        int(ii['Hypertension']),
                        int(ii['Diabetes']),
                        int(ii['Heart_D']),
                        int(ii['Hyperlipidemia']),
                        int(ii['Age']),
                        float(ii['WHR'])
                    ]

                    item = {}
                    item['label'] = int(ii['Status'])
                    item['text'] = semantic_text(patient_info)
                    item['image'] = data_normalized(key_points)

                    dataset.append(item)

    processed_data = data_fusion(dataset)
    save_dataset(processed_data, r'yourpath')


