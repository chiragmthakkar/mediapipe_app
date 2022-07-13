import cv2
import mediapipe as mp
from pathlib import Path
import glob
import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def apply_face_detection_image(input_path:str, output_path:str):

    # create directory if it doesn't exist
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # For static images:
    IMAGE_FILES = []
    land_dict = {}

    for i in Path(input_path).glob('*.jpg'):
        IMAGE_FILES.append(str(i))

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)

        landmark_list = []

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            print("no fd")
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            for i in range(len(results.multi_face_landmarks[0].landmark)):
                if i in range(478):
                    landmark_list.append((face_landmarks.landmark[i].x, face_landmarks.landmark[i].y,
                                      face_landmarks.landmark[i].z))
                    # land_dict[idx] = [(face_landmarks.landmark[i].x), face_landmarks.landmark[i].y,
                    #                   face_landmarks.landmark[i].z]

                    print(f'face_landmarks: {face_landmarks.landmark[i].x}, {face_landmarks.landmark[i].y}, {face_landmarks.landmark[i].z}')
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
                      landmark_drawing_spec=None,
                      connection_drawing_spec=mp_drawing_styles
                      .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                      image=annotated_image,
                      landmark_list=face_landmarks,
                      connections=mp_face_mesh.FACEMESH_IRISES,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=mp_drawing_styles
                      .get_default_face_mesh_iris_connections_style())
        land_dict[idx] = landmark_list
        cv2.imwrite(f'{str(output_path)}/annotated_image' + str(idx) + '.png', annotated_image)

        # write into json file
        with open('landmarks.json', 'w') as land_json:
            json.dump(land_dict, land_json)




def generate_images_from_video(input_video:str, image_path:str):
    '''
    Convert video to frames
    :param input_video: input path for the video
    :return: none
    '''

    if not Path(image_path).exists():
        Path(image_path).mkdir(parents=True, exist_ok=True)


    cap = cv2.VideoCapture(input_video)
    count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            # continue
            break

        cv2.imwrite(f"{image_path}frame_%d.jpg" % count, image)
        count += 1

def apply_face_detection_video(input_video:str, output_video:str):
    '''

    :param input_video:
    :param output_video:
    :return:
    '''
    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(input_video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame_width, frame_height))

    with mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          # continue
          break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        # write in a video
        out.write(image)

        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_images_from_video('output.mp4', '/Users/chiragthakkar/PycharmProjects/mediapip/images/')
    # apply_face_detection_video('output.mp4', 'output_path.mp4')
    apply_face_detection_image('/Users/chiragthakkar/PycharmProjects/mediapip/images/',
                               '/Users/chiragthakkar/PycharmProjects/mediapip/annotated_images/')
