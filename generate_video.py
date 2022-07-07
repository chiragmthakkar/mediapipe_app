import cv2
import time
import argparse
import logging

def record_video(output_path:str, fps:int):
    '''
    Records a video from the webcam and stores it in the given folder and name. Also takes in fps for the output video
    :param output_path: path to the output video file
    :param fps: frames per second
    :return:
    '''

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        logging.warning('Unable to read camera feed')
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # # used to record the time at which we processed current frame
    # new_frame_time = 0

    frame_no = -1

    while (True):
        ret, frame = cap.read()

        if ret == True:
            frame_no += 1

            # Our operations on the frame come here
            gray = frame

            logging.info(f'Writing frame {frame_no}')
            # Write the frame into the file 'output.avi'
            out.write(frame)

            # font which we will be using to display FPS
            font = cv2.FONT_HERSHEY_SIMPLEX
            # time when we finish processing for this frame
            new_frame_time = time.time()

            # Calculating the fps

            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # converting the fps into integer
            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = str(int(fps))

            # putting the FPS count on the frame
            cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            # displaying the frame with fps
            cv2.imshow('frame', gray)

            # # Display the resulting frame
            # cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.debug('Recording stopped')
                break

        # Break the loop
        else:
            break

        # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('-o', '--output_path', type=str, required=True, help='output file path')
    parser.add_argument('-f', '--fps', type=int, default=25, help='fps')
    parser.add_argument('-l', '--log_file', type=str, default='logger.log', help='log file path')

    # Parse the argument
    args = parser.parse_args()
    logging.basicConfig(filename=args.log_file,
                        level=logging.DEBUG,
                        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    record_video(args.output_path)
