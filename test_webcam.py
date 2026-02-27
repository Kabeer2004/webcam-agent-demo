import cv2
import config

def test_webcam(index):
    print(f"\n--- Testing webcam index {index} ---")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Index {index}: NOT available or occupied.")
        return False
    
    print(f"Index {index}: WORKING! Showing video preview...")
    print("Press 'q' or 'ESC' to stop testing this index and move to the next.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Index {index}: Fails to read frames.")
            break
        
        # Display the video
        cv2.imshow(f"Webcam {index} - Video Preview", frame)
        
        # Wait for 'q' or ESC (27)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    print("Webcam Test Utility")
    print("We will iterate through indices 0 to 4.")
    
    found_any = False
    for i in range(5):
        if test_webcam(i):
            found_any = True
            print(f"Finished testing index {i}.")
        else:
            # Short delay because some systems struggle with rapid open/close
            continue

    if not found_any:
        print("\n[ERROR] No working webcams found between indices 0-4.")
    else:
        print("\nDone! Update WEBCAM_INDEX in config.py with your preferred index.")
