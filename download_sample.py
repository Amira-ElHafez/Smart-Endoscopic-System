import requests
import os

def download_sample_image():
    # A sample colonoscopy image from a public dataset (Kvasir dataset sample)
    url = "https://raw.githubusercontent.com/simula/kvasir-dataset/master/kvasir-v2/dyed-lifted-polyps/013ec1e8-78c7-4eb1-b1e8-132d72ce666c.jpg"
    
    filename = "sample_endoscopy.jpg"
    
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print("Download complete.")
            else:
                print(f"Failed to download image. Status code: {response.status_code}")
                create_dummy_image(filename)
        except Exception as e:
            print(f"Error downloading image: {e}")
            # Create a dummy image if download fails
            create_dummy_image(filename)
    else:
        print(f"{filename} already exists.")

def create_dummy_image(filename):
    print("Creating a dummy image...")
    import cv2
    import numpy as np
    
    # Create a 512x512 pinkish image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = (120, 100, 200) # BGR
    
    # Add some text
    cv2.putText(img, "Endoscopy Simulation", (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(filename, img)

if __name__ == "__main__":
    download_sample_image()
