from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "https://zoocams.elpasozoo.org/bridgesantafe4.m3u8"

# Run inference on an image
results = model.track(video_path, stream=True, show=False, vid_stride=True,
                      classes=0)  # results list

# View results
for r in results:
    if (len(r.boxes.xyxy) > 0):
        print(r.boxes.xyxy[0])
