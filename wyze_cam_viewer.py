import os
import time
import cv2
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Union, Dict
from dotenv import load_dotenv
from twisted.protocols.amp import DateTime
from wyze_sdk import Client
from wyze_sdk.errors import WyzeApiError
from wyze_sdk.models.events import (
    Event, EventAlarmType, AiEventType,
    EventFile, EventFileType
)


class WyzeCameras:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize Wyze client
        self.client = self._initialize_client()
        self.cameras: List = []
        self.last_check_time = datetime.now()

        # Initialize display windows
        self.current_video_player = None
        self.camera_frames = {}  # Store latest frame for each camera
        self.display_grid = None
        self.window_name = "Wyze Event Feeds"

    def _initialize_client(self) -> Client:
        """Initialize and authenticate the Wyze client."""
        try:
            # Load environment variables
            load_dotenv()

            return Client(
                email=os.getenv('WYZE_EMAIL'),
                password=os.getenv('WYZE_PASSWORD'),
                key_id=os.getenv('WYZE_API_KEY_ID'),
                api_key=os.getenv('WYZE_API_KEY')
            )

        except WyzeApiError as e:
            print(f"Failed to initialize Wyze client: {e}")
            raise

    def _create_grid_layout(self, n_cameras):
        """Calculate grid dimensions based on number of cameras."""
        if n_cameras <= 2:
            return 1, 2
        elif n_cameras <= 4:
            return 2, 2
        elif n_cameras <= 6:
            return 2, 3
        elif n_cameras <= 9:
            return 3, 3
        else:
            # Add more rows as needed
            cols = 3
            rows = (n_cameras + 2) // 3
            return rows, cols

    def _combine_frames_into_grid(self):
        """Combine individual camera frames into a grid display."""
        if not self.cameras:
            return None

        # Get grid dimensions
        rows, cols = self._create_grid_layout(len(self.cameras))

        # Define frame size for each camera feed
        frame_width = 400
        frame_height = 300

        # Create blank grid
        grid = np.zeros((frame_height * rows, frame_width * cols, 3), dtype=np.uint8)

        # Sort cameras by nickname for consistent positioning
        sorted_cameras = sorted(self.cameras, key=lambda x: x.nickname if hasattr(x, 'nickname') else '')

        # Place frames in grid
        for idx, camera in enumerate(sorted_cameras):
            if idx >= rows * cols:
                break

            row = idx // cols
            col = idx % cols

            # Calculate position in grid
            y_start = row * frame_height
            y_end = (row + 1) * frame_height
            x_start = col * frame_width
            x_end = (col + 1) * frame_width

            # Get frame for this camera
            frame = self.camera_frames.get(camera.mac)
            camera_name = camera.nickname if hasattr(camera, 'nickname') else f"Camera {idx + 1}"

            if frame is not None:
                # Resize frame to fit grid cell
                frame = cv2.resize(frame, (frame_width, frame_height))

                # Add camera name to frame
                cv2.putText(frame, camera_name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Place frame in grid
                grid[y_start:y_end, x_start:x_end] = frame
            else:
                # Create placeholder for camera with no events
                placeholder = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                # Add camera name at top
                cv2.putText(placeholder, camera_name, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Add "No Events" text in center
                text = "No Events"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (frame_width - text_size[0]) // 2
                text_y = (frame_height + text_size[1]) // 2
                cv2.putText(placeholder, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Place placeholder in grid
                grid[y_start:y_end, x_start:x_end] = placeholder

        return grid

    def update_camera_frame(self, camera_mac: str, url: str):
        """Update the frame for a specific camera."""
        try:
            # Download image from URL
            response = requests.get(url)
            if response.status_code == 200:
                # Convert to numpy array
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                # Store frame for this camera
                self.camera_frames[camera_mac] = image

                # Update display
                self.update_display()

        except Exception as e:
            print(f"Error updating camera frame: {e}")

    def update_display(self):
        """Update the main display window with the latest frames."""
        grid = self._combine_frames_into_grid()
        if grid is not None:
            cv2.imshow(self.window_name, grid)
            cv2.waitKey(1)

    def play_video(self, url: str, window_name: str = "Event Video"):
        """Play a video from URL in a window."""
        try:
            # Download video to temporary file
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                temp_file = "temp_video.mp4"
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Stop any existing video playback
                if self.current_video_player is not None:
                    self.current_video_player.release()

                # Play video
                cap = cv2.VideoCapture(temp_file)
                self.current_video_player = cap

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Resize if video is too large
                    max_width = 800
                    if frame.shape[1] > max_width:
                        ratio = max_width / frame.shape[1]
                        dimensions = (max_width, int(frame.shape[0] * ratio))
                        frame = cv2.resize(frame, dimensions)

                    cv2.imshow(window_name, frame)

                    # Break if 'q' is pressed or window is closed
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                # Clean up
                cap.release()
                os.remove(temp_file)
                cv2.destroyWindow(window_name)

        except Exception as e:
            print(f"Error playing video: {e}")

    def get_cameras(self) -> List:
        """Get all cameras associated with the account."""
        try:
            devices = self.client.devices_list()
            self.cameras = [device for device in devices if device.type == 'Camera']
            return self.cameras
        except WyzeApiError as e:
            print(f"Failed to get camera list: {e}")
            return []

    def get_camera_events(self,
                          device_macs: Optional[List[str]] = None,
                          event_types: Optional[List[EventAlarmType]] = None,
                          begin: Optional[datetime] = None,
                          end: Optional[datetime] = None,
                          limit: int = 20) -> Sequence[Event]:
        """Get events for specified cameras."""
        try:
            if not device_macs and not self.cameras:
                self.get_cameras()
                device_macs = [cam.mac for cam in self.cameras]

            if not begin:
                begin = datetime.now() - timedelta(hours=24)
            if not end:
                end = datetime.now()

            events = self.client.events.list(
                device_ids=device_macs,
                event_values=event_types,
                begin=begin,
                end=end,
                limit=limit,
                order_by=2
            )

            return events

        except WyzeApiError as e:
            print(f"Failed to get events: {e}")
            return []

    def process_event_files(self, event: Event) -> Dict[str, List[str]]:
        """Process files associated with an event."""
        files = {
            'images': [],
            'videos': []
        }

        for file in event.files:
            if file.type == EventFileType.IMAGE:
                files['images'].append(file.url)
            elif file.type == EventFileType.VIDEO:
                files['videos'].append(file.url)

        return files

    def get_ai_detected_objects(self, event: Event) -> List[AiEventType]:
        """Get AI-detected objects from event tags."""
        return [tag for tag in event.tags if tag is not None]

    def monitor_events(self,
                       callback,
                       device_macs: Optional[List[str]] = None,
                       event_types: Optional[List[EventAlarmType]] = None,
                       ai_event_filters: Optional[List[AiEventType]] = None,
                       check_interval: int = 10):
        """Monitor camera events continuously."""
        print(f"Starting event monitoring for cameras: {device_macs or 'all'}")
        print(f"Monitoring event types: {event_types or 'all'}")
        if ai_event_filters:
            print(f"Filtering for AI events: {[e.describe() for e in ai_event_filters]}")

        try:
            while True:
                print(f'Checking for events - {datetime.now()}')
                current_time = datetime.now()

                events = self.get_camera_events(
                    device_macs=device_macs,
                    event_types=event_types,
                    begin=self.last_check_time,
                    end=current_time
                )

                self.last_check_time = current_time

                for event in events:
                    if ai_event_filters:
                        ai_detections = self.get_ai_detected_objects(event)
                        if not any(det in ai_event_filters for det in ai_detections):
                            continue

                    files = self.process_event_files(event)
                    event_data = {
                        'event': event,
                        'files': files,
                        'ai_detections': self.get_ai_detected_objects(event)
                    }

                    callback(event_data)

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\nStopping event monitoring...")
        except Exception as e:
            print(f"Error during event monitoring: {e}")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        wyze_cams = WyzeCameras()

        cameras = wyze_cams.get_cameras()
        print(f"\nFound {len(cameras)} cameras:")
        for camera in cameras:
            print(f"- {camera.nickname} ({camera.mac})")


        def handle_event(event_data: dict):
            event: Event = event_data['event']
            files = event_data['files']
            ai_detections = event_data['ai_detections']

            print(f"\nNew event detected:")
            print(f"From: {event.parameters.get('beginTime', '')}")
            print(f"To: {event.parameters.get('endTime', '')}")
            print(f"Length: {event.parameters.get('uploadedVideoLen', 0)} seconds")
            print(f"Device: {event.mac}")
            print(f"Type: {event.alarm_type.describe()}")

            if ai_detections:
                print("AI Detections:", [det.describe() for det in ai_detections])

            # Update the camera frame in the grid if image is available
            if files['images']:
                wyze_cams.update_camera_frame(event.mac, files['images'][0])

            # Play video in separate window if available
            if files['videos']:
                print(f"Playing video...")
                wyze_cams.play_video(files['videos'][0])


        wyze_cams.monitor_events(
            callback=handle_event,
            event_types=[EventAlarmType.MOTION],
            ai_event_filters=[AiEventType.PERSON, AiEventType.PACKAGE],
            check_interval=15
        )

    except Exception as e:
        print(f"An error occurred: {e}")