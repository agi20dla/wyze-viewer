import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from wyze_cam_viewer import WyzeCameras
from wyze_sdk.models.events import Event, EventAlarmType, AiEventType


class EventAction(ABC):
    """Base class for actions that can be taken in response to events."""

    @abstractmethod
    def execute(self, event_data: dict) -> None:
        """Execute the action based on event data."""
        pass


class WyzeEventAction(EventAction):
    """Action to handle camera display updates."""

    def __init__(self, cameras: WyzeCameras):
        self.cameras = cameras

    def execute(self, event_data: dict) -> None:
        event: Event = event_data['event']
        files = event_data['files']

        # Update the camera feed display if image is available
        if files['images']:
            self.cameras.update_camera_frame(event.mac, files['images'][0])

        # Play video in separate window if available
        if files['videos']:
            print(f"Playing video from {event.mac}...")
            self.cameras.play_video(files['videos'][0])


class NotificationAction(EventAction):
    """Action to send notifications."""

    def execute(self, event_data: dict) -> None:
        event: Event = event_data['event']
        ai_detections = event_data['ai_detections']

        # Example notification - replace with your preferred notification method
        message = f"Alert from {event.mac}: "
        if ai_detections:
            message += f"Detected {[det.describe() for det in ai_detections]}"
        print(f"NOTIFICATION: {message}")


class EventLogger(EventAction):
    """Action to log events to a file."""

    def __init__(self, log_file: str = "wyze_events.log"):
        self.log_file = log_file

    def execute(self, event_data: dict) -> None:
        event: Event = event_data['event']
        with open(self.log_file, 'a') as f:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'device': event.mac,
                'event_type': event.alarm_type.describe(),
                'ai_detections': [det.describe() for det in event_data['ai_detections']],
                'has_image': len(event_data['files']['images']) > 0,
                'has_video': len(event_data['files']['videos']) > 0
            }
            f.write(json.dumps(log_entry) + '\n')


class SecurityAction(EventAction):
    """Action to handle security-related responses."""

    def execute(self, event_data: dict) -> None:
        event: Event = event_data['event']
        ai_detections = event_data['ai_detections']

        # Example security logic
        if any(det == AiEventType.PERSON for det in ai_detections):
            if self._is_outside_hours():
                print(f"SECURITY ALERT: Person detected outside normal hours on {event.mac}")
                # Add your security response here (e.g., trigger alarm, call security)

    def _is_outside_hours(self) -> bool:
        """Check if current time is outside normal hours."""
        current_hour = datetime.now().hour
        return current_hour < 6 or current_hour > 22



class WyzeAIAgent:
    """AI agent for monitoring and responding to Wyze camera events."""

    def __init__(self):
        self.cameras = WyzeCameras()
        self.actions: List[EventAction] = []
        self.event_rules: Dict[EventAlarmType, List[EventAction]] = {}
        self.ai_rules: Dict[AiEventType, List[EventAction]] = {}

    def add_action(self, action: EventAction) -> None:
        """Add an action to be executed for all events."""
        self.actions.append(action)

    def add_event_rule(self, event_type: EventAlarmType, action: EventAction) -> None:
        """Add an action to be executed for specific event types."""
        if event_type not in self.event_rules:
            self.event_rules[event_type] = []
        self.event_rules[event_type].append(action)

    def add_ai_rule(self, ai_type: AiEventType, action: EventAction) -> None:
        """Add an action to be executed for specific AI detection types."""
        if ai_type not in self.ai_rules:
            self.ai_rules[ai_type] = []
        self.ai_rules[ai_type].append(action)


    def handle_event(self, event_data: dict) -> None:
        """Handle incoming events and execute appropriate actions."""
        event: Event = event_data['event']
        ai_detections = event_data['ai_detections']

        # Execute global actions
        for action in self.actions:
            action.execute(event_data)

        # Execute event-specific actions
        if event.alarm_type in self.event_rules:
            for action in self.event_rules[event.alarm_type]:
                action.execute(event_data)

        # Execute AI-specific actions
        for detection in ai_detections:
            if detection in self.ai_rules:
                for action in self.ai_rules[detection]:
                    action.execute(event_data)


    def run(self,
            event_types: Optional[List[EventAlarmType]] = None,
            ai_event_filters: Optional[List[AiEventType]] = None,
            check_interval: int = 30):
        """Start the AI agent's monitoring process."""

        print("Starting Wyze AI Agent...")
        print(f"Monitoring for events: {[et.describe() for et in (event_types or [])]}")
        print(f"AI event filters: {[af.describe() for af in (ai_event_filters or [])]}")

        self.cameras.monitor_events(
            callback=self.handle_event,
            event_types=event_types,
            ai_event_filters=ai_event_filters,
            check_interval=check_interval
        )


if __name__ == "__main__":
    # Example usage
    try:
        # Create AI agent
        agent = WyzeAIAgent()

        # Add display action for all events
        agent.add_action(WyzeEventAction(agent.cameras))

        # Add general logging for all events
        agent.add_action(EventLogger())

        # Add notification action for motion events
        agent.add_event_rule(
            EventAlarmType.MOTION,
            NotificationAction()
        )

        # Add security action for person detection
        agent.add_ai_rule(
            AiEventType.PERSON,
            SecurityAction()
        )


        # Get list of cameras before starting
        cameras = agent.cameras.get_cameras()
        print(f"\nMonitoring {len(cameras)} cameras:")
        for camera in cameras:
            print(f"- {camera.nickname} ({camera.mac})")

        # Start monitoring
        agent.run(
            event_types=[EventAlarmType.MOTION, EventAlarmType.FACE, EventAlarmType.TRIGGERED, EventAlarmType.SMOKE, EventAlarmType.SOUND,],
            ai_event_filters=[AiEventType.PERSON, AiEventType.PACKAGE, AiEventType.PET, AiEventType.VEHICLE],
            check_interval=int(os.getenv('CHECK_INTERVAL_SECS', '30'))
        )

    except Exception as e:
        print(f"An error occurred: {e}")