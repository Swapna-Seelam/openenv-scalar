import json
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from openenv.core.env_server import Environment

class Action(BaseModel):
    action: str = Field(..., description="Action to take: mark_urgent, mark_spam, ignore, or reply")

class Observation(BaseModel):
    email_id: int
    sender: str
    subject: str
    body: str
    possible_actions: List[str] = ["mark_urgent", "mark_spam", "ignore", "reply"]
    reward: float = 0.0
    done: bool = False

class EmailTriageEnv(Environment):
    def __init__(self):
        # Sample emails for the triage task
        self.emails = [
            {
                "id": 1,
                "sender": "security@bank.com",
                "subject": "Unauthorized Login Attempt",
                "body": "We detected an unusual login from a new device. Please verify your account immediately.",
                "correct_action": "mark_urgent"
            },
            {
                "id": 2,
                "sender": "newsletter@traveldeals.com",
                "subject": "Last minute flights to Hawaii!",
                "body": "Book now and save 50%! Limited time offer for our loyal subscribers.",
                "correct_action": "mark_spam"
            },
            {
                "id": 3,
                "sender": "hr@yourcompany.com",
                "subject": "Quarterly Review Meeting",
                "body": "Hi, please confirm your availability for the quarterly review meeting on Friday at 2 PM.",
                "correct_action": "reply"
            },
            {
                "id": 4,
                "sender": "notification@slack.com",
                "subject": "New message in #general",
                "body": "John Doe sent a message: 'Don't forget the donuts tomorrow!'",
                "correct_action": "ignore"
            },
            {
                "id": 5,
                "sender": "ceo@yourcompany.com",
                "subject": "Confidential Project Alpha",
                "body": "I need the status update on Project Alpha by the end of today. This is a top priority.",
                "correct_action": "mark_urgent"
            }
        ]
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False

    def reset(self) -> Observation:
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False
        return self._get_observation()

    def step(self, action: Action) -> Observation:
        if self.done:
            return self._get_observation(reward=0.0)

        email = self.emails[self.current_step]
        reward = 0.0
        
        # Grading logic
        if action.action == email["correct_action"]:
            reward = 1.0
        else:
            reward = -0.5
            
        self.total_reward += reward
        self.current_step += 1
        
        if self.current_step >= len(self.emails):
            self.done = True
            
        return self._get_observation(reward=reward)

    def _get_observation(self, reward: float = 0.0) -> Optional[Observation]:
        if self.done:
            return None
        
        email = self.emails[self.current_step]
        return Observation(
            email_id=email["id"],
            sender=email["sender"],
            subject=email["subject"],
            body=email["body"],
            reward=reward,
            done=self.done
        )

    @property
    def state(self) -> dict:
        return {
            "current_step": self.current_step,
            "total_reward": self.total_reward,
            "done": self.done,
            "total_emails": len(self.emails)
        }

    def close(self):
        pass
