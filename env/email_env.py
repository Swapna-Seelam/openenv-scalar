import json
from typing import List, Optional, Tuple, Dict
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

# --------------------------------------------------
# 1. UPDATED TASK DEFINITIONS (At least 3 tasks)
# --------------------------------------------------
TASKS = [
    {
        "task_id": "security_triage",
        "description": "Handle security-related alerts and account issues.",
        "emails": [
            {
                "id": 1,
                "sender": "security@bank.com",
                "subject": "Unauthorized Login Attempt",
                "body": "We detected an unusual login from a new device. Please verify your account immediately.",
                "correct_action": "mark_urgent"
            },
            {
                "id": 5,
                "sender": "ceo@yourcompany.com",
                "subject": "Confidential Project Alpha",
                "body": "I need the status update on Project Alpha by the end of today. This is a top priority.",
                "correct_action": "mark_urgent"
            }
        ]
    },
    {
        "task_id": "marketing_triage",
        "description": "Filter out promotional content and spam.",
        "emails": [
            {
                "id": 2,
                "sender": "newsletter@traveldeals.com",
                "subject": "Last minute flights to Hawaii!",
                "body": "Book now and save 50%! Limited time offer for our loyal subscribers.",
                "correct_action": "mark_spam"
            },
            {
                "id": 6,
                "sender": "promos@shopping.com",
                "subject": "Flash Sale: 70% Off",
                "body": "Don't miss out on our biggest sale of the year. Click here to shop now!",
                "correct_action": "mark_spam"
            }
        ]
    },
    {
        "task_id": "professional_triage",
        "description": "Manage work-related communications and notifications.",
        "emails": [
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
            }
        ]
    }
]

# --------------------------------------------------
# 2. CORRECTED GRADER FUNCTIONS (Strictly 0 < score < 1)
# --------------------------------------------------
def grader_fn(prediction: str, ground_truth: str) -> float:
    """
    Grades the action according to strict range requirements (0.01 to 0.99).
    Never returns 0.0 or 1.0.
    """
    try:
        # Partial scoring logic
        if prediction == ground_truth:
            score = 0.95  # Perfect match
        elif (prediction == "ignore" and ground_truth == "mark_spam") or \
             (prediction == "mark_spam" and ground_truth == "ignore"):
            score = 0.50  # Semi-correct (low risk)
        else:
            score = 0.10  # Incorrect but non-zero

        # Strict clamping to ensure valid range (max 0.99, min 0.01)
        clamped_score = max(0.01, min(0.99, score))
        return float(clamped_score)

    except Exception:
        return 0.10

class EmailTriageEnv(Environment):
    def __init__(self, task_index: int = 0):
        # Support multiple tasks by index
        self.task_data = TASKS[task_index % len(TASKS)]
        self.emails = self.task_data["emails"]
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
            # Fallback for finished state
            return self._get_observation(reward=0.01)

        email = self.emails[self.current_step]
        
        # Use the corrected grader function
        reward = grader_fn(action.action, email["correct_action"])
            
        self.total_reward += reward
        self.current_step += 1
        
        if self.current_step >= len(self.emails):
            self.done = True
            
        return self._get_observation(reward=reward)

    def _get_observation(self, reward: float = 0.01) -> Optional[Observation]:
        if self.done and reward == 0.0:
            # Final step reward adjustment to keep it in range
            reward = 0.01
            
        if self.done and self.current_step >= len(self.emails):
            # To match the logic in inference.py, we might return None or a 'done' observation
            # However, for validation, an observation with done=True is often preferred
            pass
        
        # Ensure the observation reward is also clamped
        reward = max(0.01, min(0.99, reward))
        
        # Determine which email to show (or the last one if done)
        idx = min(self.current_step, len(self.emails) - 1)
        email = self.emails[idx]
        
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
            "task_id": self.task_data["task_id"],
            "current_step": self.current_step,
            "total_reward": round(self.total_reward, 2),
            "done": self.done,
            "total_emails": len(self.emails)
        }

    def close(self):
        pass
