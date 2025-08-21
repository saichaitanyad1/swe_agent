import os
from google.adk.agents import LlmAgent
from google.adk.sessions import Session
from google.adk.tools import FunctionTool
from google.adk.executors import SequentialExecutor

# --- Step 1: Define a Tool ---
# This function simulates fetching a user profile from a database.
def get_user_profile(user_id: str) -> dict:
    """Fetches a user profile dictionary based on a user ID."""
    print(f"\n---> [Tool Executing] Fetching profile for user: {user_id}...")
    # In a real application, this would query a database or API.
    profiles = {
        "user123": {
            "name": "Alex",
            "email": "alex.c@example.com",
            "city": "San Francisco"
        },
        "user456": {
            "name": "Brenda",
            "email": "brenda.m@example.com",
            "city": "New York"
        }
    }
    return profiles.get(user_id, {})

# --- Step 2: Define Multiple Agents that will use the tool's output ---

# Agent 1: Drafts a welcome email.
email_agent = LlmAgent(
    model="gemini-1.5-flash",
    instruction=(
        "You are an onboarding specialist. Using the user profile data from "
        "`session.state['user_profile']`, draft a friendly and personalized "
        "welcome email. Address the user by their name."
    ),
    output_key="welcome_email_draft" # Where this agent stores its result
)

# Agent 2: Provides a local recommendation.
recommender_agent = LlmAgent(
    model="gemini-1.5-flash",
    instruction=(
        "You are a local tour guide. Using the user's city from "
        "`session.state['user_profile']`, recommend one fun weekend "
        "activity in that city."
    ),
    output_key="local_recommendation" # Where this agent stores its result
)

# --- Step 3: Set up the Session and Executor ---

# Create a session to hold the shared state.
session = Session()

# Define the sequential workflow.
workflow = SequentialExecutor(
    session=session,
    chain=[
        # The first step is to run the tool and store its output.
        FunctionTool(
            function=get_user_profile,
            # The output of this tool will be stored in session.state['user_profile']
            output_key="user_profile"
        ),
        # The second step is to run the email agent.
        # It will automatically read from session.state['user_profile'].
        email_agent,
        # The third step is to run the recommender agent.
        # It ALSO reads from the same session.state['user_profile'].
        recommender_agent
    ]
)

# --- Step 4: Run the Workflow ---

print("🚀 Starting multi-agent workflow...")
# We provide the initial input required by the first tool in the chain.
initial_input = {"user_id": "user123"}
workflow.execute(initial_input)

# --- Step 5: Inspect the Shared Session State ---

print("\n\n✅ Workflow finished. Inspecting final session state:")
print("---------------------------------------------------------")
# The 'user_profile' key contains the output from the tool.
print(f"Tool Output [user_profile]:\n{session.state.get('user_profile')}\n")

# The 'welcome_email_draft' key contains the output from the first agent.
print(f"Agent 1 Output [welcome_email_draft]:\n{session.state.get('welcome_email_draft')}\n")

# The 'local_recommendation' key contains the output from the second agent.
print(f"Agent 2 Output [local_recommendation]:\n{session.state.get('local_recommendation')}")
print("---------------------------------------------------------")
