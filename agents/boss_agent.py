# agents/boss_agent.py
from crewai import Agent, Task
# --- Use ChatOpenAI ---
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Instantiate ChatOpenAI with Concurrency Limit ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", # Or your preferred model
    openai_api_key=openai_api_key,
    temperature=0.7,
    max_concurrency=10 # Limit concurrent requests for this LLM instance
)

def create_boss_agent():
    """Creates the Boss Agent using ChatOpenAI."""
    return Agent(
        role="Boss",
        goal="Review and validate drafts to ensure they meet quality standards.",
        backstory="You are a strict editor with high standards, ensuring every draft meets tone, length, and depth requirements.",
        verbose=False,
        llm=llm # Use the configured ChatOpenAI instance
    )

def validation_task(agent: Agent, draft: str, research_content: str, content_tone: str, content_length: str) -> Task:
    """Creates the validation task for the Boss Agent."""
    # (Task description remains the same as the previous version)
    return Task(
        description=f"""
        Validate the following draft:
        --- DRAFT START ---
        {draft}
        --- DRAFT END ---

        Research Content (for context):
        --- RESEARCH START ---
        {research_content}
        --- RESEARCH END ---

        Requirements: Must match {content_tone} tone and {content_length} length, and have sufficient depth based on the research.

        If the draft does not meet requirements, provide specific, actionable feedback for improvement. Focus on clarity, accuracy, tone, length, and depth.

        Output your response ONLY as a valid JSON object string. Start directly with {{ and end directly with }}. No other text before or after.
        The JSON object must have two keys:
        1. 'approved' (boolean): true if the draft meets all requirements, false otherwise.
        2. 'issues' (list): A list of feedback items if 'approved' is false. Each item in the list should be a dictionary with a single key 'instructions' containing a string with specific feedback (e.g., {{"instructions": "Adjust tone to be more professional."}}). If 'approved' is true, this should be an empty list [].

        Example Output (Not Approved): {{"approved": false, "issues": [{{"instructions": "Adjust tone to be more professional."}}, {{"instructions": "Draft lacks depth. Provide more detailed information based on research."}}]}}
        Example Output (Approved): {{"approved": true, "issues": []}}
        """,
        agent=agent,
        expected_output="A single JSON object string with 'approved' (boolean) and 'issues' (list of feedback dictionaries or empty list)."
    )