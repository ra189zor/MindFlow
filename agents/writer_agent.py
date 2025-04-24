# agents/writer_agent.py
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
    max_concurrency=10 #Limit concurrent requests for this LLM instance
)

# Define the Writer Agent
def create_writer_agent():
    """Creates the Writer Agent using ChatOpenAI."""
    return Agent(
        role="Writer",
        goal="Create high-quality drafts based on research and revise them based on specific feedback.",
        backstory=(
            "You are an experienced writer with a knack for crafting engaging content tailored to specific audiences and tones. "
            "You excel at incorporating research findings smoothly into your drafts. "
            "When feedback is provided, you carefully analyze it and thoughtfully revise your work to meet all requirements."
        ),
        verbose=False,
        llm=llm # Use the configured ChatOpenAI instance
    )

# Define the Writing Task
def writing_task(agent: Agent, idea: str, research_content: str, content_type: str, target_audience: str, content_tone: str, content_length: str) -> Task:
    """Creates the initial writing task for the Writer Agent."""
    # (Task description remains the same as the previous version)
    return Task(
        description=f"""
        Write a '{content_length.lower()}' length '{content_type.lower()}' piece for a '{target_audience.lower()}' audience in a '{content_tone.lower()}' tone.

        The main topic is: "{idea}"

        Use the following Research Content as the basis for your writing. Incorporate the key findings and information naturally within the text:
        --- RESEARCH START ---
        {research_content}
        --- RESEARCH END ---

        Ensure the draft is well-structured (e.g., intro, body paragraphs, conclusion for a blog post), engaging, directly addresses the topic '{idea}', accurately reflects the research, and strictly adheres to the specified tone and length requirements.
        """,
        agent=agent,
        expected_output=f"A complete draft of the {content_type.lower()} piece, meeting all specified requirements (length, tone, audience, topic, research incorporation)."
    )

# Define the Revision Task
def revision_task(agent: Agent, draft: str, feedback: str, content_type: str, target_audience: str, content_tone: str, research_content: str) -> Task:
    """Creates the revision task for the Writer Agent."""
    # (Task description remains the same as the previous version)
    return Task(
        description=f"""
        Revise the following draft of a {content_type.lower()} (intended for {target_audience.lower()}, in a {content_tone.lower()} tone).

        You MUST address ALL points raised in the Feedback provided below.

        Original Draft:
        --- DRAFT START ---
        {draft}
        --- DRAFT END ---

        Research Content (for context and ensuring accuracy):
        --- RESEARCH START ---
        {research_content}
        --- RESEARCH END ---

        Feedback for Revision:
        --- FEEDBACK START ---
        {feedback}
        --- FEEDBACK END ---

        Your goal is to produce an improved version of the draft that directly incorporates all the feedback. Ensure the revised draft still maintains the original topic, specified tone, and style, while fixing the issues mentioned in the feedback.
        """,
        agent=agent,
        expected_output=f"A revised draft of the {content_type.lower()} that incorporates all the provided feedback while maintaining tone and style."
    )