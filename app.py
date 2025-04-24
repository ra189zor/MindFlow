# app.py
import streamlit as st
from crewai import Crew, Process
# Import agent creation and task functions
# ASSUMES these imports point to files using ChatOpenAI with max_concurrency=10
from agents.idea_agent import create_idea_agent, idea_generation_task
from agents.filter_agent import create_filter_agent, filter_ideas_task
from agents.research_agent import create_research_agent, research_task
from agents.writer_agent import create_writer_agent, writing_task, revision_task
from agents.boss_agent import create_boss_agent, validation_task
# Standard libraries
import json
import os
# import time # No longer needed
import re
import traceback

# --- Caching and Retry Imports ---
from langchain_community.cache import InMemoryCache # Corrected import
from langchain.globals import set_llm_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from litellm.exceptions import APIConnectionError, Timeout, RateLimitError, ServiceUnavailableError, BadRequestError # Keep for retry types
try:
    from openai import RateLimitError as OpenAIRateLimitError, APIError # Add generic APIError
    RETRY_ERRORS = (APIConnectionError, Timeout, RateLimitError, ServiceUnavailableError, BadRequestError, OpenAIRateLimitError, APIError)
except ImportError:
    RETRY_ERRORS = (APIConnectionError, Timeout, RateLimitError, ServiceUnavailableError, BadRequestError) # Fallback

# === Application Setup ===

# --- Setup LLM Caching ---
set_llm_cache(InMemoryCache())
print("Initialized LLM Cache (In-Memory)")

# --- Define Retry Logic (Keep tenacity) ---
retry_on_api_error = retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RETRY_ERRORS),
    reraise=True
)

# --- Wrapper function for Crew Kickoff with Retry ---
@retry_on_api_error
def kickoff_with_retry(crew: Crew):
    """ Executes crew.kickoff() with retry logic for specified API errors. """
    task_description = crew.tasks[0].description[:100] if crew.tasks else "Unknown Task"
    print(f"Attempting kickoff for task: {task_description}...")
    result = crew.kickoff()
    print(f"Kickoff successful for task: {task_description}.")
    return result

# --- Streamlit Page Configuration ---
st.set_page_config( page_title="MindFlow", layout="wide", initial_sidebar_state="expanded" )

# --- Custom CSS ---
st.markdown("""
<style>
    /* Your CSS styles here */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #1a1f2e; color: #e0e0e0; }
    .main-header { font-size: 44px; font-weight: 700; text-align: center; margin-bottom: 10px; color: #ffffff; }
    .sub-header { font-size: 20px; font-weight: 400; text-align: center; color: #a0a0b0; margin-bottom: 40px; }
    .column-header { font-size: 24px; font-weight: 700; margin-bottom: 25px; border-bottom: 2px solid #3a3f4e; padding-bottom: 12px; color: #f0f0f5; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .task-card { background-color: #2a2f3e; border-radius: 8px; padding: 18px; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); border: 1px solid #4a4f5e; animation: fadeIn 0.5s ease-out forwards; opacity: 0; animation-fill-mode: forwards; }
    .task-title { font-weight: 600; margin-bottom: 8px; color: #ffffff; font-size: 15px; }
    .task-desc { color: #c0c0d0; font-size: 14px; line-height: 1.6; }
    .selected-card { background-color: #3a4f6e; border-left: 4px solid #4fc3f7; }
    .footer { text-align: center; margin-top: 60px; padding-top: 25px; border-top: 1px solid #3a3f4e; color: #808090; font-size: 14px; }
    .stButton button { width: 100%; background-color: #4a4f5e; border: none; color: #e0e0e0; padding: 10px 0; border-radius: 6px; font-weight: 500; transition: background-color 0.2s ease; }
    .stButton button:hover { background-color: #5a5f6e; }
    .stDownloadButton button { background-color: #007bff; color: white; }
    .stDownloadButton button:hover { background-color: #0056b3; }
    [data-testid="stStatusWidget"] > div { border-color: #4a4f5e !important; background-color: #2a2f3e !important; }
    [data-testid="stStatusWidget"] p { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True) # Keep CSS

# --- Helper function for Fallback Filter Data ---
def fallback_filter_data(ideas):
    """Creates fallback filter data when JSON parsing or filtering fails."""
    print("Executing fallback_filter_data.")
    if isinstance(ideas, list) and ideas:
        print(f"Using first idea as fallback: {ideas[0]}")
        return { "Idea": [ideas[0]], "Score": [0.5], "Reasoning": ["Fallback: Filter agent output issue."] }
    elif isinstance(ideas, str) and ideas.strip():
        ideas_list = [idea.strip() for idea in ideas.split("\n") if idea.strip()]
        if ideas_list:
            print(f"Using first parsed idea as fallback: {ideas_list[0]}")
            return { "Idea": [ideas_list[0]], "Score": [0.5], "Reasoning": ["Fallback: Filter agent output issue."] }
    print("Fallback: No valid ideas provided."); return { "Idea": ["Default Fallback Idea"], "Score": [0.1], "Reasoning": ["Fallback: Critical error."] }


# --- Initialize Session State ---
default_state = {
    "niche": "", "ideas": None, "filtered_data": None, # Original state keys
    "top_ideas": None, "research_content": None, "draft_text": None,
    "validation_result": None, "pipeline_step": "not_started", "revision_count": 0,
    "needs_more_research": False, "boss_feedback": "", "draft_approved": False,
    "research_cache": {}, "max_revisions": 5, "keywords": [],
    "content_type": "Blog", "target_audience": "Beginners", "content_tone": "Professional",
    "content_length": "Medium"
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# === Sidebar UI === # <-- ENSURE THIS SECTION IS PRESENT
st.sidebar.header("Controls")
niche = st.sidebar.text_input("Niche/Industry", value=st.session_state.niche)
predefined_keywords = ["AI", "SaaS", "productivity", "marketing", "future of work", "LLMs", "automation"]
selected_keywords = st.sidebar.multiselect("Select Keywords", predefined_keywords, default=st.session_state.keywords)
custom_keywords = st.sidebar.text_input("Custom Keywords (comma-separated)")

# Combine and store keywords immediately
all_keywords = selected_keywords
if custom_keywords:
    custom_list = [kw.strip() for kw in custom_keywords.split(",") if kw.strip()]
    all_keywords.extend(k for k in custom_list if k not in all_keywords)
st.session_state.keywords = all_keywords # Update state

# Content options
content_type_options = ["Blog", "Social", "Newsletter", "Product"]
st.session_state.content_type = st.sidebar.radio(
    "Content Type", content_type_options,
    index=content_type_options.index(st.session_state.content_type)
)

audience_options = ["Beginners", "Advanced", "Experts"]
st.session_state.target_audience = st.sidebar.selectbox(
    "Target Audience", audience_options,
    index=audience_options.index(st.session_state.target_audience)
)

tone_options = ["Professional", "Humorous", "Casual"]
st.session_state.content_tone = st.sidebar.radio(
    "Content Tone", tone_options,
    index=tone_options.index(st.session_state.content_tone)
)

# Length slider
length_labels = {0: "Short", 1: "Medium", 2: "Long"}
length_map_inverse = {v: k for k, v in length_labels.items()}
current_length_index = length_map_inverse.get(st.session_state.content_length, 1)
content_length_slider = st.sidebar.slider("Content Length", 0, 2, current_length_index)
st.session_state.content_length = length_labels[content_length_slider] # Update state with label

# Update niche state
st.session_state.niche = niche

# --- Reset Workflow Button ---
if st.sidebar.button("Reset Workflow"):
    print("Workflow Reset Requested.")
    keys_to_clear = list(default_state.keys()) # Get all keys we manage
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize defaults after clearing
    for key, value in default_state.items():
        st.session_state[key] = value
    st.success("Workflow Reset!")
    # import time; time.sleep(1) # Optional brief pause
    st.rerun()

# --- Sidebar Task List ---
st.sidebar.header("Agent Tasks Status")
tasks_display = [
    ("Generate Ideas", "ideas"), # Original first step
    ("Filter Ideas", "filtered_data"),
    ("Pick Top Idea", "top_ideas"),
    ("Research", "research_content"),
    ("Write Draft", "draft_text"),
    ("Validation/Revision", "validation_result")
]
current_step = st.session_state.pipeline_step

for task_name, state_key in tasks_display:
    status_icon = "⬜" # Pending
    step_active = False

    # Determine if the step is active or done
    if state_key == "ideas" and current_step == "ideas": step_active = True
    elif state_key == "filtered_data" and current_step == "filter_ideas": step_active = True
    elif state_key == "top_ideas" and current_step == "filter_ideas": step_active = True # Active during filtering
    elif state_key == "research_content" and current_step == "research": step_active = True
    elif state_key == "draft_text" and current_step == "write_draft": step_active = True
    elif state_key == "validation_result" and current_step == "revision_loop": step_active = True

    # Determine if step is done
    if st.session_state.get(state_key) is not None:
        if state_key == "validation_result" and not st.session_state.draft_approved:
             status_icon = "🔄" # In progress if result exists but not approved
        elif state_key == "validation_result" and st.session_state.draft_approved:
             status_icon = "✅" # Done if approved
        elif state_key != "validation_result":
             status_icon = "✅" # Done for other steps if state has value

    # Override for failure state
    if current_step == "failed": status_icon = "❌"

    # Display logic
    display_text = f"{status_icon} {task_name}"
    if step_active: display_text = f"⏳ {task_name}"
    if state_key == "validation_result" and current_step == "revision_loop":
        display_text += f" (Rev {st.session_state.revision_count})"

    st.sidebar.write(display_text)

if st.session_state.needs_more_research and current_step == "revision_loop":
    st.sidebar.write("Requesting More Research 🔄")

# === END Sidebar UI ===

# === Main Page UI ===
st.markdown('<div class="main-header">MindFlow</div>', unsafe_allow_html=True);
st.markdown('<div class="sub-header">AI Content Generation Pipeline</div>', unsafe_allow_html=True) # Added subheader

# --- Progress Bar ---
pipeline_progress = { "not_started": 0.0, "ideas": 0.1, "filter_ideas": 0.3, "research": 0.5, "write_draft": 0.7, "revision_loop": 0.85, "completed": 1.0, "failed": 1.0 }
current_progress = pipeline_progress.get(st.session_state.pipeline_step, 0.0)
if st.session_state.pipeline_step == "revision_loop": current_progress += min(st.session_state.revision_count * 0.03, 0.1)
st.progress(current_progress)
# --- END Progress Bar ---

# --- UI Display Columns ---
cols = st.columns(5);

# Function to get completion checkmark
def get_checkmark(state_key):
    if state_key == "validation_result": return "✅" if st.session_state.get("draft_approved", False) else ""
    else: return "✅" if st.session_state.get(state_key) is not None else ""

# Column 1: Ideas
with cols[0]:
    checkmark = get_checkmark("ideas")
    st.markdown(f'<div class="column-header">💡 1. Ideas {checkmark}</div>', unsafe_allow_html=True)
    if st.session_state.ideas:
        ideas_list = st.session_state.ideas if isinstance(st.session_state.ideas, list) else []
        if ideas_list:
            for idea in ideas_list[:10]: st.markdown(f'<div class="task-card"><div class="task-desc">{idea}</div></div>', unsafe_allow_html=True)
        else: st.markdown('<div class="task-card"><div class="task-desc">No ideas generated.</div></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="task-card"><div class="task-desc">Waiting...</div></div>', unsafe_allow_html=True)
# Column 2: Filtered
with cols[1]:
    checkmark = get_checkmark("filtered_data")
    st.markdown(f'<div class="column-header">📊 2. Filtered {checkmark}</div>', unsafe_allow_html=True)
    if st.session_state.filtered_data:
        data = st.session_state.filtered_data; ideas_list = data.get("Idea", []); scores_list = data.get("Score", []); reasonings_list = data.get("Reasoning", [])
        if isinstance(ideas_list, list) and len(ideas_list) == len(scores_list) == len(reasonings_list) and ideas_list:
            for idea, score, reasoning in zip(ideas_list, scores_list, reasonings_list):
                 score_formatted = f"{score:.2f}" if isinstance(score, float) else score
                 st.markdown(f'<div class="task-card"><div class="task-title">{idea}</div><div class="task-desc">Score: {score_formatted} - {reasoning}</div></div>', unsafe_allow_html=True)
        else: st.markdown('<div class="task-card"><div class="task-desc">No filtered ideas or data mismatch.</div></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="task-card"><div class="task-desc">Waiting...</div></div>', unsafe_allow_html=True)
# Column 3: Selected
with cols[2]:
    checkmark = get_checkmark("top_ideas")
    st.markdown(f'<div class="column-header">🎯 3. Selected {checkmark}</div>', unsafe_allow_html=True)
    if st.session_state.top_ideas and isinstance(st.session_state.top_ideas, list) and st.session_state.top_ideas:
        st.markdown(f'<div class="task-card selected-card"><div class="task-title">{st.session_state.top_ideas[0]}</div><div class="task-desc">Chosen for development.</div></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="task-card"><div class="task-desc">Waiting...</div></div>', unsafe_allow_html=True)
# Column 4: Research
with cols[3]:
    checkmark = get_checkmark("research_content")
    st.markdown(f'<div class="column-header">🔬 4. Research {checkmark}</div>', unsafe_allow_html=True)
    if st.session_state.research_content:
        st.markdown(f'<div class="task-card"><div class="task-title">Research Summary</div><div class="task-desc" style="max-height: 300px; overflow-y: auto;">{st.session_state.research_content}</div></div>', unsafe_allow_html=True)
    else: st.markdown('<div class="task-card"><div class="task-desc">Waiting...</div></div>', unsafe_allow_html=True)
# Column 5: Draft & Status
with cols[4]:
    checkmark = get_checkmark("validation_result")
    st.markdown(f'<div class="column-header">📄 5. Draft & Status {checkmark}</div>', unsafe_allow_html=True)
    if st.session_state.draft_text:
        st.markdown(f"""
        <div class="task-card">
            <div class="task-title">Draft (Revision {st.session_state.revision_count})</div>
            <div class="task-desc" style="max-height: 200px; overflow-y: auto; border: 1px solid #4a4f5e; padding: 8px; background-color: #3a3f4e;">{st.session_state.draft_text}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.draft_approved:
            # st.success("✅ Draft Approved!") # Now handled by st.status completion
            if st.button("Export Approved Draft"):
                 filepath = "draft_export.txt"
                 try:
                     with open(filepath, "w", encoding="utf-8") as f: f.write(str(st.session_state.draft_text))
                     st.success(f"Draft exported to {filepath}")
                     st.download_button(label="Download Draft", data=str(st.session_state.draft_text), file_name=f"draft_{st.session_state.top_ideas[0][:20].replace(' ','_')}.txt", mime="text/plain")
                 except Exception as e: st.error(f"Failed to export draft: {e}")
        elif st.session_state.validation_result and isinstance(st.session_state.validation_result, dict):
             feedback = st.session_state.validation_result.get("issues", [])
             if feedback:
                 st.warning("Feedback Received (Requires Revision):")
                 feedback_text = "\n".join([f"- {item.get('instructions', 'General feedback.')}" for item in feedback if isinstance(item, dict)])
                 st.text_area("Issues to Address:", feedback_text, height=100, key="feedback_display", disabled=True)
             # else: st.info("Draft under review...") # Covered by st.status
        # else: st.info("Draft written, pending validation...") # Covered by st.status
    else: st.markdown('<div class="task-card"><div class="task-desc">Waiting...</div></div>', unsafe_allow_html=True)


# === Pipeline Execution Logic ===

# --- Create Status Placeholder ---
status_placeholder = st.empty() # Create a placeholder for status messages

# --- Wrapper function to run crew tasks with error handling ---
# Modified to accept status object for updates
def run_crew_task(crew: Crew, task_name: str, status_context):
    """ Runs a crew task using kickoff_with_retry and handles potential errors, updating status. """
    print(f"--- Running Task: {task_name} ---")
    try:
        result = kickoff_with_retry(crew)
        print(f"--- Task '{task_name}' Completed Successfully ---")
        return result
    except Exception as e:
        error_msg = f"Error during {task_name}: {type(e).__name__}"
        print(f"--- ERROR ---"); traceback.print_exc(); print(f"--- END ERROR ---")
        if status_context: # Update status context if provided
             status_context.update(label=f"❌ {error_msg}", state="error", expanded=True)
        st.error(f"{error_msg} - {e}") # Show error in main area too
        st.session_state.pipeline_step = "failed"
        st.stop() # Stop the Streamlit script execution

# --- Start Pipeline Button Logic ---
if st.button("Start Pipeline") and st.session_state.pipeline_step == "not_started":
    if not st.session_state.niche: st.error("Please enter Niche"); st.stop()
    if not st.session_state.keywords: st.error("Please select Keywords"); st.stop()
    print("Start Pipeline button clicked.");
    st.session_state.pipeline_step = "ideas"
    st.rerun()

# --- Sequential Pipeline Steps ---

# Step 1: Generate Ideas
if st.session_state.pipeline_step == "ideas":
    print("Executing Step: Generate Ideas")
    with st.status("💡 Idea Agent thinking...", expanded=True) as status: # Use st.status
        st.write("Searching for Medium trends...")
        agent = create_idea_agent()
        task = idea_generation_task(agent, st.session_state.niche, st.session_state.content_type, st.session_state.target_audience, st.session_state.content_tone, st.session_state.keywords)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        ideas_output = run_crew_task(crew, "Idea Generation", status) # Pass status
        st.write("Processing generated ideas...")
        if ideas_output:
            # Corrected parsing logic
            ideas_list = []
            if isinstance(ideas_output, str):
                lines = [line.strip() for line in ideas_output.split('\n')]
                for line in lines:
                    if line:
                        if line.startswith('"') and line.endswith('"'): line = line[1:-1]
                        elif line.startswith("'") and line.endswith("'"): line = line[1:-1]
                        ideas_list.append(line)
            elif isinstance(ideas_output, list): ideas_list = [str(item).strip().strip('"\'') for item in ideas_output]
            else: ideas_list = [str(ideas_output).strip().strip('"\'')]; st.warning("Unexpected output type from Idea Agent.")
            st.session_state.ideas = ideas_list

            if not st.session_state.ideas:
                st.error("Idea generation failed."); st.session_state.pipeline_step = "failed"
                status.update(label="❌ Idea Generation Failed!", state="error")
            else:
                num_ideas = len(st.session_state.ideas)
                print(f"Generated {num_ideas} ideas.")
                st.write(f"✅ Generated {num_ideas} ideas.")
                st.session_state.pipeline_step = "filter_ideas"
                status.update(label="💡 Ideas Generated!", state="complete", expanded=False)
                st.toast("💡 Ideas ready!")
    if st.session_state.pipeline_step != "failed": st.rerun()


# Step 2: Filter Ideas
if st.session_state.pipeline_step == "filter_ideas":
    print("Executing Step: Filter Ideas")
    if not st.session_state.get("ideas"): st.error("Cannot filter, no ideas."); st.session_state.pipeline_step = "failed"; st.rerun()
    else:
        with st.status("📊 Filter Agent selecting best ideas...", expanded=True) as status: # Use st.status
            st.write("Evaluating relevance and feasibility...")
            agent = create_filter_agent()
            task = filter_ideas_task(agent, st.session_state.ideas, st.session_state.niche, st.session_state.target_audience, st.session_state.keywords)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            crew_output = run_crew_task(crew, "Idea Filtering", status)
            st.write("Processing filtered results...")
            if crew_output:
                # Keep robust JSON processing logic
                raw_output = getattr(crew_output, 'raw', str(crew_output)); filtered_data = None
                try: # Robust JSON processing...
                    filtered_data = json.loads(raw_output)
                    if not isinstance(filtered_data.get("Idea"), list) or not isinstance(filtered_data.get("Score"), list) or not isinstance(filtered_data.get("Reasoning"), list) or len(filtered_data["Idea"]) != len(filtered_data["Score"]) != len(filtered_data["Reasoning"]): raise ValueError("Invalid JSON structure")
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    st.warning(f"Filter JSON invalid: {e}. Trying regex."); print(f"Filter Raw Output:\n{raw_output}")
                    json_match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
                    if json_match:
                        try: filtered_data = json.loads(json_match.group(1)); # Re-validate...
                        except (json.JSONDecodeError, ValueError, TypeError) as e_inner: st.warning(f"Filter regex parse failed: {e_inner}. Using fallback."); filtered_data = fallback_filter_data(st.session_state.ideas)
                    else: st.warning("No JSON via regex. Using fallback."); filtered_data = fallback_filter_data(st.session_state.ideas)

                if not filtered_data or not filtered_data.get("Idea") or not filtered_data["Idea"]:
                    st.warning("Filtering resulted in no ideas."); st.session_state.pipeline_step = "failed"
                    status.update(label="❌ Filtering Failed!", state="error")
                else:
                    num_filtered = len(filtered_data["Idea"])
                    st.write(f"✅ Selected top {num_filtered} ideas.")
                    st.session_state.filtered_data = filtered_data; st.session_state.top_ideas = [filtered_data["Idea"][0]]; print(f"Selected top idea."); st.session_state.pipeline_step = "research"
                    status.update(label="📊 Filtering Complete!", state="complete", expanded=False)
                    st.toast("📊 Ideas filtered!")
        if st.session_state.pipeline_step != "failed": st.rerun()


# Step 3: Research
if st.session_state.pipeline_step == "research":
    print("Executing Step: Research")
    if not st.session_state.get("top_ideas"): st.error("Cannot research, no top idea."); st.session_state.pipeline_step = "failed"; st.rerun()
    else:
        top_idea = st.session_state.top_ideas[0]
        if top_idea in st.session_state.research_cache: # Use cache
             st.session_state.research_content = st.session_state.research_cache[top_idea]; print(f"Using cached research.");
             st.success("✅ Research loaded from cache.") # Show success outside status
             st.toast("🔬 Research loaded from cache!")
             st.session_state.pipeline_step = "write_draft"
             st.rerun() # Rerun immediately after cache hit
        else:
            with st.status("🔬 Research Agent gathering information...", expanded=True) as status: # Use st.status
                 st.write(f"Researching topic: {top_idea[:60]}...")
                 print(f"Running research agent."); agent = create_research_agent(); task = research_task(agent, top_idea)
                 crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
                 research_summary = run_crew_task(crew, "Research", status) # Pass status
                 if research_summary:
                     st.write("✅ Research complete.")
                     st.session_state.research_content = str(research_summary); st.session_state.research_cache[top_idea] = st.session_state.research_content; print("Research successful."); st.session_state.pipeline_step = "write_draft"
                     status.update(label="🔬 Research Complete!", state="complete", expanded=False)
                     st.toast("🔬 Research gathered!")
                 # Error handled in run_crew_task
            if st.session_state.pipeline_step != "failed": st.rerun()


# Step 4: Write Initial Draft
if st.session_state.pipeline_step == "write_draft":
    print("Executing Step: Write Draft")
    if not st.session_state.get("research_content") or not st.session_state.get("top_ideas"): st.error("Cannot write draft, missing inputs."); st.session_state.pipeline_step = "failed"; st.rerun()
    else:
        with st.status("✍️ Writer Agent drafting...", expanded=True) as status: # Use st.status
            st.write("Crafting the initial version...")
            agent = create_writer_agent()
            task = writing_task(agent, st.session_state.top_ideas[0], st.session_state.research_content, st.session_state.content_type, st.session_state.target_audience, st.session_state.content_tone, st.session_state.content_length)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            draft = run_crew_task(crew, "Draft Writing", status) # Pass status
            if draft:
                st.write("✅ Initial draft complete.")
                st.session_state.draft_text = str(draft); print("Initial draft written.")
                st.session_state.pipeline_step = "revision_loop"; st.session_state.validation_result = {"approved": False, "issues": [{"instructions": "Initial draft requires review."}]}; st.session_state.revision_count = 0; st.session_state.draft_approved = False # Setup for loop
                status.update(label="✍️ Initial Draft Complete!", state="complete", expanded=False)
                st.toast("✍️ Draft ready for review!")
            # Error handled in run_crew_task
        if st.session_state.pipeline_step != "failed": st.rerun()


# Step 5: Autonomous Feedback/Revision Loop
if st.session_state.pipeline_step == "revision_loop" and not st.session_state.draft_approved:
    print(f"Executing Step: Revision Loop (Iteration {st.session_state.revision_count})")
    # Max revision check
    if st.session_state.revision_count >= st.session_state.max_revisions:
        st.session_state.draft_approved = True; st.session_state.pipeline_step = "completed"; st.warning(f"Max revisions reached."); st.session_state.validation_result = {"approved": True, "issues": [{"instructions": f"Max revisions reached. Auto-approved."}]}; st.rerun()

    # --- Validation Step ---
    print("Loop Step: Validating current draft.")
    validation_result = None
    validation_status_label = f"🧐 Boss Agent validating (Rev {st.session_state.revision_count})..."
    with st.status(validation_status_label, expanded=True) as status_validation: # Use st.status
        st.write("Checking quality standards...")
        agent = create_boss_agent()
        task = validation_task(agent, st.session_state.draft_text, st.session_state.research_content, st.session_state.content_tone, st.session_state.content_length)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        crew_output = run_crew_task(crew, f"Validation (Rev {st.session_state.revision_count})", status_validation) # Pass status
        st.write("Processing validation results...")
        if crew_output:
            # Keep robust JSON processing logic
            raw_output = getattr(crew_output, 'raw', str(crew_output));
            try:
                validation_result = json.loads(raw_output)
                if not isinstance(validation_result.get("approved"), bool) or not isinstance(validation_result.get("issues"), list): raise ValueError("Invalid JSON structure")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                st.warning(f"Boss JSON invalid: {e}. Trying regex."); print(f"Boss Raw Output:\n{raw_output}")
                json_match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
                if json_match:
                    try: validation_result = json.loads(json_match.group(1)); # Re-validate...
                    except (json.JSONDecodeError, ValueError, TypeError) as e_inner: st.warning(f"Boss regex parse failed: {e_inner}."); validation_result = None
                else: st.warning("No JSON via regex in boss output."); validation_result = None
            if validation_result is None:
                 validation_result = {"approved": False, "issues": [{"instructions": "System could not parse validation feedback."}]}
                 if st.session_state.revision_count >= 2: st.error("Multiple validation parse failures. Auto-approving."); validation_result["approved"] = True
        st.session_state.validation_result = validation_result
        # Update status based on outcome
        if validation_result and validation_result.get("approved", False):
            status_validation.update(label=f"✅ Validation Approved (Rev {st.session_state.revision_count})", state="complete", expanded=False)
        elif validation_result:
            status_validation.update(label=f"🧐 Validation Complete - Revisions Needed (Rev {st.session_state.revision_count})", state="complete", expanded=False)

    # --- Check Approval and Decide Next Action ---
    if st.session_state.pipeline_step != "failed":
        if validation_result and validation_result.get("approved", False):
            st.session_state.draft_approved = True; st.session_state.pipeline_step = "completed"; print(f"Draft approved.");
            st.toast(f"✅ Draft Approved after {st.session_state.revision_count} revisions!")
            st.rerun()
        else:
            # --- Revision is Needed ---
            print("Loop Step: Revision required.")
            st.session_state.revision_count += 1
            issues = validation_result.get("issues", [{"instructions": "Improve clarity."}]) if validation_result else [{"instructions": "Validation failed."}]
            feedback_instructions = " ".join([issue.get("instructions", "") for issue in issues if isinstance(issue, dict)])
            st.session_state.boss_feedback = feedback_instructions
            print(f"Feedback for Rev {st.session_state.revision_count}: {feedback_instructions}")
            needs_more_research = "depth" in feedback_instructions.lower() or "information" in feedback_instructions.lower() or "research" in feedback_instructions.lower()
            st.session_state.needs_more_research = needs_more_research

            if needs_more_research:
                print("Loop Step: Additional research needed.")
                with st.status(f"🔬 Research Agent gathering more info (Rev {st.session_state.revision_count})...", expanded=True) as status_research: # Use st.status
                    st.write("Looking for details based on feedback...")
                    research_agent = create_research_agent()
                    if not st.session_state.top_ideas: st.error("Cannot research, top idea missing."); st.session_state.pipeline_step = "failed"; st.rerun()
                    top_idea = st.session_state.top_ideas[0]; research_context = f"Address feedback: {feedback_instructions}"; cache_key = f"{top_idea}_additional_rev{st.session_state.revision_count}"
                    if cache_key in st.session_state.research_cache:
                        additional_research = st.session_state.research_cache[cache_key]; print("Using cached additional research.")
                        st.session_state.research_content += "\n\nAdditional Research (Cached):\n" + str(additional_research)
                        st.write("✅ Additional research loaded from cache.")
                        status_research.update(label=f"🔬 Add. research cached (Rev {st.session_state.revision_count})", state="complete", expanded=False)
                        st.toast("🔬 Additional research cached!")
                    else:
                        print("Running additional research agent.")
                        task = research_task(research_agent, top_idea, additional_context=research_context)
                        crew = Crew(agents=[research_agent], tasks=[task], process=Process.sequential, verbose=False)
                        additional_research = run_crew_task(crew, f"Additional Research (Rev {st.session_state.revision_count})", status_research) # Pass status
                        if additional_research:
                            st.write("✅ Additional research complete.")
                            additional_research_str = str(additional_research); st.session_state.research_cache[cache_key] = additional_research_str; st.session_state.research_content += "\n\nAdditional Research:\n" + additional_research_str; print("Additional research successful.")
                            status_research.update(label=f"🔬 Add. research finished (Rev {st.session_state.revision_count})", state="complete", expanded=False)
                            st.toast("🔬 Additional research complete!")
                st.session_state.needs_more_research = False

            # --- Perform Revision (only if pipeline hasn't failed) ---
            if st.session_state.pipeline_step != "failed":
                print(f"Loop Step: Revising draft (Revision {st.session_state.revision_count}).")
                with st.status(f"✍️ Writer Agent revising draft (Rev {st.session_state.revision_count})...", expanded=True) as status_revision: # Use st.status
                    st.write("Incorporating feedback...")
                    writer_agent = create_writer_agent()
                    task = revision_task(writer_agent, st.session_state.draft_text, st.session_state.boss_feedback, st.session_state.content_type, st.session_state.target_audience, st.session_state.content_tone, st.session_state.research_content)
                    crew = Crew(agents=[writer_agent], tasks=[task], process=Process.sequential, verbose=False)
                    revised_draft = run_crew_task(crew, f"Revision (Rev {st.session_state.revision_count})", status_revision) # Pass status
                    if revised_draft:
                        st.write(f"✅ Revision {st.session_state.revision_count} complete.")
                        st.session_state.draft_text = str(revised_draft); print("Revision successful.")
                        status_revision.update(label=f"✍️ Revision {st.session_state.revision_count} finished!", state="complete", expanded=False)
                        st.toast(f"✍️ Revision {st.session_state.revision_count} complete!")

            # Re-run to trigger next validation check or show fail state
            st.rerun()


# === Final Status Display ===
if st.session_state.pipeline_step == "completed" and st.session_state.draft_approved:
    st.success("✅ Workflow Completed Successfully!")
elif st.session_state.pipeline_step == "failed":
    # Error shown by run_crew_task's status update
    pass

# Footer
st.markdown('<div class="footer">MindFlow by AB @2025</div>', unsafe_allow_html=True)