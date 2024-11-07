import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load the dataset and model
data = pd.read_csv("mldata.csv")
with open('rfweights.pkl', 'rb') as f:
    rfmodel = pickle.load(f)

# Prepare mappings for categorical features (adjusted to keep things simple)
categorical_cols = data[['certifications', 'workshops', 'Interested subjects', 
                         'interested career area ', 'Type of company want to settle in?', 
                         'Interested Type of Books']]

mappings = {}
for col in categorical_cols.columns:
    col_mapping = {name: code for name, code in zip(categorical_cols[col].unique(), data[col].astype('category').cat.codes.unique())}
    mappings[col] = col_mapping

# Prediction function
def rfprediction(name, logical_thinking, hackathon_attend, coding_skills, public_speaking_skills,
                 self_learning, extra_course, certificate_code, workshop_code, read_writing_skill, memory_capability,
                 subject_interest, career_interest, company_intend, senior_elder_advise, book_interest, introvert_extro,
                 team_player, management_technical, smart_hardworker):

    # Build the input dataframe
    user_data = pd.DataFrame({
        "logical_thinking": [logical_thinking],
        "hackathon_attend": [hackathon_attend],
        "coding_skills": [coding_skills],
        "public_speaking_skills": [public_speaking_skills],
        "self_learning": [self_learning],
        "extra_course": [extra_course],
        "certificate": [mappings['certifications'][certificate_code]],
        "workshop": [mappings['workshops'][workshop_code]],
        "read_writing_skills": [0 if read_writing_skill == "poor" else 1 if read_writing_skill == "medium" else 2],
        "memory_capability": [0 if memory_capability == "poor" else 1 if memory_capability == "medium" else 2],
        "subject_interest": [mappings['Interested subjects'][subject_interest]],
        "career_interest": [mappings['interested career area '][career_interest]],
        "company_intend": [mappings['Type of company want to settle in?'][company_intend]],
        "senior_elder_advise": [senior_elder_advise],
        "book_interest": [mappings['Interested Type of Books'][book_interest]],
        "introvert_extro": [introvert_extro],
        "team_player": [team_player],
        "management_technical": [1 if management_technical == "Management" else 0],
        "smart_hardworker": [1 if smart_hardworker == "smart worker" else 0]
    })

    # Prediction
    prediction = rfmodel.predict(user_data)[0]
    prediction_proba = rfmodel.predict_proba(user_data)[0]

    # Output as a dictionary with probabilities
    result = {
        "Applications Developer": prediction_proba[0],
        "CRM Technical Developer": prediction_proba[1],
        "Database Developer": prediction_proba[2],
        "Mobile Applications Developer": prediction_proba[3],
        "Network Security Engineer": prediction_proba[4],
        "Software Developer": prediction_proba[5],
        "Software Engineer": prediction_proba[6],
        "Software QA/Testing": prediction_proba[7],
        "Systems Security Administrator": prediction_proba[8],
        "Technical Support": prediction_proba[9],
        "UX Designer": prediction_proba[10],
        "Web Developer": prediction_proba[11],
    }

    return result

# Gradio Interface
interface = gr.Interface(
    fn=rfprediction,
    inputs=[
        gr.Textbox(placeholder="Enter your name", label="Name"),
        gr.Slider(1, 9, step=1, label="Logical Thinking (1-9)"),
        gr.Slider(0, 6, step=1, label="Hackathons Attended"),
        gr.Slider(1, 9, step=1, label="Coding Skills (1-9)"),
        gr.Slider(1, 9, step=1, label="Public Speaking Skills (1-9)"),
        gr.Radio(["Yes", "No"], label="Self-Learner"),
        gr.Radio(["Yes", "No"], label="Takes Extra Courses"),
        gr.Dropdown(list(mappings['certifications'].keys()), label="Certificate Taken"),
        gr.Dropdown(list(mappings['workshops'].keys()), label="Workshop Attended"),
        gr.Dropdown(["poor", "medium", "excellent"], label="Reading/Writing Skill"),
        gr.Dropdown(["poor", "medium", "excellent"], label="Memory Capability"),
        gr.Dropdown(list(mappings['Interested subjects'].keys()), label="Interested Subject"),
        gr.Dropdown(list(mappings['interested career area '].keys()), label="Career Interest"),
        gr.Dropdown(list(mappings['Type of company want to settle in?'].keys()), label="Preferred Company Type"),
        gr.Radio(["Yes", "No"], label="Advice from Seniors"),
        gr.Dropdown(list(mappings['Interested Type of Books'].keys()), label="Favorite Book Genre"),
        gr.Radio(["Yes", "No"], label="Introvert (No for Extrovert)"),
        gr.Radio(["Yes", "No"], label="Works Well in Team"),
        gr.Dropdown(["Management", "Technical"], label="Preferred Area"),
        gr.Dropdown(["hard worker", "smart worker"], label="Work Style"),
    ],
    outputs=gr.Label(num_top_classes=5),
    title="IT-Career Recommendation System",
    description="Provides career recommendations based on input skills and interests."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
