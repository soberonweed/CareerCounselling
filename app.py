import gradio as gr
import pandas as pd
import pickle

# Load the model directly
with open('rfweights.pkl', 'rb') as f:
    rfmodel = pickle.load(f)

# Define mappings directly (example values, adjust as needed)
certificates_references = {
    "app development": 0,
    "distro making": 1,
    "full stack": 2,
    "hadoop": 3,
    "information security": 4,
    "machine learning": 5,
    "python": 6,
    "r programming": 7,
    "shell programming": 8
}

workshop_references = {
    "cloud computing": 0,
    "data science": 1,
    "database security": 2,
    "game development": 3,
    "hacking": 4,
    "system designing": 5,
    "testing": 6,
    "web technologies": 7
}

subjects_interest_references = {
    "cloud computing": 0,
    "Computer Architecture": 1,
    "data engineering": 2,
    "hacking": 3,
    "IOT": 4,
    "Management": 5,
    "networks": 6,
    "parallel computing": 7,
    "programming": 8,
    "Software Engineering": 9
}

career_interest_references = {
    "Business process analyst": 0,
    "cloud computing": 1,
    "developer": 2,
    "security": 3,
    "system developer": 4,
    "testing": 5
}

company_intends_references = {
    "BPA": 0,
    "Cloud Services": 1,
    "Finance": 2,
    "Product based": 3,
    "product development": 4,
    "SAaS services": 5,
    "Sales and Marketing": 6,
    "Service Based": 7,
    "Testing and Maintenance Services": 8,
    "Web Services": 9
}

book_interest_references = {
    "Action and Adventure": 0,
    "Anthology": 1,
    "Art": 2,
    "Autobiographies": 3,
    "Biographies": 4,
    "Children's": 5,
    "Comics": 6,
    "Cookbooks": 7,
    "Diaries": 8,
    "Dictionaries": 9,
    "Drama": 10,
    "Encyclopedias": 11,
    "Fantasy": 12,
    "Guide": 13,
    "Health": 14,
    "History": 15,
    "Horror": 16,
    "Journals": 17,
    "Math": 18,
    "Mystery": 19,
    "Poetry": 20,
    "Prayer books": 21,
    "Religion-Spirituality": 22,
    "Romance": 23,
    "Satire": 24,
    "Science": 25,
    "Science fiction": 26,
    "Self-help": 27,
    "Series": 28,
    "Travel": 29,
    "Trilogy": 30
}

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
        "certificate": [certificates_references[certificate_code]],
        "workshop": [workshop_references[workshop_code]],
        "read_writing_skills": [0 if read_writing_skill == "poor" else 1 if read_writing_skill == "medium" else 2],
        "memory_capability": [0 if memory_capability == "poor" else 1 if memory_capability == "medium" else 2],
        "subject_interest": [subjects_interest_references[subject_interest]],
        "career_interest": [career_interest_references[career_interest]],
        "company_intend": [company_intends_references[company_intend]],
        "senior_elder_advise": [senior_elder_advise],
        "book_interest": [book_interest_references[book_interest]],
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
        gr.Dropdown(list(certificates_references.keys()), label="Certificate Taken"),
        gr.Dropdown(list(workshop_references.keys()), label="Workshop Attended"),
        gr.Dropdown(["poor", "medium", "excellent"], label="Reading/Writing Skill"),
        gr.Dropdown(["poor", "medium", "excellent"], label="Memory Capability"),
        gr.Dropdown(list(subjects_interest_references.keys()), label="Interested Subject"),
        gr.Dropdown(list(career_interest_references.keys()), label="Career Interest"),
        gr.Dropdown(list(company_intends_references.keys()), label="Preferred Company Type"),
        gr.Radio(["Yes", "No"], label="Advice from Seniors"),
        gr.Dropdown(list(book_interest_references.keys()), label="Favorite Book Genre"),
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
