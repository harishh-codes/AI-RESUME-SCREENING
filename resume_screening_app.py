import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googlesearch import search

# Streamlit page config
st.set_page_config(page_title="AI Resume Screening System", layout="wide", initial_sidebar_state="expanded")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to extract email from resume text
def extract_email(text):
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return email_match.group(0) if email_match else "Not Found"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Function to perform skill matching
def skill_matching(job_description, resume_text, skills):
    missing_skills = [skill for skill in skills if skill.lower() not in resume_text.lower()]
    match_score = (len(skills) - len(missing_skills)) / len(skills) if skills else 0
    return match_score, missing_skills

# Function to search YouTube for learning resources
def get_youtube_links(skill, num_links=2):
    query = f"{skill} tutorial site:youtube.com"
    links = [j for j in search(query, num_results=num_links)]
    return links

st.title("📄 AI Resume Screening & Candidate Ranking System")

# UI Layout - Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📈 Visualizations", "💡 Suggestions"])

with tab1:
    st.header("📝 Job Description")
    job_description = st.text_area("Enter the job description")

    st.header("🔍 Required Skills (Comma-separated)")
    skills_input = st.text_input("Enter important skills (e.g., Python, Machine Learning, NLP)")

    st.header("📂 Upload Resumes (PDF)")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        if st.button("🚀 Start Ranking"):
            st.header("🏆 Ranking Resumes")

            resumes = []
            emails = []
            missing_skills_list = []
            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):
                text = extract_text_from_pdf(file)
                resumes.append(text)
                email = extract_email(text)
                emails.append(email)
                progress.progress((i + 1) / len(uploaded_files))

            similarity_scores = rank_resumes(job_description, resumes)

            skills = [skill.strip() for skill in skills_input.split(",") if skill.strip()]
            skill_scores = []
            missing_skills_all = []

            for resume_text in resumes:
                score, missing_skills = skill_matching(job_description, resume_text, skills)
                skill_scores.append(score)
                missing_skills_all.append(", ".join(missing_skills))

            final_scores = np.array(similarity_scores) * 0.7 + np.array(skill_scores) * 0.3
            final_scores_float = [float(score) for score in final_scores]

            results = pd.DataFrame({
                "Resume": [file.name for file in uploaded_files],
                "Email": emails,
                "Score": final_scores_float,
                "Missing Skills": missing_skills_all
            })

            results = results.sort_values(by="Score", ascending=False).reset_index(drop=True)
            results.insert(0, "Rank", range(1, len(results) + 1))
            results["Score"] = results["Score"].apply(lambda x: f"{x * 100:.2f}%")

            st.dataframe(results, height=400)

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Results as CSV", data=csv, file_name="resume_ranking_results.csv", mime="text/csv")

            top_candidates = results.head(3)
            st.subheader("📧 Top 3 Candidate Emails")
            for index, row in top_candidates.iterrows():
                st.write(f"**{row['Rank']}. {row['Resume']}** - 📩 {row['Email']}")

            st.session_state['results'] = results

with tab2:
    if 'results' in st.session_state:
        results = st.session_state['results']
        st.header("📈 Visualizations")

        # Bar Chart - Resume Scores
        st.subheader("📊 Resume Scores")
        plt.figure(figsize=(10, 5))
        sns.barplot(x=results['Resume'], y=[float(score.strip('%')) for score in results['Score']])
        plt.xticks(rotation=45)
        plt.title("Resume Scores")
        plt.xlabel("Resumes")
        plt.ylabel("Score (%)")
        st.pyplot(plt)

        # Pie Chart - Skill Match Distribution
        st.subheader("🍰 Skill Match Distribution")
        skill_counts = [len(skills) - len(missing.split(", ")) for missing in results['Missing Skills']]
        plt.figure(figsize=(6, 6))
        plt.pie(skill_counts, labels=results['Resume'], autopct='%1.1f%%', startangle=140)
        plt.title("Skill Match Percentage")
        st.pyplot(plt)

with tab3:
    if 'results' in st.session_state:
        results = st.session_state['results']
        st.header("💡 Suggestions & Improvements")
        
        for index, row in results.iterrows():
            if row['Missing Skills']:
                st.write(f"📄 **{row['Resume']}** is missing the following skills: {row['Missing Skills']}")
                
                missing_skills = row['Missing Skills'].split(", ")
                for skill in missing_skills:
                    st.write(f"🔍 **Learning Resources for {skill}:**")
                    try:
                        links = get_youtube_links(skill)
                        for link in links:
                            st.markdown(f"- [Learn {skill}]({link})")
                    except:
                        st.write(f"- No resources found for {skill}.")
                        
        st.write("✅ To improve ranking, candidates should include the missing skills in their resume if relevant to their experience.")
