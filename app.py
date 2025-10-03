from flask import Flask, request, render_template, jsonify
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Synonym and abbreviation dictionary for common skills and attributes
SYNONYMS = {
    'enthusiastic': ['passionate', 'motivated', 'eager', 'proactive', 'driven', 'consistent', 'dedicated'],
    'experienced': ['skilled', 'proficient', 'expert', 'seasoned', 'veteran'],
    'developer': ['programmer', 'engineer', 'coder', 'software engineer', 'software developer'],
    'python': ['python3', 'py', 'python programming'],
    'java': ['core java', 'java programming', 'j2ee', 'java ee'],
    'javascript': ['js', 'node', 'nodejs', 'node.js', 'ecmascript'],
    'machine learning': ['ml', 'ai', 'artificial intelligence'],
    'deep learning': ['dl', 'neural networks', 'neural network'],
    'data structures': ['dsa', 'data structures and algorithms'],
    'algorithms': ['dsa', 'data structures and algorithms', 'algo'],
    'natural language processing': ['nlp', 'text processing', 'text analytics'],
    'computer vision': ['cv', 'image processing'],
    'database': ['db', 'databases', 'dbms'],
    'sql': ['structured query language', 'mysql', 'postgresql', 'oracle'],
    'nosql': ['mongodb', 'cassandra', 'dynamodb'],
    'cloud computing': ['cloud', 'aws', 'azure', 'gcp', 'google cloud'],
    'devops': ['dev ops', 'ci/cd', 'continuous integration'],
    'react': ['reactjs', 'react.js'],
    'angular': ['angularjs', 'angular.js'],
    'leadership': ['management', 'lead', 'manager', 'supervisor', 'team lead'],
    'communication': ['interpersonal', 'verbal', 'written communication', 'communication skills'],
    'teamwork': ['collaboration', 'team player', 'cooperative', 'team work'],
    'problem solving': ['analytical', 'critical thinking', 'troubleshooting', 'problem-solving'],
}

# Abbreviation mapping (key: abbreviation, value: full form)
ABBREVIATIONS = {
    'ml': 'machine learning',
    'dl': 'deep learning',
    'ai': 'artificial intelligence',
    'nlp': 'natural language processing',
    'cv': 'computer vision',
    'dsa': 'data structures and algorithms',
    'oop': 'object oriented programming',
    'api': 'application programming interface',
    'ui': 'user interface',
    'ux': 'user experience',
    'db': 'database',
    'dbms': 'database management system',
    'os': 'operating system',
    'aws': 'amazon web services',
    'gcp': 'google cloud platform',
    'cicd': 'continuous integration continuous deployment',
    'html': 'hypertext markup language',
    'css': 'cascading style sheets',
    'sql': 'structured query language',
    'rest': 'representational state transfer',
    'json': 'javascript object notation',
    'xml': 'extensible markup language',
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

def correct_spelling(text):
    """Correct spelling mistakes using TextBlob"""
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

def preprocess_text(text):
    """Preprocess text: lowercase, remove special chars, tokenize, remove stopwords, lemmatize"""
    # Lowercase
    text = text.lower()
    
    # Spelling correction (optional - can be slow for large texts)
    # text = correct_spelling(text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    
    return ' '.join(processed_tokens), processed_tokens

def extract_key_skills(text):
    """Extract key skills and requirements from text - case insensitive"""
    # Common technical skills and keywords (case insensitive patterns)
    skills_patterns = [
        r'\b(python|java|javascript|c\+\+|c\#|ruby|php|sql|html|css|typescript|go|rust|swift|kotlin)\b',
        r'\b(machine learning|deep learning|ai|artificial intelligence|data science|analytics|nlp|cv)\b',
        r'\b(ml|dl|dsa|oop|api|ui|ux|db|dbms)\b',  # Abbreviations
        r'\b(aws|azure|gcp|google cloud|cloud|docker|kubernetes|terraform)\b',
        r'\b(git|github|gitlab|bitbucket|agile|scrum|devops|ci/cd|cicd)\b',
        r'\b(react|reactjs|angular|angularjs|vue|vuejs|node|nodejs|django|flask|spring|express)\b',
        r'\b(mongodb|mysql|postgresql|oracle|redis|cassandra|dynamodb|nosql)\b',
        r'\b(excel|powerpoint|word|communication|leadership|teamwork|problem solving)\b',
        r'\b(data structures|algorithms|computer vision|natural language processing)\b',
        r'\b(rest|restful|json|xml|api|microservices|backend|frontend|full stack|fullstack)\b',
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for pattern in skills_patterns:
        matches = re.findall(pattern, text_lower)
        found_skills.extend(matches)
    
    # Remove duplicates and return
    return list(set(found_skills))

def expand_with_synonyms(tokens):
    """Expand tokens with their synonyms and handle abbreviations for better matching"""
    expanded_tokens = set(tokens)
    
    for token in tokens:
        token_lower = token.lower()
        
        # Check if token is an abbreviation and expand it
        if token_lower in ABBREVIATIONS:
            expanded_tokens.add(ABBREVIATIONS[token_lower])
            # Also add the full form's words
            expanded_tokens.update(ABBREVIATIONS[token_lower].split())
        
        # Check if token has synonyms
        if token_lower in SYNONYMS:
            expanded_tokens.update(SYNONYMS[token_lower])
        
        # Check if token is a synonym of any key
        for key, synonyms in SYNONYMS.items():
            if token_lower in synonyms:
                expanded_tokens.add(key)
                expanded_tokens.update(synonyms)
        
        # Also check reverse: if abbreviation exists for this full form
        for abbr, full_form in ABBREVIATIONS.items():
            if token_lower in full_form.split() or token_lower == full_form:
                expanded_tokens.add(abbr)
                expanded_tokens.add(full_form)
    
    return list(expanded_tokens)

def calculate_match_score(job_desc, resume_text):
    """Calculate match score between job description and resume"""
    # Preprocess both texts
    job_processed, job_tokens = preprocess_text(job_desc)
    resume_processed, resume_tokens = preprocess_text(resume_text)
    
    # Extract key skills (case insensitive)
    job_skills = extract_key_skills(job_desc)
    resume_skills = extract_key_skills(resume_text)
    
    # Expand with synonyms and abbreviations
    job_tokens_expanded = expand_with_synonyms(job_tokens)
    resume_tokens_expanded = expand_with_synonyms(resume_tokens)
    
    # Also expand skills with abbreviations
    job_skills_expanded = []
    for skill in job_skills:
        job_skills_expanded.append(skill)
        if skill in ABBREVIATIONS:
            job_skills_expanded.append(ABBREVIATIONS[skill])
        # Check reverse
        for abbr, full in ABBREVIATIONS.items():
            if skill == full or skill in full:
                job_skills_expanded.append(abbr)
    
    resume_skills_expanded = []
    for skill in resume_skills:
        resume_skills_expanded.append(skill)
        if skill in ABBREVIATIONS:
            resume_skills_expanded.append(ABBREVIATIONS[skill])
        # Check reverse
        for abbr, full in ABBREVIATIONS.items():
            if skill == full or skill in full:
                resume_skills_expanded.append(abbr)
    
    # Calculate TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([job_processed, resume_processed])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0.0
    
    # Calculate skill match percentage with expanded skills
    matched_skills = []
    if len(job_skills_expanded) > 0:
        for job_skill in job_skills:
            # Check if this skill or its variations exist in resume
            skill_found = False
            for resume_skill in resume_skills_expanded:
                if job_skill in resume_skill or resume_skill in job_skill:
                    skill_found = True
                    break
                # Check with synonyms
                if job_skill.lower() in SYNONYMS and resume_skill in SYNONYMS[job_skill.lower()]:
                    skill_found = True
                    break
            
            if skill_found:
                matched_skills.append(job_skill)
        
        skill_match_ratio = len(matched_skills) / len(set(job_skills))
    else:
        matched_skills = []
        skill_match_ratio = 0.5  # Default if no specific skills found
    
    # Calculate token overlap with synonym expansion
    job_set = set([t.lower() for t in job_tokens_expanded])
    resume_set = set([t.lower() for t in resume_tokens_expanded])
    
    if len(job_set) > 0:
        token_overlap = len(job_set.intersection(resume_set)) / len(job_set)
    else:
        token_overlap = 0.0
    
    # Combined score (weighted average - giving more weight to skill matching)
    final_score = (cosine_sim * 0.3 + skill_match_ratio * 0.5 + token_overlap * 0.2) * 100
    
    # Find missing skills
    missing_skills = [skill for skill in job_skills if skill not in matched_skills]
    
    return {
        'score': round(final_score, 2),
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'total_required_skills': len(set(job_skills)),
        'cosine_similarity': round(cosine_sim * 100, 2),
        'skill_match_percentage': round(skill_match_ratio * 100, 2)
    }

def generate_suggestions(match_data):
    """Generate suggestions based on match score"""
    score = match_data['score']
    suggestions = []
    
    if score >= 90:
        suggestions.append("Excellent match! Your resume is highly suitable for this position.")
        suggestions.append("Highlight your key achievements during the interview.")
        suggestions.append("Prepare specific examples of your work with the matched skills.")
    elif score >= 70:
        suggestions.append("Good match! Your resume aligns well with the job requirements.")
        if match_data['missing_skills']:
            suggestions.append(f"Consider adding these skills to strengthen your application: {', '.join(match_data['missing_skills'][:3])}")
        suggestions.append("Expand descriptions of your relevant experience with quantifiable results.")
    elif score >= 50:
        suggestions.append("Moderate match. Your resume needs improvement to better align with the job.")
        if match_data['missing_skills']:
            suggestions.append(f"Add these important skills to your resume: {', '.join(match_data['missing_skills'])}")
        suggestions.append("Tailor your professional summary to match the job requirements.")
        suggestions.append("Use keywords from the job description throughout your resume.")
    else:
        suggestions.append("Low match. Significant modifications needed to match job requirements.")
        suggestions.append("This position may not align with your current skill set.")
        if match_data['missing_skills']:
            suggestions.append(f"Critical missing skills: {', '.join(match_data['missing_skills'])}")
        suggestions.append("Consider gaining experience in the required areas before applying.")
    
    return suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check-match', methods=['POST'])
def check_match():
    try:
        # Get job description
        job_description = request.form.get('jobDescription')
        
        # Get uploaded resume file
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        resume_file = request.files['resume']
        
        if resume_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(file_path)
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(file_path)
        
        if not resume_text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Calculate match score
        match_data = calculate_match_score(job_description, resume_text)
        
        # Generate suggestions
        suggestions = generate_suggestions(match_data)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        # Return results
        return jsonify({
            'success': True,
            'score': match_data['score'],
            'matched_skills': match_data['matched_skills'],
            'missing_skills': match_data['missing_skills'],
            'total_required_skills': match_data['total_required_skills'],
            'cosine_similarity': match_data['cosine_similarity'],
            'skill_match_percentage': match_data['skill_match_percentage'],
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)