import streamlit as st
import pickle
import pandas as pd


# Load your model
@st.cache_data
def get_data():
    emp_df = pd.read_csv('employers_database.csv')
    return emp_df

@st.cache_resource
def load_model():
    model = pickle.load(open('disability_employment_matching_model.pkl', 'rb'))
    return model

features = ['disability_type', 
            'experience_years', 
            'work_preference',
            'company_size', 
            'industry', 
            'remote_policy',
            'emp_needs_noise reduction', 
            'employer_provides_noise reduction',
            'emp_needs_modified training materials',
            'employer_provides_modified training materials',
            'emp_needs_remote work', 
            'employer_provides_remote work',
            'emp_needs_physical workspace modifications',
            'employer_provides_physical workspace modifications',
            'emp_needs_flexible schedule', 
            'employer_provides_flexible schedule',
            'emp_needs_interpreter services',
            'employer_provides_interpreter services',
            'emp_needs_assistive technology',
            'employer_provides_assistive technology',
        ]

all_accommodation_types = [
    "noise reduction", "modified training materials", "remote work", 
    "physical workspace modifications", "flexible schedule", 
    "interpreter services", "assistive technology"
]

st.title("Disability Employment Matching Tool")

# Create input form for employee data
st.header("Employee Information")
disability_type = st.selectbox("Disability Type", ["mobility", "vision", "hearing", "cognitive", "psychiatric", "chronic health"])
experience_years = st.slider("Years of Experience", 0, 30, 5)
work_preference = st.selectbox("Work Preference", ["fully remote", "hybrid", "in-office"])
industry_preference = st.selectbox("Industry Sector", ["tech", "healthcare", "finance", "retail", "manufacturing"])
selected_options = st.multiselect(
    "Select accommodations needed:",
    options=all_accommodation_types,
    default=None  # No options selected by default
)

# Create input form for employer selection
# st.header("Available Employers")

employers = get_data()
model = load_model()

# employer_names = employers['name'].tolist()
# selected_employers = st.multiselect("Select employers to evaluate", employer_names)

if st.button("Find Best Matches"):
    results = []
    # Filter employers by the selected industry
    filtered_employers = employers[employers['industry'] == industry_preference]
    for _, employer in filtered_employers.iterrows():
        # Create feature dictionary
        features = {
            'disability_type': disability_type,
            'experience_years': experience_years,
            'work_preference': work_preference,
            'company_size': employer['company_size'],
            'industry': employer['industry'],
            'remote_policy': employer['remote_policy'],
        }
        
       # Add accommodation binary flags
        for acc_type in all_accommodation_types:
            # 1 if this accommodation is selected by user
            features[f'emp_needs_{acc_type}'] = int(acc_type in selected_options)
            # 1 if employer provides this accommodation
            features[f'employer_provides_{acc_type}'] = int(acc_type in employer['available_accommodations'])
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([features])
        
        # Get prediction
        match_score = model.predict(input_df)[0]

        # Add to results
        results.append({
            'employer_name': employer['name'],
            'match_score': match_score * 100  # Convert to percentage
        })
    
    # Sort and display top matches
    results_df = pd.DataFrame(results).sort_values('match_score', ascending=False).head(25)
    st.write("got results")
    if not results_df.empty:
        # Get the highest score
        st.write("got results")       
        top_score = results_df['match_score'].iloc[0]
        
        # Get all matches with the highest score
        top_matches = results_df[results_df['match_score'] == top_score]
        num_top_matches = len(top_matches)
        
        # Apply your display logic based on the number of matches with the same top score
        if num_top_matches <= 5:
            # Show all matches (5 or fewer) with their individual scores
            st.subheader(f"Top {num_top_matches} Employer Matches")
            for i, (index, row) in enumerate(results_df.head(5).iterrows()):
                st.write(f"{i+1}. {row['employer_name']} - Match Score: {row['match_score']:.1f}%")                 
        elif 6 <= num_top_matches <= 20:
            # Show just the number and score, not individual employers
            st.subheader(f"Top {num_top_matches} Employer Matches (Score: {top_score:.1f}%")
            for i, (_, row) in enumerate(top_matches.iterrows()):
                st.write(f"{i+1}. {row['employer_name']}")
        else:  # More than 20 matches with the same score
            st.subheader("Many potential matches found. Here are the first 20 matches.")
            st.subheader(f"Employers with score: {top_score:.1f}%")
            for i, (index, row) in enumerate(top_matches.head(20).iterrows()):
                st.write(f"{i+1}. {row['employer_name']}")
    else:
        st.write("No matching employers found.")