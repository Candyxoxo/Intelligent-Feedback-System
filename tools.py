import numpy as np             
import pandas as pd      
from crewai.tools import tool
from helpers import load_ml_models, preprocess_text, extract_feature_summary, extract_technical_details, estimate_severity, classify_feedback_ml, generate_titles_batch
import uuid

pipeline_data = {}
@tool("csv_reader")
def csv_reader_tool(instruction: str) -> str:
    """Tool to read and clean CSV files containing user feedback."""
    global pipeline_data
    
    review_path = "D:\\Agentic_AI\\Capstone\\data\\app_store_reviews.csv"
    email_path = "D:\\Agentic_AI\\Capstone\\data\\support_emails.csv"

    try:
        print(f"Reading CSV files from: {review_path} and {email_path}")
        reviews = pd.read_csv(review_path)
        emails = pd.read_csv(email_path)
        
        print("Loading the model")
        # load model
        load_ml_models()

        # Fill missing priorities
        emails['priority'] = emails['priority'].apply(
            lambda x: np.random.choice(['Medium', 'Low']) if pd.isna(x) else x
        )

        # Store in pipeline_data
        pipeline_data['reviews'] = reviews
        pipeline_data['emails'] = emails
        
        return f"Successfully loaded {len(reviews)} reviews and {len(emails)} emails. Data is ready for classification."

    except FileNotFoundError as e:
        return f"CSV files not found: {e}. Please check file paths."
    except Exception as e:
        return f"Error reading CSV files: {e}"

@tool("classifier")
def feedback_classifier_tool(instruction: str) -> str:
    """Tool to classify user feedback into categories: Bug, Feature Request, Praise, Complaint, Spam."""
    global pipeline_data
    
    if 'reviews' not in pipeline_data or 'emails' not in pipeline_data:
        return "Error: No data available for classification. Please run CSV reader first."
    
    try:
        print("Starting feedback classification...")
        classified_reviews = classify_feedback_ml(pipeline_data['reviews'], 'review')
        classified_emails = classify_feedback_ml(pipeline_data['emails'], 'email')
        
        pipeline_data['classified_reviews'] = classified_reviews
        pipeline_data['classified_emails'] = classified_emails
        
        # Count classifications
        review_counts = classified_reviews['category'].value_counts().to_dict()
        email_counts = classified_emails['category'].value_counts().to_dict()
        
        return f"Classification completed successfully!\nReview categories: {review_counts}\nEmail categories: {email_counts}"
        
    except Exception as e:
        return f"Error during classification: {e}"

@tool("bug_analyzer")
def bug_analysis_tool(instruction: str) -> str:
    """Analyze bug-related feedback and assign severity ratings."""
    global pipeline_data
    
    if 'classified_reviews' not in pipeline_data or 'classified_emails' not in pipeline_data:
        return "Error: No classified data available. Please run classification first."
    
    try:
        print("Starting bug analysis...")
        
        # Filter for Bug category
        bug_reviews = pipeline_data['classified_reviews'][
            pipeline_data['classified_reviews']['category'] == 'Bug'
        ].copy()
        bug_emails = pipeline_data['classified_emails'][
            pipeline_data['classified_emails']['category'] == 'Bug'
        ].copy()

        # Extract technical info and assign severity
        if not bug_reviews.empty:
            bug_reviews['technical_details'] = bug_reviews['feedback_text'].apply(extract_technical_details)
            bug_reviews['severity'] = bug_reviews['technical_details'].apply(estimate_severity)

        if not bug_emails.empty:
            bug_emails['technical_details'] = bug_emails['feedback_text'].apply(extract_technical_details)
            bug_emails['severity'] = bug_emails['technical_details'].apply(estimate_severity)

        pipeline_data['bug_reviews'] = bug_reviews
        pipeline_data['bug_emails'] = bug_emails
        
        total_bugs = len(bug_reviews) + len(bug_emails)
        if total_bugs > 0:
            severity_counts = pd.concat([bug_reviews['severity'] if not bug_reviews.empty else pd.Series(), 
                                       bug_emails['severity'] if not bug_emails.empty else pd.Series()]).value_counts().to_dict()
        else:
            severity_counts = {}
        
        return f"Bug analysis completed! Total bugs found: {total_bugs}\nSeverity distribution: {severity_counts}"
        
    except Exception as e:
        return f"Error during bug analysis: {e}"

@tool("feature_extractor")
def feature_extractor_tool(instruction: str) -> str:
    """Extract and summarize feature requests from user feedback."""
    global pipeline_data
    
    if 'classified_reviews' not in pipeline_data or 'classified_emails' not in pipeline_data:
        return "Error: No classified data available. Please run classification first."
    
    try:
        print("Starting feature extraction...")
        
        df_reviews = pipeline_data['classified_reviews']
        df_emails = pipeline_data['classified_emails']
        
        # Direct feature requests
        feat_reviews_direct = df_reviews[df_reviews['category'] == 'Feature Request'].copy()
        feat_emails_direct = df_emails[df_emails['category'] == 'Feature Request'].copy()
        
        # Feature-related bugs
        feat_reviews_bug_related = df_reviews[
            (df_reviews['category'] == 'Bug') &
            (df_reviews['feedback_text'].str.contains("feature", case=False, na=False))
        ].copy()
        feat_emails_bug_related = df_emails[
            (df_emails['category'] == 'Bug') &
            (df_emails['feedback_text'].str.contains("feature", case=False, na=False))
        ].copy()
        
        # Combine and deduplicate
        feat_reviews = pd.concat([feat_reviews_direct, feat_reviews_bug_related], ignore_index=True).drop_duplicates()
        feat_emails = pd.concat([feat_emails_direct, feat_emails_bug_related], ignore_index=True).drop_duplicates()
        
        # Extract summaries
        if not feat_reviews.empty:
            feat_reviews['feature_summary'] = feat_reviews['review_text'].apply(extract_feature_summary)
        if not feat_emails.empty:
            feat_emails['feature_summary'] = feat_emails['body'].apply(extract_feature_summary)
        
        pipeline_data['feature_reviews'] = feat_reviews
        pipeline_data['feature_emails'] = feat_emails
        
        total_features = len(feat_reviews) + len(feat_emails)
        return f"Feature extraction completed! Total feature requests found: {total_features}"
        
    except Exception as e:
        return f"Error during feature extraction: {e}"
    
@tool("ticket_generator")
def ticket_generator_tool(instruction: str) -> str:
    """Generate structured support and development tickets from processed feedback."""
    global pipeline_data
    
    try:
        print("Starting ticket generation...")
        
        # Collect all processed feedback
        all_dfs = []
        for key in ['bug_reviews', 'bug_emails', 'feature_reviews', 'feature_emails']:
            if key in pipeline_data and not pipeline_data[key].empty:
                all_dfs.append(pipeline_data[key])
        
        if not all_dfs:
            return "No processed feedback available for ticket generation. Please run bug analysis and feature extraction first."
        
        all_feedback = pd.concat(all_dfs, ignore_index=True)
        
        # Extract all feedback texts
        feedback_texts = []
        for _, row in all_feedback.iterrows():
            feedback_texts.append(str(row.get("feedback_text", "")))
        
        print(f"Generating titles for {len(feedback_texts)} feedbacks using batch processing...")
        
        # Generate ALL titles at once using batch processing
        all_titles = generate_titles_batch(feedback_texts, batch_size=10)
        
        # Create tickets
        tickets_list = []
        for i, (_, row) in enumerate(all_feedback.iterrows()):
            
            # Get the generated title (with fallback)
            title = all_titles[i] if i < len(all_titles) else str(row.get("feedback_text", ""))[:40]
            
            # Get priority
            priority = "Medium"  # Default 
            
            if row.get("category") == "Bug" and pd.notna(row.get("severity")):
                priority = str(row.get("severity"))
            elif pd.notna(row.get("priority")):
                priority = str(row.get("priority"))
            else:
                if row.get("category") == "Bug":
                    priority = "High"
                elif row.get("category") == "Feature Request":
                    priority = "Medium"
            
            tickets_list.append({
                "ticket_id": str(uuid.uuid4())[:8],
                "source_id": row.get("source_id", "Unknown"),
                "source_type": row.get("source_type", "Unknown"),
                "title": title,  
                "category": row.get("category", "Unknown"),
                "priority": priority,
                "confidence": row.get("confidence", 0.5),
                "raw_text": str(row.get("feedback_text", ""))
            })
        
        tickets_df = pd.DataFrame(tickets_list)
        
        # Fill any missing values
        tickets_df = tickets_df.fillna({
            'ticket_id': 'Unknown',
            'source_id': 'Unknown', 
            'source_type': 'Unknown',
            'title': 'User Feedback',
            'category': 'Unknown',
            'priority': 'Medium',
            'confidence': 0.5,
            'raw_text': 'No content available'
        })
        
        # Save to CSV
        tickets_df.to_csv("generated_tickets.csv", index=False)
        pipeline_data['tickets'] = tickets_df
        
        return f"Successfully generated {len(tickets_df)} tickets using batch processing! Tickets saved to 'generated_tickets.csv'"
        
    except Exception as e:
        return f"Error during ticket generation: {e}"
@tool("quality_critic")
def quality_critic_tool(instruction: str) -> str:
    """Review generated tickets for quality issues and flag problematic entries."""
    global pipeline_data
    
    if 'tickets' not in pipeline_data:
        return "Error: No tickets available for quality review. Please run ticket generation first."
    
    try:
        print("Starting quality review...")
        
        tickets_df = pipeline_data['tickets']
        
        # Flag low confidence tickets
        flagged = tickets_df[tickets_df['confidence'] < 0.6].copy()
        flagged['flag_reason'] = "Low confidence score (< 0.6)"
        
        title_issues = tickets_df[tickets_df['title'].str.len() < 10]
        if not title_issues.empty:
            title_flagged = title_issues.copy()
            title_flagged['flag_reason'] = "Title too short or unclear"
            flagged = pd.concat([flagged, title_flagged], ignore_index=True).drop_duplicates(subset=['ticket_id'])
        
        # Save flagged tickets
        if not flagged.empty:
            flagged.to_csv("processing_log.csv", index=False)
            pipeline_data['flagged'] = flagged
            
            flag_reasons = flagged['flag_reason'].value_counts().to_dict()
            return f"Quality review completed. {len(flagged)} tickets flagged for review.\nFlag reasons: {flag_reasons}\nFlagged tickets saved to 'processing_log.csv'"
        else:
            return "Quality review completed. All tickets passed quality checks - no tickets flagged!"
            
    except Exception as e:
        return f"Error during quality review: {e}"