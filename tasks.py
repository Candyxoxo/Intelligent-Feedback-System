from crewai import Task 
from agents import (
    csv_reader_agent,
    feedback_classifier_agent,
    bug_analysis_agent,
    feature_extractor_agent,
    ticket_generator_agent,
    quality_critic_agent
)
from tools import (
    csv_reader_tool,
    feedback_classifier_tool,
    bug_analysis_tool,
    feature_extractor_tool,
    ticket_generator_tool,
    quality_critic_tool
)

csv_reader_task = Task(
    description="Load user feedback data from CSV files. Read both app store reviews and support emails, clean the data, and prepare it for classification.",
    expected_output="A confirmation message with the number of reviews and emails loaded successfully.",
    agent=csv_reader_agent,
    tools=[csv_reader_tool]
)

feedback_classifier_task = Task(
    description="Classify all loaded feedback into these categories: Bug, Feature Request, Praise, Complaint, or Spam. Assign confidence scores to each classification.",
    expected_output="A summary of classification results showing the count of each category for both reviews and emails.",
    agent=feedback_classifier_agent,
    tools=[feedback_classifier_tool]
)

bug_analysis_task = Task(
    description="Analyze all feedback classified as 'Bug' category. Extract technical details like device type, crash information, and login issues. Assign severity ratings (Critical, High, Medium, Low).",
    expected_output="Analysis results showing total number of bugs found and their severity distribution.",
    agent=bug_analysis_agent,
    tools=[bug_analysis_tool]
)

feature_extractor_task = Task(
    description="Extract and summarize feature requests from classified feedback. Look for both direct feature requests and feature-related comments in bug reports.",
    expected_output="Summary of feature extraction showing the total number of feature requests identified.",
    agent=feature_extractor_agent,
    tools=[feature_extractor_tool]
)

ticket_generator_task = Task(
    description="Generate structured support and development tickets from all processed feedback (bugs and features)such that each ticket should have a unique ID, title, category, priority, and source information.",
    expected_output="Confirmation of successful ticket generation with the total count of tickets created.",
    agent=ticket_generator_agent,
    tools=[ticket_generator_tool]
)


quality_critic_task = Task(
    description="Review all generated tickets for quality issues. Flag tickets with low confidence scores, unclear titles, or missing information. Generate a processing log of flagged items.",
    expected_output="Quality review results showing number of tickets flagged and reasons for flagging.",
    agent=quality_critic_agent,
    tools=[quality_critic_tool]
)