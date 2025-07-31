from crewai import Crew
from agents import (
    csv_reader_agent,
    feedback_classifier_agent,
    bug_analysis_agent,
    feature_extractor_agent,
    ticket_generator_agent,
    quality_critic_agent
)
from tasks import (
    csv_reader_task,
    feedback_classifier_task,
    bug_analysis_task,
    feature_extractor_task,
    ticket_generator_task,
    quality_critic_task
)
from tools import pipeline_data
crew = Crew(
    agents=[
        csv_reader_agent,
        feedback_classifier_agent,
        bug_analysis_agent,
        feature_extractor_agent,
        ticket_generator_agent,
        quality_critic_agent
    ],
    tasks=[
        csv_reader_task,
        feedback_classifier_task,
        bug_analysis_task,
        feature_extractor_task,
        ticket_generator_task,
        quality_critic_task
    ],
    verbose=True
)

if __name__ == "__main__":
    try:
        print("Starting CrewAI Feedback Processing Pipeline...")
        result = crew.kickoff()
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*60)
        print(result)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        if 'reviews' in pipeline_data:
            print(f"Reviews loaded: {len(pipeline_data['reviews'])}")
        if 'emails' in pipeline_data:
            print(f"Emails loaded: {len(pipeline_data['emails'])}")
        if 'tickets' in pipeline_data:
            print(f"Tickets generated: {len(pipeline_data['tickets'])}")
            print(f"Tickets saved to: generated_tickets.csv")
        if 'flagged' in pipeline_data:
            print(f"Tickets flagged for review: {len(pipeline_data['flagged'])}")
            print(f"Processing log saved to: processing_log.csv")
        else:
            print("All tickets passed quality review")
            
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()