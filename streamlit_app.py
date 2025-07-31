

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
import json
import os
import sys

# Import your CrewAI pipeline
try:
    from crew_run import crew
    from tools import pipeline_data
except ImportError:
    st.error("Could not import crew_run.py. Please ensure the file is in the same directory.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Feedback Processing System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'tickets_df' not in st.session_state:
    st.session_state.tickets_df = pd.DataFrame()
if 'pipeline_executed' not in st.session_state:
    st.session_state.pipeline_executed = False
if 'execution_logs' not in st.session_state:
    st.session_state.execution_logs = []
if 'manual_edits' not in st.session_state:
    st.session_state.manual_edits = {}
if 'accepted_tickets' not in st.session_state:
    st.session_state.accepted_tickets = set()
if 'rejected_tickets' not in st.session_state:
    st.session_state.rejected_tickets = set()
if 'processing_metrics' not in st.session_state:
    st.session_state.processing_metrics = {}
if 'config_settings' not in st.session_state:
    st.session_state.config_settings = {
        'confidence_threshold': 0.6,
        'bug_priority_mapping': {'Critical': 5, 'High': 4, 'Medium': 3, 'Low': 2},
        'auto_approve_threshold': 0.9,
        'spam_confidence_threshold': 0.8
    }

def log_message(message, level="INFO"):
    """Add message to execution logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    st.session_state.execution_logs.append(log_entry)

def load_expected_classifications():
    """Load expected classifications for accuracy calculation"""
    try:
        expected_df = pd.read_csv("D:\\Agentic_AI\\Capstone\\data\\expected_classifications.csv")
        return expected_df
    except FileNotFoundError:
        st.warning("expected_classifications.csv not found.")
        return None
    except Exception as e:
        st.error(f"Error loading expected classifications: {e}")
        return None

def calculate_processing_metrics():
    """Calculate comprehensive processing metrics"""
    if st.session_state.tickets_df.empty:
        return {}
    
    tickets_df = st.session_state.tickets_df
    expected_df = load_expected_classifications()
    
    metrics = {
        'total_tickets': len(tickets_df),
        'processing_time': datetime.now().isoformat(),
        'category_distribution': tickets_df['category'].value_counts().to_dict(),
        'priority_distribution': tickets_df['priority'].value_counts().to_dict(),
        'confidence_stats': {
            'mean': float(tickets_df['confidence'].mean()),
            'std': float(tickets_df['confidence'].std()),
            'min': float(tickets_df['confidence'].min()),
            'max': float(tickets_df['confidence'].max())
        },
        'manual_interventions': len(st.session_state.manual_edits),
        'accepted_count': len(st.session_state.accepted_tickets),
        'rejected_count': len(st.session_state.rejected_tickets),
        'pending_count': len(tickets_df) - len(st.session_state.accepted_tickets) - len(st.session_state.rejected_tickets)
    }
    
    # Calculate accuracy if expected classifications available
    if expected_df is not None:
        accuracy_data = calculate_accuracy_metrics(tickets_df, expected_df)
        if accuracy_data:
            metrics['accuracy'] = {
                'overall_accuracy': accuracy_data['overall_accuracy'],
                'total_compared': accuracy_data['total_compared'],
                'correct_predictions': accuracy_data['correct_predictions'],
                'category_accuracy': {k: v['accuracy'] for k, v in accuracy_data['category_metrics'].items()}
            }
    
    st.session_state.processing_metrics = metrics
    return metrics

def calculate_accuracy_metrics(tickets_df, expected_df):
    """Calculate accuracy metrics against expected classifications"""
    if expected_df is None or tickets_df.empty:
        return None
    
    try:
        # Merge on source_id
        merged = pd.merge(tickets_df, expected_df, on='source_id', how='inner', suffixes=('_actual', '_expected'))
        
        if merged.empty:
            return None
        
        # Calculate overall accuracy
        correct_predictions = (merged['category_actual'] == merged['category_expected']).sum()
        total_predictions = len(merged)
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate per-category metrics
        categories = merged['category_expected'].unique()
        category_metrics = {}
        
        for category in categories:
            category_data = merged[merged['category_expected'] == category]
            if len(category_data) > 0:
                category_accuracy = (category_data['category_actual'] == category_data['category_expected']).sum() / len(category_data)
                category_metrics[category] = {
                    'accuracy': category_accuracy,
                    'total': len(category_data),
                    'correct': (category_data['category_actual'] == category_data['category_expected']).sum()
                }
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_compared': total_predictions,
            'correct_predictions': correct_predictions,
            'category_metrics': category_metrics,
            'confusion_data': merged
        }
    except Exception as e:
        st.error(f"Error calculating accuracy metrics: {e}")
        return None

def save_output_files():
    """Save all required output files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # 1. generated_tickets.csv
        if not st.session_state.tickets_df.empty:
            output_tickets = st.session_state.tickets_df.copy()
            output_tickets['status'] = output_tickets['ticket_id'].apply(
                lambda x: 'Accepted' if x in st.session_state.accepted_tickets 
                else 'Rejected' if x in st.session_state.rejected_tickets 
                else 'Pending'
            )
            output_tickets['manually_edited'] = output_tickets['ticket_id'].apply(
                lambda x: 'Yes' if x in st.session_state.manual_edits else 'No'
            )
            output_tickets.to_csv(f"generated_tickets_{timestamp}.csv", index=False)
            
        # 2. processing_log.csv
        if st.session_state.execution_logs:
            logs_df = pd.DataFrame(st.session_state.execution_logs)
            logs_df.to_csv(f"processing_log_{timestamp}.csv", index=False)
        
        # 3. metrics.csv
        metrics = calculate_processing_metrics()
        if metrics:
            # Flatten metrics for CSV
            flattened_metrics = []
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flattened_metrics.append({
                            'metric_category': key,
                            'metric_name': sub_key,
                            'metric_value': sub_value
                        })
                else:
                    flattened_metrics.append({
                        'metric_category': 'general',
                        'metric_name': key,
                        'metric_value': value
                    })
            
            metrics_df = pd.DataFrame(flattened_metrics)
            metrics_df.to_csv(f"metrics_{timestamp}.csv", index=False)
        
        log_message(f"All output files saved with timestamp {timestamp}", "SUCCESS")
        return timestamp
        
    except Exception as e:
        log_message(f"Error saving output files: {e}", "ERROR")
        return None

def run_pipeline():
    """Execute the CrewAI pipeline with configuration settings"""
    with st.spinner("Executing AI Feedback Processing Pipeline..."):
        try:
            log_message("Starting AI Feedback Processing Pipeline", "INFO")
            
            # Clear pipeline_data to ensure fresh run
            pipeline_data.clear()
            
            # Execute the crew
            result = crew.kickoff()
            
            log_message("Pipeline crew execution completed", "INFO")
            
            # Get the generated tickets from pipeline_data
            if 'tickets' in pipeline_data and not pipeline_data['tickets'].empty:
                st.session_state.tickets_df = pipeline_data['tickets'].copy()
                
                # Apply configuration filters
                confidence_threshold = st.session_state.config_settings['confidence_threshold']
                spam_threshold = st.session_state.config_settings['spam_confidence_threshold']
                
                # Auto-reject low confidence tickets
                low_confidence_tickets = st.session_state.tickets_df[
                    st.session_state.tickets_df['confidence'] < confidence_threshold
                ]['ticket_id'].tolist()
                
                # Auto-reject spam with high confidence
                spam_tickets = st.session_state.tickets_df[
                    (st.session_state.tickets_df['category'] == 'Spam') & 
                    (st.session_state.tickets_df['confidence'] > spam_threshold)
                ]['ticket_id'].tolist()
                
                st.session_state.rejected_tickets.update(low_confidence_tickets + spam_tickets)
                
                if low_confidence_tickets:
                    log_message(f"Auto-rejected {len(low_confidence_tickets)} low confidence tickets", "INFO")
                if spam_tickets:
                    log_message(f"Auto-rejected {len(spam_tickets)} spam tickets", "INFO")
                
                st.session_state.pipeline_executed = True
                log_message(f"Pipeline completed successfully. Generated {len(st.session_state.tickets_df)} tickets.", "SUCCESS")
                
                # Calculate and save metrics
                calculate_processing_metrics()
                
                # Save output files
                save_output_files()
                
                st.success(f"Pipeline executed successfully! Generated {len(st.session_state.tickets_df)} tickets.")
                
                # Show pipeline results summary
                with st.expander("📋 Pipeline Results Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'reviews' in pipeline_data:
                            st.metric("Reviews Processed", len(pipeline_data['reviews']))
                    with col2:
                        if 'emails' in pipeline_data:
                            st.metric("Emails Processed", len(pipeline_data['emails']))
                    with col3:
                        st.metric("Tickets Generated", len(st.session_state.tickets_df))
                
            else:
                st.error("Pipeline completed but no tickets were generated.")
                log_message("Pipeline completed but no tickets were generated.", "ERROR")
                
                # Show debugging info
                st.write("**Pipeline Data Keys:**", list(pipeline_data.keys()))
                for key, value in pipeline_data.items():
                    if hasattr(value, '__len__'):
                        st.write(f"- {key}: {len(value)} items")
                    else:
                        st.write(f"- {key}: {type(value)}")
                
        except Exception as e:
            st.error(f"Pipeline execution failed: {e}")
            log_message(f"Pipeline execution failed: {e}", "ERROR")
            
            # Show detailed error info
            import traceback
            st.code(traceback.format_exc())

def display_dashboard():
    """Display main dashboard with overview"""
    st.title("AI Feedback Processing Dashboard")
    
    if not st.session_state.pipeline_executed:
        st.info("Welcome! Please run the pipeline first to see the dashboard.")
        
        # Show data file status
        st.subheader("Data Files Status")
        col1, col2 = st.columns(2)
        
        with col1:
            review_path = "D:\\Agentic_AI\\Capstone\\data\\app_store_reviews.csv"
            if os.path.exists(review_path):
                st.success("App Store Reviews - Found")
                try:
                    df = pd.read_csv(review_path)
                    st.write(f"{len(df)} reviews available")
                except:
                    st.warning("File exists but couldn't read")
            else:
                st.error("App Store Reviews - Not Found")
        
        with col2:
            email_path = "D:\\Agentic_AI\\Capstone\\data\\support_emails.csv"
            if os.path.exists(email_path):
                st.success("Support Emails - Found")
                try:
                    df = pd.read_csv(email_path)
                    st.write(f"{len(df)} emails available")
                except:
                    st.warning("File exists but couldn't read")
            else:
                st.error("Support Emails - Not Found")
        
        # Quick start section
        st.subheader("Quick Start")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Pipeline", type="primary", use_container_width=True, key="dashboard_run_pipeline"):
                run_pipeline()
        with col2:
            if st.button("Configure Settings", use_container_width=True, key="dashboard_configure"):
                st.session_state.page = 'Configuration Panel'
                st.rerun()
        return
    
    # Main metrics
    st.subheader("Processing Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Tickets", 
            len(st.session_state.tickets_df),
            help="Total number of tickets generated"
        )
    with col2:
        accepted = len(st.session_state.accepted_tickets)
        total = len(st.session_state.tickets_df)
        acceptance_rate = (accepted / total * 100) if total > 0 else 0
        st.metric(
            "Accepted", 
            accepted,
            f"{acceptance_rate:.1f}%"
        )
    with col3:
        rejected = len(st.session_state.rejected_tickets)
        rejection_rate = (rejected / total * 100) if total > 0 else 0
        st.metric(
            "Rejected", 
            rejected,
            f"{rejection_rate:.1f}%"
        )
    with col4:
        pending = total - accepted - rejected
        st.metric("Pending Review", pending)
    with col5:
        avg_confidence = st.session_state.tickets_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Visual analytics
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_counts = st.session_state.tickets_df['category'].value_counts()
        fig_category = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Ticket Categories",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_category.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Priority vs Status
        status_data = []
        for _, ticket in st.session_state.tickets_df.iterrows():
            ticket_id = ticket['ticket_id']
            if ticket_id in st.session_state.accepted_tickets:
                status = "Accepted"
            elif ticket_id in st.session_state.rejected_tickets:
                status = "Rejected"
            else:
                status = "Pending"
            
            status_data.append({
                'priority': ticket['priority'],
                'status': status,
                'count': 1
            })
        
        status_df = pd.DataFrame(status_data)
        status_summary = status_df.groupby(['priority', 'status']).size().reset_index(name='count')
        
        fig_status = px.bar(
            status_summary,
            x='priority',
            y='count',
            color='status',
            title="Priority vs Status",
            color_discrete_map={'Accepted': 'green', 'Rejected': 'red', 'Pending': 'orange'}
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    if st.session_state.execution_logs:
        recent_logs = st.session_state.execution_logs[-5:]
        for log_entry in reversed(recent_logs):
            timestamp = log_entry['timestamp']
            level = log_entry['level']
            message = log_entry['message']
            
            if level == "ERROR":
                st.error(f"[{timestamp}] {message}")
            elif level == "SUCCESS":
                st.success(f"[{timestamp}] {message}")
            else:
                st.info(f"[{timestamp}] {message}")
    else:
        st.info("No recent activity")

def display_configuration_panel():
    """Display configuration panel for adjusting settings"""
    st.title("Configuration Panel")
    st.markdown("Adjust classification thresholds and processing parameters")
    
    # Classification Thresholds
    st.subheader("Classification Thresholds")
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config_settings['confidence_threshold'],
            step=0.1,
            help="Tickets below this confidence will be auto-rejected"
        )
        
        spam_threshold = st.slider(
            "Spam Auto-Reject Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config_settings['spam_confidence_threshold'],
            step=0.1,
            help="Spam tickets above this confidence will be auto-rejected"
        )
    
    with col2:
        auto_approve_threshold = st.slider(
            "Auto-Approve Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config_settings['auto_approve_threshold'],
            step=0.1,
            help="High-confidence tickets above this threshold can be auto-approved"
        )
    
    # Priority Mapping
    st.subheader("Priority Mapping")
    st.markdown("Adjust priority weights for different severity levels")
    
    priority_mapping = st.session_state.config_settings['bug_priority_mapping'].copy()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        priority_mapping['Critical'] = st.number_input("Critical", value=priority_mapping['Critical'], min_value=1, max_value=10)
    with col2:
        priority_mapping['High'] = st.number_input("High", value=priority_mapping['High'], min_value=1, max_value=10)
    with col3:
        priority_mapping['Medium'] = st.number_input("Medium", value=priority_mapping['Medium'], min_value=1, max_value=10)
    with col4:
        priority_mapping['Low'] = st.number_input("Low", value=priority_mapping['Low'], min_value=1, max_value=10)
    
    # Data File Configuration
    st.subheader("Data File Paths")
    st.info("These paths are currently hardcoded in crew_run.py. Future versions will make them configurable.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("App Store Reviews", value="D:\\Agentic_AI\\Capstone\\data\\app_store_reviews.csv", disabled=True)
    with col2:
        st.text_input("Support Emails", value="D:\\Agentic_AI\\Capstone\\data\\support_emails.csv", disabled=True)
    
    # Save configuration
    if st.button("Save Configuration", type="primary", key="config_save"):
        st.session_state.config_settings.update({
            'confidence_threshold': confidence_threshold,
            'spam_confidence_threshold': spam_threshold,
            'auto_approve_threshold': auto_approve_threshold,
            'bug_priority_mapping': priority_mapping
        })
        
        # Save to file
        with open('config_settings.json', 'w') as f:
            json.dump(st.session_state.config_settings, f, indent=2)
        
        log_message("Configuration settings saved", "SUCCESS")
        st.success("Configuration saved successfully!")
    
    # Load default configuration
    if st.button("Reset to Defaults", key="config_reset"):
        st.session_state.config_settings = {
            'confidence_threshold': 0.6,
            'bug_priority_mapping': {'Critical': 5, 'High': 4, 'Medium': 3, 'Low': 2},
            'auto_approve_threshold': 0.9,
            'spam_confidence_threshold': 0.8
        }
        st.success("Configuration reset to defaults!")
        st.rerun()

def display_manual_override():
    """Display manual override interface for editing and approving tickets"""
    st.title("Manual Override")
    st.markdown("Edit or approve generated tickets")
    
    if st.session_state.tickets_df.empty:
        st.info("No tickets available. Please run the pipeline first.")
        return
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        category_filter = st.multiselect(
            "Category",
            options=st.session_state.tickets_df['category'].unique(),
            default=st.session_state.tickets_df['category'].unique()
        )
    with col2:
        priority_filter = st.multiselect(
            "Priority",
            options=st.session_state.tickets_df['priority'].unique(),
            default=st.session_state.tickets_df['priority'].unique()
        )
    with col3:
        status_filter = st.selectbox(
            "Status",
            options=["All", "Accepted", "Rejected", "Pending"],
            index=0
        )
    with col4:
        confidence_range = st.slider(
            "Confidence Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.1
        )
    
    # Apply filters
    filtered_df = st.session_state.tickets_df[
        (st.session_state.tickets_df['category'].isin(category_filter)) &
        (st.session_state.tickets_df['priority'].isin(priority_filter)) &
        (st.session_state.tickets_df['confidence'] >= confidence_range[0]) &
        (st.session_state.tickets_df['confidence'] <= confidence_range[1])
    ].copy()
    
    # Apply status filter
    if status_filter == "Accepted":
        filtered_df = filtered_df[filtered_df['ticket_id'].isin(st.session_state.accepted_tickets)]
    elif status_filter == "Rejected":
        filtered_df = filtered_df[filtered_df['ticket_id'].isin(st.session_state.rejected_tickets)]
    elif status_filter == "Pending":
        filtered_df = filtered_df[
            ~filtered_df['ticket_id'].isin(st.session_state.accepted_tickets) &
            ~filtered_df['ticket_id'].isin(st.session_state.rejected_tickets)
        ]
    
    # Bulk actions
    st.subheader("Bulk Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Accept All Filtered", key="bulk_accept"):
            ticket_ids = filtered_df['ticket_id'].tolist()
            st.session_state.accepted_tickets.update(ticket_ids)
            st.session_state.rejected_tickets.difference_update(ticket_ids)
            log_message(f"Bulk accepted {len(ticket_ids)} tickets", "INFO")
            st.success(f"Accepted {len(ticket_ids)} tickets")
            st.rerun()
    
    with col2:
        if st.button("Reject All Filtered", key="bulk_reject"):
            ticket_ids = filtered_df['ticket_id'].tolist()
            st.session_state.rejected_tickets.update(ticket_ids)
            st.session_state.accepted_tickets.difference_update(ticket_ids)
            log_message(f"Bulk rejected {len(ticket_ids)} tickets", "INFO")
            st.success(f"Rejected {len(ticket_ids)} tickets")
            st.rerun()
    
    with col3:
        if st.button("Reset All Filtered", key="bulk_reset"):
            ticket_ids = filtered_df['ticket_id'].tolist()
            st.session_state.accepted_tickets.difference_update(ticket_ids)
            st.session_state.rejected_tickets.difference_update(ticket_ids)
            log_message(f"Reset status for {len(ticket_ids)} tickets", "INFO")
            st.success(f"Reset {len(ticket_ids)} tickets")
            st.rerun()
    
    # Individual ticket management
    st.subheader(f"Tickets ({len(filtered_df)} shown)")
    
    for idx, ticket in filtered_df.iterrows():
        ticket_id = ticket['ticket_id']
        
        # Determine status
        if ticket_id in st.session_state.accepted_tickets:
            status = "Accepted"
            status_color = "green"
        elif ticket_id in st.session_state.rejected_tickets:
            status = "Rejected"
            status_color = "red"
        else:
            status = "Pending"
            status_color = "orange"
        
        with st.expander(f"Ticket:{ticket_id} - {ticket['title'][:50]}... ({status})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Editable fields
                edited_title = st.text_input(
                    "Title",
                    value=st.session_state.manual_edits.get(ticket_id, {}).get('title', ticket['title']),
                    key=f"title_{ticket_id}"
                )
                
                edited_category = st.selectbox(
                    "Category",
                    options=["Bug", "Feature Request", "Praise", "Complaint", "Spam"],
                    index=["Bug", "Feature Request", "Praise", "Complaint", "Spam"].index(
                        st.session_state.manual_edits.get(ticket_id, {}).get('category', ticket['category'])
                    ),
                    key=f"category_{ticket_id}"
                )
                
                edited_priority = st.selectbox(
                    "Priority",
                    options=["Critical", "High", "Medium", "Low"],
                    index=["Critical", "High", "Medium", "Low"].index(
                        st.session_state.manual_edits.get(ticket_id, {}).get('priority', ticket['priority'])
                    ) if st.session_state.manual_edits.get(ticket_id, {}).get('priority', ticket['priority']) in ["Critical", "High", "Medium", "Low"] else 2,
                    key=f"priority_{ticket_id}"
                )
                
                st.text_area(
                    "Raw Text (Read-only)",
                    value=ticket['raw_text'],
                    height=100,
                    disabled=True,
                    key=f"raw_text_{ticket_id}"
                )
                
                # Save edits button
                if st.button(f"Save Changes", key=f"save_{ticket_id}"):
                    st.session_state.manual_edits[ticket_id] = {
                        'title': edited_title,
                        'category': edited_category,
                        'priority': edited_priority,
                        'timestamp': datetime.now().isoformat()
                    }
                    # Update the main dataframe
                    st.session_state.tickets_df.loc[
                        st.session_state.tickets_df['ticket_id'] == ticket_id, 
                        ['title', 'category', 'priority']
                    ] = [edited_title, edited_category, edited_priority]
                    
                    log_message(f"Manual edit applied to ticket {ticket_id}", "INFO")
                    st.success("Changes saved!")
                    st.rerun()
            
            with col2:
                st.markdown(f"**Status:** <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
                st.write(f"**Source:** {ticket['source_type']}")
                st.write(f"**Confidence:** {ticket['confidence']:.2f}")
                st.write(f"**Source ID:** {ticket['source_id']}")
                
                if ticket_id in st.session_state.manual_edits:
                    st.write("**Manually Edited**")
                
                # Action buttons
                col_accept, col_reject = st.columns(2)
                with col_accept:
                    if st.button("Accept", key=f"accept_{ticket_id}", disabled=ticket_id in st.session_state.accepted_tickets):
                        st.session_state.accepted_tickets.add(ticket_id)
                        st.session_state.rejected_tickets.discard(ticket_id)
                        log_message(f"Ticket {ticket_id} accepted", "INFO")
                        st.rerun()
                
                with col_reject:
                    if st.button("Reject", key=f"reject_{ticket_id}", disabled=ticket_id in st.session_state.rejected_tickets):
                        st.session_state.rejected_tickets.add(ticket_id)
                        st.session_state.accepted_tickets.discard(ticket_id)
                        log_message(f"Ticket {ticket_id} rejected", "INFO")
                        st.rerun()

def display_analytics():
    """Display comprehensive analytics and performance metrics"""
    st.title("Analytics")
    st.markdown("Processing statistics and performance metrics")
    
    if st.session_state.tickets_df.empty:
        st.info("No data available for analytics. Please run the pipeline first.")
        return
    
    # Load expected classifications for accuracy
    expected_df = load_expected_classifications()
    
    # Performance metrics
    st.subheader("Performance Metrics")
    metrics = calculate_processing_metrics()
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processing Accuracy", 
                     f"{metrics.get('accuracy', {}).get('overall_accuracy', 0):.2%}" if 'accuracy' in metrics else "N/A")
        with col2:
            st.metric("Manual Interventions", metrics.get('manual_interventions', 0))
        with col3:
            st.metric("Avg Confidence", f"{metrics['confidence_stats']['mean']:.2f}")
        with col4:
            st.metric("Processing Time", datetime.now().strftime("%H:%M:%S"))
    
    # Detailed analytics
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig_confidence = px.histogram(
            st.session_state.tickets_df,
            x='confidence',
            nbins=20,
            title="Confidence Score Distribution",
            color_discrete_sequence=['skyblue']
        )
        fig_confidence.add_vline(
            x=st.session_state.config_settings['confidence_threshold'],
            line_dash="dash",
            line_color="red",
            annotation_text="Confidence Threshold"
        )
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    with col2:
        # Category vs Priority heatmap
        category_priority = pd.crosstab(
            st.session_state.tickets_df['category'],
            st.session_state.tickets_df['priority']
        )
        fig_heatmap = px.imshow(
            category_priority.values,
            x=category_priority.columns,
            y=category_priority.index,
            title="Category vs Priority Heatmap",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Accuracy analysis (if available)
    if expected_df is not None:
        st.subheader("Accuracy Analysis")
        accuracy_data = calculate_accuracy_metrics(st.session_state.tickets_df, expected_df)
        
        if accuracy_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Overall accuracy metrics
                st.metric("Overall Accuracy", f"{accuracy_data['overall_accuracy']:.2%}")
                st.metric("Total Compared", accuracy_data['total_compared'])
                st.metric("Correct Predictions", accuracy_data['correct_predictions'])
            
            with col2:
                # Category-wise accuracy
                category_metrics_df = pd.DataFrame.from_dict(accuracy_data['category_metrics'], orient='index')
                fig_accuracy = px.bar(
                    category_metrics_df.reset_index(),
                    x='index',
                    y='accuracy',
                    title="Accuracy by Category",
                    labels={'index': 'Category', 'accuracy': 'Accuracy'},
                    color='accuracy',
                    color_continuous_scale='RdYlGn'
                )
                fig_accuracy.update_layout(showlegend=False)
                st.plotly_chart(fig_accuracy, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            confusion_data = accuracy_data['confusion_data']
            confusion_matrix = pd.crosstab(
                confusion_data['category_actual'], 
                confusion_data['category_expected'],
                margins=True
            )
            st.dataframe(confusion_matrix, use_container_width=True)
    

def display_output_files():
    """Display and manage output files"""
    st.title("Output Files")
    st.markdown("Download generated files and reports")
    
    # File generation
    st.subheader("Generate Output Files")
    if st.button("Generate All Files", type="primary", key="generate_files"):
        timestamp = save_output_files()
        if timestamp:
            st.success(f"All files generated successfully with timestamp: {timestamp}")
        else:
            st.error("Error generating files")
    
    # Individual downloads
    st.subheader(" Download Files")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### generated_tickets.csv")
        
        if not st.session_state.tickets_df.empty:
            download_tickets = st.session_state.tickets_df.copy()
            download_tickets['status'] = download_tickets['ticket_id'].apply(
                lambda x: 'Accepted' if x in st.session_state.accepted_tickets 
                else 'Rejected' if x in st.session_state.rejected_tickets 
                else 'Pending'
            )
            download_tickets['manually_edited'] = download_tickets['ticket_id'].apply(
                lambda x: 'Yes' if x in st.session_state.manual_edits else 'No'
            )
            
            csv_buffer = io.StringIO()
            download_tickets.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download Tickets CSV",
                data=csv_buffer.getvalue(),
                file_name=f"generated_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_tickets"
            )
        else:
            st.info("No tickets available")
    
    with col2:
        st.markdown("### processing_log.csv")
        st.markdown("*Detailed processing history and decisions*")
        
        if st.session_state.execution_logs:
            logs_df = pd.DataFrame(st.session_state.execution_logs)
            
            csv_buffer = io.StringIO()
            logs_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download Processing Log",
                data=csv_buffer.getvalue(),
                file_name=f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_logs"
            )
        else:
            st.info("No logs available")
    
    with col3:
        st.markdown("###  metrics.csv")
        st.markdown("*Performance and accuracy metrics*")
        
        metrics = calculate_processing_metrics()
        if metrics:
            # Flatten metrics for CSV
            flattened_metrics = []
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                flattened_metrics.append({
                                    'metric_category': f"{key}.{sub_key}",
                                    'metric_name': sub_sub_key,
                                    'metric_value': sub_sub_value
                                })
                        else:
                            flattened_metrics.append({
                                'metric_category': key,
                                'metric_name': sub_key,
                                'metric_value': sub_value
                            })
                else:
                    flattened_metrics.append({
                        'metric_category': 'general',
                        'metric_name': key,
                        'metric_value': value
                    })
            
            metrics_df = pd.DataFrame(flattened_metrics)
            
            csv_buffer = io.StringIO()
            metrics_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label=" Download Metrics CSV",
                data=csv_buffer.getvalue(),
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_metrics"
            )
        else:
            st.info("No metrics available")
    
    # File preview
    st.subheader(" File Previews")
    
    tab1, tab2, tab3 = st.tabs([" Tickets", " Logs", " Metrics"])
    
    with tab1:
        if not st.session_state.tickets_df.empty:
            preview_tickets = st.session_state.tickets_df.copy()
            preview_tickets['status'] = preview_tickets['ticket_id'].apply(
                lambda x: 'Accepted' if x in st.session_state.accepted_tickets 
                else 'Rejected' if x in st.session_state.rejected_tickets 
                else 'Pending'
            )
            st.dataframe(preview_tickets, use_container_width=True)
        else:
            st.info("No tickets to preview")
    
    with tab2:
        if st.session_state.execution_logs:
            logs_df = pd.DataFrame(st.session_state.execution_logs)
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No logs to preview")
    
    with tab3:
        metrics = calculate_processing_metrics()
        if metrics:
            # Display metrics in a structured way
            st.json(metrics)
        else:
            st.info("No metrics to preview")

def main():
    # Load configuration if exists
    try:
        with open('config_settings.json', 'r') as f:
            st.session_state.config_settings.update(json.load(f))
    except FileNotFoundError:
        pass
    
    # Initialize page navigation in session state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    
    # Sidebar navigation
    with st.sidebar:
        st.title(" AI Feedback System")
        st.markdown("---")
        
        # Pipeline control
        st.subheader(" Pipeline Control")
        if st.button(" Run Pipeline", type="primary", use_container_width=True, key="sidebar_run_pipeline"):
            run_pipeline()
        
        if st.session_state.pipeline_executed:
            st.success(" Pipeline Active")
            st.metric("Tickets Generated", len(st.session_state.tickets_df))
        
        st.markdown("---")
        
        # Navigation - Changed from radio to selectbox
        st.subheader(" Navigation")
        page_options = [
            "Dashboard",
            "Configuration Panel", 
            "Manual Override",
            "Analytics",
            "Output Files"
        ]
        
        current_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
        
        selected_page = st.selectbox(
            "Select Page",
            page_options,
            index=current_index,
            key="page_selector"
        )
        
        # Update session state when selectbox selection changes
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.pipeline_executed:
            st.subheader(" Quick Stats")
            total = len(st.session_state.tickets_df)
            accepted = len(st.session_state.accepted_tickets)
            rejected = len(st.session_state.rejected_tickets)
            pending = total - accepted - rejected
            
            st.metric("Total", total)
            st.metric("Accepted", accepted, f"{(accepted/total*100):.1f}%" if total > 0 else "0%")
            st.metric("Rejected", rejected, f"{(rejected/total*100):.1f}%" if total > 0 else "0%")
            st.metric("Pending", pending)
        
        # Data source info
        st.markdown("---")
        st.subheader(" Data Sources")
        st.text("Reviews: app_store_reviews.csv")
        st.text("Emails: support_emails.csv")
        st.text("Expected: expected_classifications.csv")
    
    # Main content area
    page = st.session_state.page
    if page == "Dashboard":
        display_dashboard()
    elif page == "Configuration Panel":
        display_configuration_panel()
    elif page == "Manual Override":
        display_manual_override()
    elif page == "Analytics":
        display_analytics()
    elif page == "Output Files":
        display_output_files()

if __name__ == "__main__":
    main()