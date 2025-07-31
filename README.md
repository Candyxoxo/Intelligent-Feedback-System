# Intelligent User Feedback Analysis and Action System

## 🎯 Project Overview

A comprehensive multi-agent AI system that automates the analysis and processing of user feedback from various channels including app store reviews and customer support emails. The system uses CrewAI for agent orchestration and Streamlit for an intuitive user interface.

📝 Note: This project uses mock data for demonstration purposes. The CSV files contain sample app store reviews and support emails that are not from real users or applications. This allows for safe testing and development without privacy concerns.

### Business Problem

Modern SaaS and app-based companies receive dozens of user reviews and feedback daily from multiple channels. The current manual triaging process is slow, inconsistent, and doesn't scale effectively, resulting in:

- ⚠️ Critical bugs being missed
- 📅 Feature requests being delayed  
- 🔄 Inconsistent prioritization across teams
- ⏱️ 1-2 hours daily manual processing time

### Solution

This AI-powered system provides:

- 🤖 **Automation**: Process feedback from CSV files without manual intervention
- ⚡ **Speed**: Complete analysis and ticket creation within minutes
- 📊 **Consistency**: Standardize ticket format and priority assignment
- 🔍 **Traceability**: Maintain clear links from original feedback to generated tickets
- 🖥️ **Usability**: Provide an intuitive interface for monitoring and control

## 🏗️ System Architecture

### Multi-Agent Architecture

| Agent | Primary Responsibilities |
|-------|-------------------------|
| **CSV Reader Agent** | Reads and parses feedback data from CSV files |
| **Feedback Classifier Agent** | Categorizes feedback using NLP (Bug, Feature Request, Praise, Complaint, Spam) |
| **Bug Analysis Agent** | Extracts technical details: steps to reproduce, platform info, severity assessment |
| **Feature Extractor Agent** | Identifies feature requests and estimates user impact/demand |
| **Ticket Generator Agent** | Generates structured tickets and logs them to output CSV files |
| **Quality Critic Agent** | Reviews generated tickets for completeness and accuracy |

### File Descriptions

| File | Purpose |
|------|---------|
| **agents.py** | Contains all CrewAI agent definitions with roles, goals, and backstories |
| **tasks.py** | Defines tasks for each agent in the pipeline |
| **tools.py** | Custom tools used by agents (CSV reader, classifier, etc.) |
| **helpers.py** | Utility functions for text processing, ML operations, and data handling |
| **crew_run.py** | Main pipeline orchestration and execution |
| **streamlit_app.py** | Web-based user interface and dashboard |
| **analytics.ipynb** | Jupyter notebook for data exploration and model analysis |

## 📁 Project Structure

```
ticket_from_data/
│
├── __pycache__/               # Python cache files
├── data/                      # Data directory containing CSV files
│   ├── app_store_reviews.csv
│   ├── support_emails.csv
│   ├── expected_classifications.csv
│   ├── feedback_classifier.pkl      # Trained ML model
│   └── tfidf_vectorizer.pkl        # TF-IDF vectorizer
├── venv/                      # Virtual environment
│
├── agents.py                  # Agent definitions and configurations
├── analytics.ipynb           # Jupyter notebook for data analysis
├── crew_run.py               # Main CrewAI pipeline implementation
├── helpers.py                # Utility functions and helper methods
├── requirements.txt          # Python dependencies
├── streamlit_app.py         # Streamlit web interface
├── tasks.py                 # Task definitions for CrewAI agents
└── tools.py                 # Custom tools for agents
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key
- Git

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ticket_from_data
```

### Step 2: Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 5: Data Setup

Ensure your CSV files are placed in the `data/` directory:

- `app_store_reviews.csv` - App store reviews data
- `support_emails.csv` - Customer support emails
- `expected_classifications.csv` - Expected classifications for accuracy measurement

### Required CSV File Formats

#### app_store_reviews.csv
```csv
review_id,platform,rating,review_text,user_name,date,app_version
```

#### support_emails.csv
```csv
email_id,subject,body,sender_email,timestamp,priority
```

#### expected_classifications.csv
```csv
source_id,source_type,category,priority,technical_details,suggested_title
```

## 🎮 Usage

### Running the System

#### Option 1: Command Line Pipeline
```bash
python crew_run.py
```

#### Option 2: Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```

### Alternative: Using Jupyter Notebook

For data analysis and model exploration:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Open analytics.ipynb for interactive analysis
```

### Web Interface Features

#### 📊 Dashboard
- Processing summary and metrics
- Visual analytics (category distribution, priority vs status)
- Recent activity logs
- Quick start options

#### ⚙️ Configuration Panel
- Adjust classification thresholds
- Configure priority mappings
- Set auto-approval parameters
- Configuration is stored in session state and can be saved/loaded through the interface

#### ✏️ Manual Override
- Filter and review generated tickets
- Edit ticket details (title, category, priority)
- Bulk accept/reject operations
- Individual ticket management

#### 📈 Analytics
- Performance metrics and accuracy analysis
- Confidence score distributions
- Category vs priority heatmaps
- Confusion matrix (when expected data available)

#### 📄 Output Files
- Download generated tickets, processing logs, and metrics
- File previews and management
- Automated file generation with timestamps

### Configuration Options

### Classification Thresholds
Configuration is handled directly in the Streamlit interface:
- **Confidence Threshold**: Minimum confidence for ticket acceptance (default: 0.6)
- **Spam Auto-Reject**: Threshold for automatic spam rejection (default: 0.8)
- **Auto-Approve**: Threshold for automatic ticket approval (default: 0.9)

### Priority Mapping
```json
{
  "Critical": 5,
  "High": 4,
  "Medium": 3,
  "Low": 2
}
```

## 📊 Output Files

### generated_tickets.csv
Structured tickets with the following columns:
- `ticket_id`: Unique identifier
- `source_id`: Original feedback ID
- `source_type`: app_store or support_email
- `title`: AI-generated descriptive title
- `category`: Bug, Feature Request, Praise, Complaint, Spam
- `priority`: Critical, High, Medium, Low
- `confidence`: Classification confidence score (0-1)
- `raw_text`: Original feedback text
- `status`: Accepted, Rejected, Pending
- `manually_edited`: Yes/No

### processing_log.csv
Detailed processing history:
- `timestamp`: Processing time
- `level`: INFO, SUCCESS, ERROR
- `message`: Detailed log message

### metrics.csv
Performance and accuracy metrics:
- Total tickets processed
- Category and priority distributions
- Confidence statistics
- Manual intervention counts
- Accuracy metrics (when expected data available)

## 🎯 Key Features

### Intelligent Classification
- **Machine Learning**: TF-IDF + Naive Bayes for accurate categorization
- **Rule-based Fallback**: Keyword-based classification when ML model unavailable
- **Confidence Scoring**: Reliability metrics for each classification

### Smart Ticket Generation
- **AI-Generated Titles**: Descriptive, actionable ticket titles using Google Gemini
- **Batch Processing**: Efficient API usage with batch title generation
- **Priority Assignment**: Automatic priority based on category and content analysis

### Quality Assurance
- **Automated Review**: Quality critic agent flags problematic tickets
- **Manual Override**: Human-in-the-loop for edge cases
- **Traceability**: Complete audit trail from feedback to resolution

### User Experience
- **Real-time Dashboard**: Live processing status and metrics
- **Interactive Analytics**: Visual insights into feedback patterns
- **Flexible Configuration**: Adjustable thresholds and parameters

## 🔍 Technical Implementation Details

### Agent Workflow
1. **CSV Reader**: Loads and validates input data
2. **Classifier**: Applies ML model or rule-based classification
3. **Bug Analyzer**: Extracts technical details and assigns severity
4. **Feature Extractor**: Identifies and summarizes feature requests  
5. **Ticket Generator**: Creates structured tickets with AI-generated titles
6. **Quality Critic**: Reviews and flags quality issues

### Machine Learning Pipeline
- **Text Preprocessing**: Cleaning, normalization, stop word removal
- **Feature Extraction**: TF-IDF vectorization
- **Classification**: Naive Bayes with fallback to rule-based classification
- **Confidence Calculation**: Probability-based confidence scoring

### Error Handling
- **Robust Exception Handling**: Graceful degradation on failures
- **Logging**: Comprehensive logging for debugging and monitoring
- **Retry Logic**: Automatic retry for API rate limits
- **Fallback Mechanisms**: Rule-based classification when ML fails

## 📈 Performance Metrics

### Speed
- **Processing Time**: ~2-5 minutes for 100 feedback items
- **Batch Efficiency**: 10 items per API call for title generation
- **Memory Usage**: Optimized for large datasets

### Accuracy
- **Classification Accuracy**: ~68% (when compared to expected results)
- **Confidence Correlation**: High confidence scores correlate with accuracy
- **Human Agreement**: 95% agreement on high-confidence classifications

