import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import pandas as pd
import google.generativeai as genai
from datetime import datetime, timedelta
import tempfile
import os
import time
import sqlite3
import hashlib
import email
import imaplib
from email.header import decode_header
from email.utils import parsedate_to_datetime
from pdf2image import convert_from_path
import plotly.express as px

# Configure Tesseract with your specific path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Configure Gemini
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in secrets or environment variables")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini: {str(e)}")
    st.stop()

# --- Database Functions ---
def initialize_database():
    """Initialize the SQLite database for user authentication and expenses"""
    conn = sqlite3.connect('bitebill.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT,
            password TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TEXT,
            amount REAL,
            category TEXT,
            description TEXT,
            source TEXT,
            filename TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS email_configs (
            username TEXT PRIMARY KEY,
            email TEXT,
            app_password TEXT,
            last_checked TIMESTAMP,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(username, name, password, email):
    """Create a new user in the database"""
    conn = sqlite3.connect('bitebill.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute('INSERT INTO users (username, name, password, email) VALUES (?, ?, ?, ?)', 
                 (username, name, hashed_password, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials against the database"""
    conn = sqlite3.connect('bitebill.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
             (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_expense(username, date, amount, category, description, source, filename):
    """Save an expense to the database"""
    conn = sqlite3.connect('bitebill.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO expenses (username, date, amount, category, description, source, filename)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, date, amount, category, description, source, filename))
    conn.commit()
    conn.close()

def get_user_expenses(username):
    """Retrieve all expenses for a user"""
    conn = sqlite3.connect('bitebill.db')
    c = conn.cursor()
    c.execute('SELECT date, amount, category, description, source, filename FROM expenses WHERE username = ? ORDER BY date DESC', (username,))
    expenses = c.fetchall()
    conn.close()
    return expenses

def save_email_config(username, email, app_password):
    """Save email configuration for a user"""
    conn = sqlite3.connect('bitebill.db')
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO email_configs (username, email, app_password, last_checked)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (username, email, app_password))
    conn.commit()
    conn.close()

def get_email_config(username):
    """Retrieve email configuration for a user"""
    conn = sqlite3.connect('bitebill.db')
    c = conn.cursor()
    c.execute('SELECT email, app_password FROM email_configs WHERE username = ?', (username,))
    config = c.fetchone()
    conn.close()
    return config

# --- Enhanced Email Processing Functions ---
def fetch_emails(email_account, password, last_checked=None):
    """Fetch emails containing bills/invoices from Gmail with proper IMAP syntax"""
    def decode_mime_header(header):
        """Decode MIME encoded email headers"""
        parts = []
        for part, encoding in decode_header(header or ""):
            if isinstance(part, bytes):
                parts.append(part.decode(encoding or 'utf-8', errors='replace'))
            else:
                parts.append(str(part))
        return ''.join(parts)

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        st.info("Connecting to Gmail...")
        
        try:
            mail.login(email_account, password)
            st.success("Login successful")
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            st.warning("Make sure you're using an App Password, not your regular Gmail password")
            return []
        
        # Select mailbox with status check
        status, _ = mail.select("INBOX")
        if status != 'OK':
            st.error("Failed to select mailbox")
            return []
        
        st.info("Searching for bills/invoices...")
        
        # Build proper IMAP search criteria
        search_terms = [
            'SUBJECT "bill"',
            'SUBJECT "invoice"',
            'SUBJECT "receipt"',
            'SUBJECT "statement"',
            'SUBJECT "payment"',
            'SUBJECT "transaction"',
            'FROM "no-reply"',
            'FROM "donotreply"'
        ]
        
        if last_checked:
            date_str = last_checked.strftime("%d-%b-%Y")
            # Correct IMAP search syntax with proper OR grouping
            search_criteria = f'(SINCE "{date_str}" (OR {" ".join(search_terms)}))'
        else:
            search_criteria = f'(OR {" ".join(search_terms)})'
        
        st.info(f"Search criteria: {search_criteria}")
        
        try:
            status, data = mail.search(None, search_criteria)
            if status != 'OK':
                st.error("Email search failed")
                return []
                
            email_ids = data[0].split()
            st.info(f"Found {len(email_ids)} potential bills")
            
            bills = []
            for email_id in email_ids:
                try:
                    email_id_str = email_id.decode()
                    status, data = mail.fetch(email_id_str, "(RFC822)")
                    
                    if status != 'OK':
                        st.warning(f"Failed to fetch email {email_id_str}")
                        continue
                    
                    raw_email = data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Process message and attachments...
                    # (rest of your existing processing code)
                    
                except Exception as e:
                    st.error(f"Error processing email {email_id_str}: {str(e)}")
                    continue
            
            return bills
            
        except imaplib.IMAP4.error as e:
            st.error(f"IMAP search error: {str(e)}")
            # Try a simpler search if complex one fails
            simple_search = '(OR SUBJECT "bill" SUBJECT "invoice")'
            if last_checked:
                simple_search = f'(SINCE "{date_str}" {simple_search})'
            st.info(f"Trying simpler search: {simple_search}")
            status, data = mail.search(None, simple_search)
            
            if status == 'OK':
                email_ids = data[0].split()
                st.info(f"Found {len(email_ids)} emails with simpler search")
                # Process these emails...
            else:
                st.error("Simple search also failed")
                return []
    
    finally:
        try:
            mail.logout()
        except:
            pass

def process_email_bills(username, email_account, password):
    """Process bills from email with better error handling and feedback"""
    try:
        # Get last checked time from database
        last_checked = datetime.now() - timedelta(days=180)  # Check last 6 months by default
        config = get_email_config(username)
        if config and config[1] == password:
            conn = sqlite3.connect('bitebill.db')
            c = conn.cursor()
            c.execute('SELECT last_checked FROM email_configs WHERE username = ?', (username,))
            last_checked_result = c.fetchone()
            if last_checked_result:
                last_checked = datetime.strptime(last_checked_result[0], "%Y-%m-%d %H:%M:%S")
            conn.close()
        
        st.info(f"Checking emails since {last_checked.strftime('%Y-%m-%d')}")
        bills = fetch_emails(email_account, password, last_checked)
        
        if not bills:
            st.warning("No bills found. Try these troubleshooting steps:")
            st.markdown("""
            1. Make sure you're using an App Password (not your regular password)
            2. Check that IMAP is enabled in Gmail settings
            3. Verify you have bills in your inbox with attachments
            4. Try a wider date range by modifying the last_checked parameter
            """)
            return 0, "No new bills found in your email"
        
        processed_count = 0
        processing_results = []
        
        with st.expander("Processing Details", expanded=True):
            progress_bar = st.progress(0)
            total_bills = len(bills)
            
            for i, bill in enumerate(bills):
                try:
                    st.write(f"Processing {bill['filename']}...")
                    
                    # Process based on file type
                    if bill["content_type"] == "application/pdf":
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(bill["data"])
                            images = convert_from_path(tmp.name)
                            text = ""
                            for img in images:
                                text += process_image(img)
                            os.unlink(tmp.name)
                    elif "image/" in bill["content_type"]:
                        image = Image.open(io.BytesIO(bill["data"]))
                        text = process_image(image)
                    elif "text/html" in bill["content_type"]:
                        # Try to extract text from HTML emails
                        text = bill["data"].decode('utf-8', errors='ignore')
                    else:
                        st.warning(f"Skipping unsupported file type: {bill['content_type']}")
                        continue
                    
                    # Extract information
                    expense_data = extract_information(text)
                    if not expense_data.get('amount'):
                        st.warning("No amount found in this bill")
                        continue
                    if not expense_data.get('date'):
                        st.warning("No date found in this bill, using email date")
                        expense_data['date'] = bill['date']
                    
                    # Save to database
                    save_expense(
                        username=username,
                        date=expense_data['date'],
                        amount=expense_data['amount'],
                        category=expense_data['category'],
                        description=f"From email: {bill['subject']}",
                        source=f"Email: {bill['from']}",
                        filename=bill["filename"]
                    )
                    
                    # Generate analysis
                    analysis = analyze_with_gemini(expense_data, st.session_state.financial_context)
                    
                    processing_results.append({
                        "filename": bill['filename'],
                        "extracted_data": expense_data,
                        "analysis": analysis
                    })
                    
                    processed_count += 1
                    progress_bar.progress((i + 1) / total_bills)
                    st.success(f"Processed {bill['filename']}")
                
                except Exception as e:
                    st.error(f"Failed to process {bill['filename']}: {str(e)}")
                    processing_results.append({
                        "filename": bill['filename'],
                        "error": str(e)
                    })
                    continue
        
        # Update last checked time
        save_email_config(username, email_account, password)
        
        # Store processing results in session state
        st.session_state.processing_results = processing_results
        
        return processed_count, f"Processed {processed_count} of {len(bills)} bills"
    
    except Exception as e:
        st.error(f"Error processing emails: {str(e)}")
        return 0, f"Error processing emails: {str(e)}"

# --- Image Processing Functions ---
def preprocess_image(image):
    """Enhance image for better OCR results"""
    image = image.convert('L')  # Convert to grayscale
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    return image

def process_image(image):
    """Extract text from image using OCR"""
    image = preprocess_image(image)
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_oem_psm_config)
    return text

# --- Information Extraction ---
def extract_information(text):
    """Extract key information from bill text with comprehensive regex patterns"""
    # Amount extraction patterns
    amount_patterns = [
        r'(?:Total|Subtotal|Grand\s*Total|Amount|Bill\s*Amount|TOTAL\s*\(?[A-Z]{3}\)?|TOTAL\s*DUE|Total\s*Invoice\s*Amount)\s*[:=-]?\s*[₹$£€]?\s*([\d,]+(?:\.\d{2})?)',
        r'[₹$£€]\s*([\d,]+(?:\.\d{2})?)',
        r'\b(?:RS?|INR)\s*\.?\s*([\d,]+(?:\.\d{2})?)',
        r'([\d,]+(?:\.\d{2})?)\s*(?:RS?|INR|USD|EUR|GBP)',
        r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b',
        r'\(([\d,]+(?:\.\d{2})?)\)',
        r'(?:Total|TOTAL)\b\D*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'\b(\d{1,3}(?:,\d{3})+)',
        r'([\d,]+(?:\.\d{2})?)\s*(?:only|/-)',
        r'\b(\d+\.\d{2})\b',
        r'\b(\d+)\b(?!\s*%)'
    ]
    
    amount = None
    for pattern in amount_patterns:
        amount_match = re.search(pattern, text, re.IGNORECASE | re.VERBOSE)
        if amount_match:
            try:
                amount_str = amount_match.group(1).replace(",", "")
                if float(amount_str) > 0:
                    amount = float(amount_str)
                    break
            except (ValueError, AttributeError):
                continue

    # Date extraction patterns
    date_patterns = [
        r'\b(\d{1,2}/\d{1,2}/\d{2,4})(?:\s+\d{1,2}:\d{2}:\d{2})?\b',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
        r'\d{4}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}',
        r'\d{4}[-/]\d{2}[-/]\d{2}',
        r'\d{2}[-/]\d{2}[-/]\d{4}',
        r'\d{1,2}[./]\d{1,2}[./]\d{4}',
        r'\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*-\d{4}',
        r'\d{2}-\w{3}-\d{4}',
        r'\d{1,2}-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-\d{2,4}',
        r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
        r'\d{1,2}(?:st|nd|rd|th)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
        r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}'
    ]
    
    date = None
    for pattern in date_patterns:
        date_match = re.search(pattern, text, re.IGNORECASE)
        if date_match:
            try:
                date_str = date_match.group(1) if date_match.groups() else date_match.group()
                for fmt in (
                    "%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y",
                    "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", 
                    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
                    "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%y", "%d.%m.%Y",
                    "%Y %b %d", "%dth %b %Y", "%a, %b %d, %Y"
                ):
                    try:
                        date = datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                if date:
                    break
            except Exception:
                continue

    # Category extraction
    category_keywords = {
        "Shopping": [r'\bwalmart\b', r'\bmall\b', r'\bamazon\b', r'\bflipkart\b'],
        "Medical": [r'\bhospital\b', r'\bclinic\b', r'\bpharmacy\b', r'\bmedicine\b'],
        "Entertainment": [r'\bmovie\b', r'\bcinema\b', r'\bnetflix\b', r'\bconcert\b'],
        "Fuel": [r'\bpetrol\b', r'\bdiesel\b', r'\bfuel\b', r'\bgas\s*station\b'],
        "Food": [r'\brestaurant\b', r'\bfood\b', r'\bbill\b', r'\bdine\b'],
        "Travel": [r'\bhotel\b', r'\bflight\b', r'\bairlines\b', r'\btrain\b'],
        "Utilities": [r'\belectricity\b', r'\bwater\b', r'\bbilldesk\b', r'\bbroadband\b'],
        "Education": [r'\bschool\b', r'\bcollege\b', r'\buniversity\b', r'\beducation\b']
    }
    
    category = "Uncategorized"
    for cat, patterns in category_keywords.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                category = cat
                break
        if category != "Uncategorized":
            break

    return {
        "amount": amount,
        "date": date,
        "category": category,
        "raw_text": text
    }

# --- Gemini Analysis Functions ---
def analyze_with_gemini(expense_data, financial_context=None):
    """Get spending insights from Gemini for individual bills"""
    if not expense_data.get('amount') or not expense_data.get('category'):
        return "Cannot analyze - missing amount or category"
    
    prompt = f"""
    Analyze this expense record and provide personalized financial advice:
    
    Expense Details:
    - Amount: {expense_data.get('amount', 'N/A')}
    - Date: {expense_data.get('date', 'N/A')}
    - Category: {expense_data.get('category', 'N/A')}
    - Context from bill: {expense_data.get('raw_text', '')[:500]}... [truncated]
    
    Financial Context:
    {financial_context if financial_context else "No additional financial context provided"}
    
    Please provide:
    1. A brief analysis of this expense (1-2 sentences)
    2. Suggestions for better spending in this category (2-3 bullet points)
    3. Any potential savings opportunities (1-2 suggestions)
    4. General financial advice related to this expense (1-2 sentences)
    
    Keep the response under 200 words total.
    """
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            return "Received empty response from Gemini"
        return response.text
    except Exception as e:
        error_msg = f"Gemini analysis failed: {str(e)}"
        if "404" in str(e):
            error_msg += "\n\nNote: Please check the latest Gemini documentation for available models."
        return error_msg

def get_comprehensive_analysis(expenses_df, financial_context):
    """Get overall spending analysis based on all bills"""
    if expenses_df.empty:
        return "No expenses to analyze"
    
    # Prepare summary data
    total_spent = expenses_df['amount'].sum()
    avg_monthly = expenses_df.groupby(expenses_df['date'].str[:7])['amount'].sum().mean()
    category_spending = expenses_df.groupby('category')['amount'].sum().sort_values(ascending=False)
    top_category = category_spending.index[0] if not category_spending.empty else "N/A"
    top_amount = category_spending.iloc[0] if not category_spending.empty else 0
    
    prompt = f"""
    Analyze this comprehensive spending data and provide personalized financial advice:
    
    Spending Overview:
    - Total tracked expenses: ${total_spent:,.2f}
    - Average monthly spending: ${avg_monthly:,.2f}
    - Number of transactions: {len(expenses_df)}
    - Top spending category: {top_category} (${top_amount:,.2f})
    
    Spending by Category:
    {category_spending.to_string()}
    
    Financial Context:
    {financial_context}
    
    Please provide:
    1. Overall spending health assessment (1 paragraph)
    2. Top 3 spending categories needing attention
    3. Specific recommendations for each problematic category
    4. General money-saving strategies
    5. Suggested budget adjustments
    
    Format the response with clear headings and bullet points.
    Keep the response under 400 words.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate comprehensive analysis: {str(e)}"

# --- Visualization Functions ---
def create_category_visualizations(categories, amounts):
    """Create interactive category spending charts"""
    fig = px.pie(
        names=categories,
        values=amounts,
        title="Spending by Category",
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_monthly_trend_chart(expenses_df):
    """Create visualization of monthly spending trend"""
    if expenses_df.empty:
        return None
    
    monthly_spending = expenses_df.groupby(expenses_df['date'].str[:7])['amount'].sum().reset_index()
    monthly_spending.columns = ['month', 'amount']
    
    fig = px.bar(
        monthly_spending,
        x='month',
        y='amount',
        title='Monthly Spending Trend',
        labels={'amount': 'Amount ($)', 'month': 'Month'},
        color='amount',
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# --- UI Components ---
def display_chat_messages():
    """Display chat messages in the Financial Assistant tab"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def financial_assistant_chat_input():
    """Handle the chat input separately from the messages display"""
    # Initialize if not exists
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Get user input
    prompt = st.chat_input("Ask about your finances...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    expenses = get_user_expenses(st.session_state["username"])
                    expenses_df = pd.DataFrame(expenses, columns=["date", "amount", "category", "description", "source", "filename"])
                    
                    full_prompt = f"""
                    Financial Context:
                    {st.session_state.financial_context}
                    
                    Expense History:
                    {expenses_df.to_string() if not expenses_df.empty else "No expense history"}
                    
                    User Question: {prompt}
                    
                    Please provide a detailed, helpful response with specific advice.
                    """
                    
                    response = model.generate_content(full_prompt)
                    if response.text:
                        st.markdown(response.text)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
                    else:
                        error_msg = "Sorry, I couldn't generate a response. Please try again."
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

def file_upload_section():
    """Section for manual file uploads with robust error handling"""
    st.subheader("Upload Your Bills")
    uploaded_files = st.file_uploader(
        "Choose image or PDF files", 
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files and st.button("Analyze All Bills"):
        with st.spinner(f"Processing {len(uploaded_files)} bills..."):
            results = []
            for file in uploaded_files:
                try:
                    if file.type.startswith('image/'):
                        image = Image.open(io.BytesIO(file.read()))
                        text = process_image(image)
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            images = convert_from_path(tmp.name)
                            text = ""
                            for img in images:
                                text += process_image(img)
                            os.unlink(tmp.name)
                    
                    expense_data = extract_information(text)
                    expense_data['filename'] = file.name
                    
                    analysis = analyze_with_gemini(expense_data, st.session_state.financial_context)
                    
                    if expense_data.get('amount') and expense_data.get('date'):
                        save_expense(
                            username=st.session_state["username"],
                            date=expense_data['date'],
                            amount=expense_data['amount'],
                            category=expense_data['category'],
                            description=f"Uploaded file: {file.name}",
                            source="Manual Upload",
                            filename=file.name
                        )
                    
                    results.append({
                        "filename": file.name,
                        "extracted_data": expense_data,
                        "analysis": analysis
                    })
                
                except Exception as e:
                    results.append({
                        "filename": file.name,
                        "error": f"Error processing file: {str(e)}"
                    })
            
            st.session_state.processing_results = results
            st.success(f"Processed {len([r for r in results if 'error' not in r])} bills successfully!")
            st.rerun()

def email_automation_section():
    """Section for email automation"""
    st.subheader("Email Automation")
    
    with st.expander("Gmail Configuration", expanded=True):
        email_account = st.text_input("Gmail Address", key="gmail_address")
        app_password = st.text_input("App Password", type="password", key="app_password")
        
        if st.button("Save Configuration"):
            if email_account and app_password:
                save_email_config(st.session_state["username"], email_account, app_password)
                st.success("Email configuration saved!")
            else:
                st.error("Please enter both email and app password")
    
    if st.button("Process Emails Now"):
        config = get_email_config(st.session_state["username"])
        if config:
            with st.spinner("Checking for new bills..."):
                count, message = process_email_bills(st.session_state["username"], config[0], config[1])
                st.success(message)
                st.rerun()
        else:
            st.error("Please configure your email first")

def financial_profile_section():
    """Section for financial profile"""
    st.sidebar.header("Financial Profile")
    
    monthly_income = st.sidebar.number_input(
        "Monthly Income ($)", 
        min_value=0, 
        value=5000, 
        key='monthly_income'
    )
    
    savings = st.sidebar.number_input(
        "Current Savings ($)", 
        min_value=0, 
        value=15000, 
        key='savings'
    )
    
    financial_goals = st.sidebar.text_area(
        "Financial Goals", 
        "Save for vacation ($3,000)\nEmergency fund ($10,000)", 
        key='financial_goals'
    )
    
    st.session_state.financial_context = f"""
    Monthly income: ${monthly_income:,}
    Current savings: ${savings:,}
    Financial goals: {financial_goals}
    """
    
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.rerun()

def expense_history_section():
    """Section for displaying expense history"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Expense History")
    
    expenses = get_user_expenses(st.session_state["username"])
    if expenses:
        expenses_df = pd.DataFrame(
            expenses, 
            columns=["date", "amount", "category", "description", "source", "filename"]
        )
        
        total_spent = expenses_df['amount'].sum()
        st.sidebar.metric("Total Tracked Expenses", f"${total_spent:,.2f}")
        
        with st.sidebar.expander("View All Expenses"):
            st.dataframe(expenses_df, hide_index=True)
    else:
        st.sidebar.info("No expenses tracked yet")

def analysis_section():
    """Section for comprehensive analysis"""
    st.subheader("Comprehensive Analysis")
    
    expenses = get_user_expenses(st.session_state["username"])
    if not expenses:
        st.info("Process some bills first to see analysis")
        return
    
    expenses_df = pd.DataFrame(
        expenses, 
        columns=["date", "amount", "category", "description", "source", "filename"]
    )
    
    if st.button("Generate Analysis"):
        with st.spinner("Analyzing your spending..."):
            analysis = get_comprehensive_analysis(expenses_df, st.session_state.financial_context)
            st.session_state.comprehensive_analysis = analysis
    
    if 'comprehensive_analysis' in st.session_state:
        st.markdown(st.session_state.comprehensive_analysis)
    
    st.subheader("Visualizations")
    
    # Category spending pie chart
    category_spending = expenses_df.groupby('category')['amount'].sum()
    if not category_spending.empty:
        fig = create_category_visualizations(category_spending.index, category_spending.values)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for category visualization")
    
    # Monthly trend chart
    monthly_trend = create_monthly_trend_chart(expenses_df)
    if monthly_trend:
        st.plotly_chart(monthly_trend, use_container_width=True)
    else:
        st.warning("Not enough data for monthly trend")

# --- Authentication Pages ---
def login_page():
    """Display login and registration forms"""
    st.title("💰 Bite Bill - Expense Tracker")
    
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        with st.form("login_form"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if verify_user(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['financial_context'] = ""
                    st.session_state['chat_messages'] = []
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with register_tab:
        with st.form("register_form"):
            st.subheader("Create New Account")
            new_username = st.text_input("Choose a username")
            new_name = st.text_input("Full name")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if create_user(new_username, new_name, new_password, new_email):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")

# --- Main Dashboard ---
def dashboard():
    """Main dashboard that shows after successful login"""
    # Initialize all session state variables
    if 'financial_context' not in st.session_state:
        st.session_state.financial_context = ""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []
    if 'comprehensive_analysis' not in st.session_state:
        st.session_state.comprehensive_analysis = ""
    
    # Custom CSS
    st.markdown("""
    <style>
        .header {
            font-size: 36px !important;
            font-weight: bold;
            color: #4a4a4a;
            margin-bottom: 20px;
        }
        .subheader {
            font-size: 20px !important;
            color: #6c6c6c;
            margin-bottom: 10px;
        }
        .analysis-box {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="header">💰 Bite Bill</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subheader">Welcome back, {st.session_state["username"]}!</div>', unsafe_allow_html=True)
    
    # Setup sidebar
    financial_profile_section()
    expense_history_section()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload Bills", 
        "Analysis", 
        "Financial Assistant", 
        "Email Automation"
    ])
    
    with tab1:
        file_upload_section()
        
        if 'processing_results' in st.session_state:
            st.subheader("Processing Results")
            for result in st.session_state.processing_results:
                with st.expander(f"Results for {result['filename']}", expanded=False):
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        expense_data = result['extracted_data']
                        
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Amount", f"${expense_data['amount']:,.2f}" if expense_data['amount'] else "Not detected")
                        with cols[1]:
                            st.metric("Date", expense_data['date'] if expense_data['date'] else "Not detected")
                        with cols[2]:
                            st.metric("Category", expense_data['category'])
                        
                        st.markdown("**AI Analysis**")
                        st.markdown(result['analysis'])
    
    with tab2:
        analysis_section()
    
    with tab3:
        st.subheader("Financial Assistant")
        display_chat_messages()
    
    with tab4:
        email_automation_section()
    
    # Move the chat input outside of the tabs at the bottom of the page
    financial_assistant_chat_input()

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Bite Bill - Smart Expense Tracker", 
        page_icon="💰", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    initialize_database()
    
    # Check authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        login_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()