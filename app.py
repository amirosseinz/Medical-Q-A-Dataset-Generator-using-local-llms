import os
import xml.etree.ElementTree as ET
import pandas as pd
import re
import PyPDF2
from Bio import Entrez
import time
import json
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import threading
from datetime import datetime, timedelta


app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
SESSION_RECORDS_FOLDER = 'session_records' # New folder for session metadata
ALLOWED_EXTENSIONS = {'pdf', 'xml'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SESSION_RECORDS_FOLDER, exist_ok=True)


job_progress = {}
job_results = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#Dataset Generation Class
class DatasetGenerator:
    def __init__(self, job_id, config):
        self.job_id = job_id
        self.config = config
        self.start_time = datetime.now()
        self.update_progress("Initializing...", 0)
        # Ensure Entrez email is set
        Entrez.email = self.config.get('email', 'your.email@example.com') 

    def update_progress(self, message, percentage):
        """Updates the progress dictionary for a given job ID."""
        job_progress[self.job_id] = {
            'message': message,
            'percentage': percentage,
            'timestamp': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat()
        }
        logger.info(f"Job {self.job_id}: {message} ({percentage}%)")

    def extract_medquad_all(self, xml_files, medical_terms):
        """Extract Q&A pairs from MedQuAD XML files based on medical terms.
        Adds 'source' and 'quality_status' to each pair.
        """
        self.update_progress("Processing MedQuAD XML files...", 10)
        qa_pairs = []

        # Convert medical terms to lowercase for matching
        keywords = [term.lower().strip() for term in medical_terms if term.strip()]

        xml_count = 0
        total_qas_checked = 0

        for xml_file in xml_files:
            xml_count += 1
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for doc in root.findall(".//document"):
                    title_elem = doc.find("full_title")
                    title = title_elem.text if title_elem is not None and title_elem.text else ""

                    qa_sections = doc.findall(".//qa_pairs/qa_pair") or doc.findall(".//qa_pair")
                    for pair in qa_sections:
                        total_qas_checked += 1
                        question_elem = pair.find("question")
                        answer_elem = pair.find("answer")
                        question = question_elem.text if question_elem is not None and question_elem.text else ""
                        answer = answer_elem.text if answer_elem is not None and answer_elem.text else ""

                        if not question or not answer:
                            continue

                        combined_text = f"{title} {question} {answer}".lower()
                        if any(kw in combined_text for kw in keywords):
                            qa_pairs.append({
                                "source": "MedQuAD",
                                "file": os.path.basename(xml_file),
                                "question": question.strip(),
                                "answer": answer.strip(),
                                "quality_status": "Verified - Human Labeled" # MedQuAD is human-curated
                            })
            except Exception as e:
                logger.error(f"Error parsing {xml_file}: {e}")

        logger.info(f"XML files scanned: {xml_count}, found {len(qa_pairs)} relevant MedQuAD pairs.")
        return qa_pairs

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text() or ''
                        text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} in {pdf_path}: {e}")
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
        return text

    def fetch_pubmed_abstracts(self, medical_terms, retmax=1000, batch_size=100, sleep_time=0.5):
        """Fetch abstracts from PubMed based on medical terms."""
        self.update_progress("Fetching PubMed data...", 40)

        term_queries = []
        for term in medical_terms:
            if term.strip():
                # Escape special characters and add quotes for exact matching
                escaped_term = term.strip().replace('"', '\\"')
                term_queries.append(f'"{escaped_term}"[Title/Abstract]')

        if not term_queries:
            logger.info("No medical terms for PubMed search.")
            return []

        query = "(" + " OR ".join(term_queries) + ")"

        abstracts = []
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]

            logger.info(f"Found {len(id_list)} PubMed IDs for query: '{query}'")

            if id_list:
                batches = [id_list[i:i+batch_size] for i in range(0, len(id_list), batch_size)]

                for i, batch in enumerate(batches):
                    # Update progress for PubMed fetching, scaling from 40% to 50%
                    progress = 40 + (i / len(batches)) * 10
                    self.update_progress(f"Fetching PubMed batch {i+1}/{len(batches)}", progress)

                    try:
                        handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="abstract", retmode="text")
                        batch_abstracts = handle.read()
                        handle.close()

                        # Split and clean abstracts
                        if batch_abstracts:
                            split_abstracts = re.split(r'\n\n(?:PMID:\s*\d+\s*)?\d+\.\s*', batch_abstracts)
                            abstracts.extend([abs.strip() for abs in split_abstracts if abs.strip()])

                        time.sleep(sleep_time)

                    except Exception as e:
                        logger.error(f"Error fetching batch of PubMed abstracts: {e}")
                        time.sleep(sleep_time * 2) # Longer sleep on error

        except Exception as e:
            logger.error(f"Error in PubMed search: {e}")
        return abstracts

    def clean_text(self, text):
        """Clean and prepare text for Q&A generation."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common PDF artifacts or citation numbers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(cid:\d+\)', '', text) # Common from PDF text extraction
        text = re.sub(r'-\s*\n', '', text) # Hyphenated words split across lines
        text = re.sub(r'\n+', '\n', text).strip() # Reduce multiple newlines
        return text

    def split_into_chunks(self, text, chunk_size=500, overlap=50):
        """Split text into chunks with optional overlap."""
        if not text:
            return []

        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = ' '.join(words[i : i + chunk_size])
            if len(chunk.split()) > 20: # Only include chunks with sufficient content
                chunks.append(chunk)
            i += (chunk_size - overlap)
        return chunks

    def generate_qa_with_ollama(self, text_chunk, ollama_url, model_name, medical_domain, original_source_type, original_file_identifier=None):
        """Generate a question-answer pair from a text chunk using Ollama."""
        prompt = f"""You are a medical domain expert and NLP assistant trained to generate high-quality, clinically relevant Question-Answer (QA) pairs from medical literature.

Your task:
Given the following medical text related to "{medical_domain}", generate exactly **ONE** insightful clinical question and its precise, evidence-based answer, using only the information explicitly stated in the text.

TEXT:
{text_chunk}

Instructions:
- Formulate a question that reflects a **real-world clinical concern** (e.g., diagnosis, treatment, adverse effects, risk factors, mechanisms, prognosis, etc.).
- Focus on medically significant details — avoid vague or trivial questions.
- Use professional, unambiguous medical terminology suitable for a clinician audience.
- Ensure the answer is **strictly derived from the text**, without introducing outside knowledge or assumptions.
- Be clear, concise, and accurate. Do not speculate.
- Avoid yes/no or overly simplistic questions — prioritize questions that require explanation or elaboration.

Output Format (exactly as shown):
Question: [Write one high-quality, clinically meaningful question]
Answer: [Write a complete, well-supported answer based only on the given text]

Generate only one QA pair. Do not include any introductory or closing remarks.
"""

        try:
            # Append /api/generate if not already present
            full_ollama_url = ollama_url.rstrip('/')
            if not full_ollama_url.endswith('/api/generate'):
                full_ollama_url = f"{full_ollama_url}/api/generate"

            response = requests.post(
                full_ollama_url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=180 
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

             
                question_match = re.search(r'(?:Question|Q):\s*(.*?)(?=\s*(?:Answer|A):|$)', response_text, re.IGNORECASE | re.DOTALL)
                answer_match = re.search(r'(?:Answer|A):\s*(.*?)$', response_text, re.IGNORECASE | re.DOTALL)

                if question_match and answer_match:
                    question = question_match.group(1).strip()
                    answer = answer_match.group(1).strip()
                    question = re.sub(r'\s+', ' ', question).strip()
                    answer = re.sub(r'\s+', ' ', answer).strip()

                    # Basic length validation
                    if len(question) > 15 and len(answer) > 30:
                        return {
                            "question": question,
                            "answer": answer,
                            "ollama_model_used": model_name, # Return model used
                            "original_chunk_source": original_source_type, # Return original source type (PDF/PubMed)
                            "original_file_identifier": original_file_identifier # Return original filename for PDF
                        }

            logger.warning(f"Ollama response could not be parsed or was too short for chunk from {original_source_type}: {response_text[:200]}...")
            return None

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama URL '{ollama_url}': {e}. Is Ollama running and accessible?")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout from Ollama URL '{ollama_url}'. Ollama might be overloaded or the response is too slow.")
            return None
        except Exception as e:
            logger.error(f"Error generating QA with Ollama: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
            return None

    def process_chunks_with_ollama(self, chunks_with_sources, ollama_url, model_name, medical_domain, max_workers=3, max_pairs=None):
        """Process text chunks in parallel to generate QA pairs."""
        qa_pairs = []
        completed = 0

        # Limit chunks if max_pairs is set
        chunks_to_process = chunks_with_sources
        if max_pairs:
            target_chunks_for_ollama = min(len(chunks_with_sources), int(max_pairs * 1.5) if max_pairs > 0 else len(chunks_with_sources))
            chunks_to_process = chunks_with_sources[:target_chunks_for_ollama]
            logger.info(f"Targeting {target_chunks_for_ollama} chunks for Ollama to potentially generate {max_pairs} pairs.")


        total_chunks = len(chunks_to_process)
        if total_chunks == 0:
            self.update_progress("No chunks to process for AI generation.", 90)
            return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.generate_qa_with_ollama, chunk_data[0], ollama_url, model_name, medical_domain, chunk_data[1], chunk_data[2]): chunk_data
                for chunk_data in chunks_to_process
            }

            for future in as_completed(future_to_chunk):
                # Check if we have already generated enough pairs
                if max_pairs and len(qa_pairs) >= max_pairs:
                    future.cancel() 
                    continue

                result = future.result()
                completed += 1

                # Update progress for AI generation, scaling from 60% to 90%
                progress = 60 + (completed / total_chunks) * 30
                self.update_progress(f"Generating AI Q&A pairs: {completed}/{total_chunks} chunks processed", progress)

                if result and isinstance(result, dict) and 'question' in result and 'answer' in result:
                    ollama_model = result.get('ollama_model_used', 'Unknown Model')
                    original_source = result.get('original_chunk_source', 'Unknown Source')
                    original_file = result.get('original_file_identifier')

                    ai_source = f"Ollama ({ollama_model}) from {original_source}"
                    if original_file: # Add filename for PDF sources
                        ai_source += f": {original_file}"

                    qa_pairs.append({
                        "source": ai_source,
                        "text_chunk_source_snippet": future_to_chunk[future][0][:100] + "...", # Original chunk for debugging
                        "question": result['question'],
                        "answer": result['answer'],
                        "quality_status": "AI-Generated - Needs Review" # Mark as needs review, human review is still essential
                    })

        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs from Ollama.")
        return qa_pairs

    def evaluate_qa_pair(self, pair):
        """
        More robust heuristic to evaluate if a QA pair is decent quality.
        IMPORTANT: This is a HEURISTIC and does NOT replace human medical review.
        """
        if not pair or not isinstance(pair, dict):
            return False

        question = pair.get('question', '').strip()
        answer = pair.get('answer', '').strip()
        source = pair.get('source', '')

        # Basic length checks
        if len(question) < 15 or len(answer) < 30: 
            return False

        # Check if question looks like a question (more comprehensive list)
        question_starters = ['what', 'how', 'why', 'when', 'where', 'which', 'who',
                              'can', 'do', 'is', 'are', 'does', 'will', 'should',
                              'explain', 'describe', 'define', 'what is', 'how to']
        if not (question.endswith('?') or any(question.lower().startswith(word) for word in question_starters)):
            return False

        # Avoid generic or clearly bad AI answers (simple keyword checks)
        bad_keywords = ["I cannot generate", "I am a language model", "I don't have enough information", "Based on the text provided"] 
        if any(kw in answer for kw in bad_keywords):
            return False

        # For AI-generated, ensure some content from the original chunk is reflected (simple check)
        if "Ollama" in source and "text_chunk_source_snippet" in pair:
            chunk_snippet = pair["text_chunk_source_snippet"].lower()
            if not any(word in answer.lower() for word in chunk_snippet.split()[:5]): # Check first few words
                 pass # Allow it, but flag in quality_status if we had more detailed checks

        return True

    def generate_dataset(self, config):
        """Main function to generate the medical dataset."""
        try:
            self.update_progress("Starting dataset generation...", 5)

            medical_terms = [term.strip() for term in config.get('medical_terms', '').split(',') if term.strip()]
            medical_domain = config.get('medical_domain', 'medical conditions')

            if not medical_terms:
                raise ValueError("No medical terms provided. Please provide at least one term.")

            #Process XML files (MedQuAD)
            medquad_pairs = []
            if config.get('xml_files'):
                self.update_progress("Extracting data from XML files (MedQuAD)...", 10)
                medquad_pairs = self.extract_medquad_all(config['xml_files'], medical_terms)
                logger.info(f"Extracted {len(medquad_pairs)} relevant MedQuAD pairs.")

            #Process PDF files
            all_pdf_text_chunks_with_filenames = [] # Will store (chunk_text, filename)
            if config.get('pdf_files'):
                self.update_progress("Extracting text from PDF files...", 20)
                for i, pdf_file_path in enumerate(config['pdf_files']): 
                    if os.path.exists(pdf_file_path):
                        text = self.extract_text_from_pdf(pdf_file_path)
                        if text:
                            # Split PDF text into chunks and associate with original filename
                            pdf_chunks_from_this_file = self.split_into_chunks(self.clean_text(text), config.get('chunk_size', 500))
                            all_pdf_text_chunks_with_filenames.extend([(chunk, os.path.basename(pdf_file_path)) for chunk in pdf_chunks_from_this_file])
                    self.update_progress(f"Extracted text from PDF {i+1}/{len(config['pdf_files'])}", 20 + (i / len(config['pdf_files'])) * 10)
                logger.info(f"Total PDF text chunks prepared: {len(all_pdf_text_chunks_with_filenames)}.")
                # Removed deletion of PDFs after processing as per user request

            #Fetch PubMed abstracts
            pubmed_chunks_with_sources = [] # Will store (chunk_text, 'PubMed', None)
            if config.get('use_pubmed', True) and medical_terms:
                pubmed_abstracts = self.fetch_pubmed_abstracts(
                    medical_terms,
                    retmax=config.get('pubmed_retmax', 1000)
                )
                logger.info(f"Fetched {len(pubmed_abstracts)} PubMed abstracts.")
                
                pubmed_text = "\n\n".join(pubmed_abstracts)
                pubmed_chunks = self.split_into_chunks(self.clean_text(pubmed_text), config.get('chunk_size', 500))
                pubmed_chunks_with_sources = [(chunk, 'PubMed', None) for chunk in pubmed_chunks] # No specific file for PubMed
                logger.info(f"Total PubMed chunks prepared: {len(pubmed_chunks_with_sources)}.")


            #Prepare text chunks for AI Q&A generation
            self.update_progress("Cleaning and chunking text for AI generation...", 50)
            
            # Combine all chunks with their source information: (chunk_text, source_type, original_file_identifier)
            # original_file_identifier is the basename for PDF, and None for PubMed
            chunks_with_sources = [(chunk, 'PDF', filename) for chunk, filename in all_pdf_text_chunks_with_filenames] + \
                                  pubmed_chunks_with_sources

            #Combine and shuffle chunks for better diversity in AI generation
            import random
            random.shuffle(chunks_with_sources)
            logger.info(f"Total {len(chunks_with_sources)} chunks prepared for AI generation.")

            #Generate QA pairs using AI (Ollama)
            generated_pairs = []
            #Calculate remaining pairs needed after MedQuAD
            remaining_pairs_needed = max(0, config.get('target_pairs', 1000) - len(medquad_pairs))

            if remaining_pairs_needed > 0 and chunks_with_sources and config.get('use_ollama', True):
                self.update_progress("Initiating AI Q&A generation with Ollama...", 60)
                generated_pairs = self.process_chunks_with_ollama(
                    chunks_with_sources,
                    config.get('ollama_url', 'http://localhost:11434'), # Ensure base URL is passed
                    config.get('ollama_model', 'llama3'),
                    medical_domain,
                    max_workers=config.get('max_workers', 3),
                    max_pairs=remaining_pairs_needed
                )
                logger.info(f"Generated {len(generated_pairs)} AI QA pairs.")
            elif remaining_pairs_needed <= 0:
                logger.info("Target pair count already met or exceeded by MedQuAD data. Skipping AI generation.")
            else:
                logger.info("AI generation skipped: no chunks or use_ollama is false.")


            #combine and filter all pairs
            self.update_progress("Finalizing and filtering dataset...", 90)
            all_pairs = medquad_pairs + generated_pairs
            filtered_pairs = [pair for pair in all_pairs if self.evaluate_qa_pair(pair)]
            logger.info(f"After initial heuristic quality filtering: {len(filtered_pairs)} QA pairs.")

            if not filtered_pairs:
                raise ValueError("No valid QA pairs generated. Please check your input data, medical terms, and Ollama configuration.")

            # Create final DataFrame for deduplication
            df = pd.DataFrame(filtered_pairs)

            # Remove duplicates based on question and answer to ensure uniqueness
            df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
            logger.info(f"After deduplication: {len(df)} final QA pairs.")

            #reate output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{medical_domain.replace(' ', '_').replace('/', '_')}_{timestamp}"
            csv_file_name = f"{base_filename}_training.csv"
            csv_path = os.path.join(OUTPUT_FOLDER, csv_file_name)
            df_training = df.copy()
            df_training["text"] = "question: " + df_training["question"] + " answer: " + df_training["answer"]
            df_training[["text"]].to_csv(csv_path, index=False, header=False, encoding='utf-8')

            # JSON format (more detailed, includes source and quality status)
            json_file_name = f"{base_filename}_full.json"
            json_path = os.path.join(OUTPUT_FOLDER, json_file_name)
            json_data = df[["question", "answer", "source", "quality_status"]].to_dict('records')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # Metadata file (full details including original file/chunk if applicable)
            metadata_file_name = f"{base_filename}_metadata.csv"
            metadata_path = os.path.join(OUTPUT_FOLDER, metadata_file_name)
            metadata_columns = ["question", "answer", "source", "quality_status"]
            # Add text_chunk_source_snippet only if it exists in the DataFrame (from AI-generated)
            if 'text_chunk_source_snippet' in df.columns:
                metadata_columns.append('text_chunk_source_snippet')
            df[metadata_columns].to_csv(metadata_path, index=False, encoding='utf-8')

            self.update_progress("Dataset generation completed!", 100)

            result = {
                'success': True,
                'csv_file': csv_file_name, # Just the filename
                'json_file': json_file_name, # Just the filename
                'metadata_file': metadata_file_name, # Just the filename
                'total_pairs': len(df),
                'sources': df['source'].value_counts().to_dict(),
                'quality_breakdown': df['quality_status'].value_counts().to_dict()
            }

            # Store result for later retrieval
            job_results[self.job_id] = result
            logger.info(f"Job {self.job_id} completed successfully. Results: {result}")

            #Save session record 
            session_record = {
                'job_id': self.job_id,
                'timestamp': datetime.now().isoformat(),
                'config': config, # Store the full config, including input file paths
                'output_files': {
                    'csv': csv_file_name,
                    'json': json_file_name,
                    'metadata': metadata_file_name
                },
                'summary_results': result # Summary results for quick overview
            }
            session_record_path = os.path.join(SESSION_RECORDS_FOLDER, f"{self.job_id}_session.json")
            with open(session_record_path, 'w', encoding='utf-8') as f:
                json.dump(session_record, f, indent=2, ensure_ascii=False)
            logger.info(f"Session record saved to: {session_record_path}")

            return result

        except Exception as e:
            logger.exception(f"Error in dataset generation for job {self.job_id}: {e}")
            self.update_progress(f"Error: {str(e)}", -1) # -1 indicates error state
            return {'success': False, 'error': str(e)}
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles file uploads (PDF and XML)."""
    uploaded_file_paths = {'pdf_files': [], 'xml_files': []}
    logger.info("Receiving file uploads...")

    for file_type in ['pdf_files', 'xml_files']:
        if file_type in request.files:
            files = request.files.getlist(file_type)
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    name, ext = os.path.splitext(filename)
                    unique_filename = f"{name}_{uuid.uuid4().hex}{ext}"
                    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                    try:
                        file.save(file_path)
                        uploaded_file_paths[file_type].append(file_path)
                        logger.info(f"Uploaded: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to save file {filename}: {e}")
                        return jsonify({'error': f"Failed to save file {filename}: {e}"}), 500
                elif file.filename:
                    logger.warning(f"File {file.filename} has disallowed extension.")
    return jsonify(uploaded_file_paths) # Return actual file paths on server

@app.route('/generate', methods=['POST'])
def generate_dataset_endpoint():
    """Starts the dataset generation process in a background thread."""
    try:
        config = request.json
        job_id = str(uuid.uuid4())
        logger.info(f"Received request to generate dataset. Job ID: {job_id}, Config: {config}")

        # Store initial progress
        job_progress[job_id] = {
            'message': 'Job submitted, starting...',
            'percentage': 0,
            'timestamp': datetime.now().isoformat(),
            'start_time': datetime.now().isoformat()
        }

        # Start dataset generation in background
        generator = DatasetGenerator(job_id, config)

        def run_generation():
            result = generator.generate_dataset(config)
            job_results[job_id] = result

        thread = threading.Thread(target=run_generation)
        thread.daemon = True # Allow the main program to exit even if thread is running
        thread.start()

        return jsonify({'job_id': job_id})

    except Exception as e:
        logger.exception(f"Error starting generation endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Returns the current progress and estimated time for a given job."""
    progress = job_progress.get(job_id, None)

    if progress is None:
        return jsonify({'message': 'Job not found', 'percentage': 0, 'status': 'not_found'}), 404

    current_percentage = progress.get('percentage', 0)
    start_time_str = progress.get('start_time')
    current_time_str = progress.get('timestamp')

    estimated_time_remaining_seconds = None
    if start_time_str and current_percentage > 0 and current_percentage < 100:
        try:
            start_time = datetime.fromisoformat(start_time_str)
            current_time = datetime.fromisoformat(current_time_str)
            time_elapsed = (current_time - start_time).total_seconds()

            # Calculate total estimated time based on current progress
            total_estimated_time = time_elapsed / (current_percentage / 100)
            estimated_time_remaining_seconds = max(0, total_estimated_time - time_elapsed)
        except ValueError as e:
            logger.warning(f"Could not calculate estimated time for job {job_id}: {e}")
            estimated_time_remaining_seconds = None

    response_data = {
        'message': progress.get('message', 'Processing...'),
        'percentage': current_percentage,
        'timestamp': current_time_str,
        'estimated_time_remaining_seconds': estimated_time_remaining_seconds
    }

    # If job is completed or in error, include final results
    if current_percentage == 100:
        response_data['status'] = 'completed'
        response_data['results'] = job_results.get(job_id)
    elif current_percentage == -1: # Error state
        response_data['status'] = 'error'
        response_data['results'] = job_results.get(job_id) # Should contain error message
    else:
        response_data['status'] = 'in_progress'

    return jsonify(response_data)

@app.route('/download/<filename>')
def download_file(filename):
    """Allows downloading generated dataset files."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            logger.info(f"Downloading file: {file_path}")
            return send_file(file_path, as_attachment=True)
        else:
            logger.warning(f"Download file not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.exception(f"Error downloading file {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test_ollama', methods=['POST'])
def test_ollama():
    """Tests connectivity to the Ollama server and lists available models."""
    try:
        config = request.json
        ollama_url = config.get('ollama_url', 'http://localhost:11434') # Default for local Ollama

        # Ensur always target the /api/tags endpoint for testing connection and getting models
        base_url = ollama_url.rstrip('/')
        tags_url = f"{base_url}/api/tags"
        
        logger.info(f"Testing Ollama connection to: {tags_url}")

        # Test connection with longer timeout for initial connection
        test_response = requests.get(tags_url, timeout=15) # Increased timeout

        if test_response.status_code == 200:
            data = test_response.json()
            models = data.get('models', [])
            model_names = [model.get('name', 'Unknown') for model in models]
            logger.info(f"Ollama connection successful. Found models: {model_names}")
            return jsonify({
                'success': True,
                'models': model_names,
                'message': f'Connected successfully! Found {len(model_names)} models.'
            })
        else:
            logger.error(f"Ollama connection failed with status: {test_response.status_code}. Response: {test_response.text}")
            return jsonify({
                'success': False,
                'error': f'HTTP {test_response.status_code}: {test_response.text}'
            })

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to Ollama: {e}. Ensure Ollama is running and accessible at the specified URL.")
        return jsonify({
            'success': False,
            'error': 'Cannot connect to Ollama. Make sure Ollama is running and accessible at the provided URL.'
        })
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout connecting to Ollama: {e}. Ollama might be slow to respond or busy.")
        return jsonify({
            'success': False,
            'error': 'Connection timeout. Ollama might be slow to respond. Try increasing the timeout in the code or check Ollama server load.'
        })
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from Ollama: {e}. Response text: {test_response.text if 'test_response' in locals() else 'N/A'}")
        return jsonify({
            'success': False,
            'error': 'Invalid response from Ollama. It might not be serving a valid API.'
        })
    except Exception as e:
        logger.exception(f"Unexpected error testing Ollama: {e}")
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.3.0' 
    })

if __name__ == '__main__':
    print("🏥 Medical Q&A Dataset Generator")
    print("=" * 50)
    print("Starting web server...")
    print("📝 Web interface will be available at: http://localhost:1377")
    print("📋 Make sure you have:")
    print("   - Ollama running (if using AI generation) at the specified URL.")
    print("   - Valid email for PubMed API (e.g., in config).")
    print("   - PDF/XML files ready for upload.")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=1377, threaded=True) # Ensure threaded for background jobs
