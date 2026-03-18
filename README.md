# LLM Judge — Clinical NLP Evaluation API

A FastAPI application that serves as a reference-free LLM judge for evaluating clinical NLP pipeline output. The judge evaluates whether pipeline-generated medical terms are clinically correct by reading only the clinical note and predicted terms.

## Features

- **Reference-free evaluation**: No gold standard required at runtime
- **Multi-judge support**: Aggregate scores from multiple LLM judges
- **Multiple provider support**: AWS Bedrock (Claude, Nova), Azure OpenAI (GPT-5), Ollama (local models)
- **Granular scoring**: Per-term evaluation across 4 dimensions (clinical correctness, completeness, specificity, component coverage)
- **Diabetes-specific rules**: Built-in term construction rules for diabetes-related terms
- **Gold standard validation**: Compute P/R/F1 to validate judge performance
- **Structured logging**: Comprehensive logging across all services
- **HTML test client**: Standalone UI for easy API testing with hierarchical tree output

## Architecture

```
llm-judge/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Config loader
│   ├── routers/                # API endpoints
│   ├── services/               # Business logic
│   ├── providers/              # LLM provider implementations
│   ├── schemas/                # Pydantic models
│   └── utils/                  # Helper functions
├── config/
│   └── config.yaml             # Application configuration
├── prompts/
│   ├── prompt_a_single.txt     # Single-shot prompt
│   └── prompt_b_cot.txt        # Chain-of-thought prompt
├── data/
│   ├── gold_standard/          # Gold standard annotations
│   └── pipeline_output/        # Pipeline outputs
├── requirements.txt
├── Dockerfile
├── test-client.html            # HTML test client UI
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- AWS credentials (for Bedrock)
- Azure OpenAI credentials (for GPT-5)
- Ollama (optional, for local models)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-judge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. Update `config/config.yaml` with your settings.

### Environment Variables

Create a `.env` file with the following variables:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## Running the Application

### Local Development

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t llm-judge .
docker run -p 8000:8000 --env-file .env llm-judge
```

### Access the API

- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Test client**: Open `test-client.html` in your browser

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status and available judges.

### List Models

```bash
GET /models
```

Returns all configured models and their status.

### Single Note Evaluation

```bash
POST /evaluate
```

Request body:
```json
{
  "pipeline_output": {
    "note_id": "note_4",
    "docname": "Note 4",
    "api_response": {
      "raw_text": "Clinical note text...",
      "normalized_terms": [...]
    }
  },
  "judges": ["claude-3.7-sonnet", "nova-premier"],
  "prompt_template": "prompt_a"
}
```

### Batch Evaluation

```bash
POST /batch_evaluate
```

Evaluate multiple notes at once.

### Gold Standard Evaluation

```bash
POST /gold_evaluate
```

Run judge on gold standard notes and compute P/R/F1 metrics.

## Test Client (HTML UI)

A standalone HTML test client is provided for easy API testing without external tools:

```bash
# Open test-client.html in your browser
# (Make sure FastAPI server is running first)
open test-client.html
```

**Features:**
- 🎯 **Single-page app** with tabs for each endpoint
- 📝 **Raw JSON input** via textareas
- 🌳 **Hierarchical tree display** for readable output
- ✅ **Color-coded verdicts** (PASS/FAIL)
- 📊 **Detailed breakdowns**:
  - Scores per dimension (aggregated + per-judge)
  - Failed dimensions highlighted
  - Justifications per judge per metric
  - Suggested corrections with judge attribution
- 🔧 **Configurable API URL** (defaults to localhost:8000)
- 📈 **Aggregate statistics** for batch evaluations
- 🎓 **P/R/F1 metrics** for gold standard validation

**Example Output:**
```
📄 Note: note_4 | Verdict: FAIL ✗
├─ Term 1: Improving Unspecified diabetes mellitus
│  ├─ Default Lexical: Diabetes mellitus
│  ├─ ICD-10 Codes: E11.9
│  ├─ SNOMED Codes: 73211009
│  ├─ Verdict: FAIL ✗
│  ├─ Scores:
│  │  ├─ Clinical Correctness: 0.90 ✓
│  │  ├─ Completeness: 0.50 ✗
│  │  └─ ...
│  ├─ Justifications:
│  │  └─ qwen3:1.7b: "Missing hypoglycemia..."
│  └─ Suggested Corrections:
│     └─ [qwen3:1.7b] Missing components
│        ├─ Current: Diabetes mellitus
│        └─ Suggested: Uncontrolled diabetes with hypoglycemia
└─ Summary: 0/2 terms passed
```

This makes it easy to:
- Test API endpoints visually
- Review evaluation results in readable format
- Compare different judges and prompts
- Debug pipeline output issues

## Configuration

Edit `config/config.yaml` to configure:

- **Judge models**: Enable/disable specific models
- **Thresholds**: Per-dimension score thresholds
- **Azure OpenAI**: Endpoint and credentials
- **Metrics**: Evaluation dimensions
- **Gold standard paths**: Location of gold files

## Scoring Dimensions

Each term is evaluated on 4 dimensions (0.0 to 1.0):

1. **Clinical Correctness**: Is the term supported by the clinical note?
2. **Completeness**: Does the term fully capture the clinical finding?
3. **Specificity**: Is the term precise enough?
4. **Component Coverage**: Does the term include all relevant modifiers?

## Diabetes Term Construction Rules

For diabetes-related terms, the expected component ordering is:

1. Control Status (e.g., "Controlled" or "Uncontrolled")
2. Diabetes Type (e.g., "Type 1" or "Type 2")
3. Complications (e.g., "with retinopathy")
4. Insulin Status (e.g., "with long-term current use of insulin")
5. Oral Medication (e.g., "with oral medication")

Example: `Uncontrolled Type 2 diabetes mellitus with retinopathy with long-term current use of insulin`

## Logging

All services include structured logging. Logs are written to stdout in the format:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

Key log points:
- Configuration loading
- Judge initialization
- API request/response
- LLM provider calls
- Score aggregation
- Threshold evaluation
- Error handling

## Testing

Place your test data in:
- `data/gold_standard/gold_standard.json`
- `data/pipeline_output/pipeline_output.json`

Run gold evaluation to validate judge performance:

```bash
curl -X POST "http://localhost:8000/gold_evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "judges": ["claude-3.7-sonnet"],
    "prompt_template": "prompt_a"
  }'
```

## Provider Configuration

### AWS Bedrock

Requires AWS credentials configured via AWS CLI or environment variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

### Azure OpenAI

Configured via environment variables in `.env`.

### Ollama

Start Ollama locally:
```bash
ollama serve
```

Enable in `config/config.yaml`:
```yaml
judges:
  - provider: ollama
    model: qwen3:1.7b
    endpoint: http://localhost:11434
    enabled: true
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Invalid request (e.g., invalid judge names)
- `404`: Resource not found (e.g., gold standard file)
- `422`: Validation error (e.g., missing required fields)
- `500`: Internal server error
- `503`: Service unavailable (e.g., Ollama unreachable)

## Development

### Adding a New Provider

1. Create a new provider class in `app/providers/` inheriting from `BaseProvider`
2. Implement the `evaluate()` method
3. Register the provider in `app/services/judge_registry.py`
4. Update `config/config.yaml` with the new provider

### Adding a New Scoring Dimension

1. Add the dimension to `config/config.yaml` metrics list
2. Update prompt templates in `prompts/` to include the new dimension
3. Update `app/schemas/responses.py` if needed
4. Add threshold for the new dimension in `config/config.yaml`

## License

[Add your license here]

## Support

For issues and questions, please open an issue on the GitHub repository.
