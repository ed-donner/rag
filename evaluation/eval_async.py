import sys
import math
from pydantic import BaseModel, Field
from litellm import acompletion
from dotenv import load_dotenv
from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context
import asyncio

load_dotenv(override=True)
MODEL = "gpt-4.1-nano"
db_name = "vector_db"

class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")

class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""
    feedback: str = Field(
        description="1 sentence feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - fully comprehensive)"
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question)"
    )

def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0

def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate the Discounted Cumulative Gain (DCG)."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg

def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance)."""
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """Evaluate retrieval performance for a test question."""
    retrieved_docs = fetch_context(test.question)
    
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0
    
    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )

async def call_llm_async_with_retry(
    call_fn,
    max_retries=10,
    base_delay=10.0,
    max_delay=60.0,
    rate_limit_error_codes=("rate_limit_exceeded", 429)
):
    """
    Async retry wrapper with exponential backoff for rate limit handling.
    call_fn should be an async function with no parameters.
    """
    retries = 0
    delay = base_delay
    
    while retries < max_retries:
        try:
            return await call_fn()
        except Exception as e:
            err_is_rate = False
            
            # Check status code
            if hasattr(e, "status_code") and e.status_code == 429:
                err_is_rate = True
            
            # Check response JSON
            if hasattr(e, "response") and hasattr(e.response, "json"):
                try:
                    resp_json = e.response.json()
                    if "error" in resp_json:
                        error_data = resp_json["error"]
                        if "code" in error_data and error_data["code"] in rate_limit_error_codes:
                            err_is_rate = True
                        if "message" in error_data and "rate limit" in error_data["message"].lower():
                            err_is_rate = True
                except Exception:
                    pass
            
            # Check error message string
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or any(str(code) in err_str for code in rate_limit_error_codes):
                err_is_rate = True
            
            if err_is_rate:
                print(f"Rate limit hit (attempt {retries+1}/{max_retries}): {e}")
                print(f"Sleeping {delay:.1f}s before retry...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                retries += 1
                continue
            
            # Not a rate limit error, raise immediately
            raise
    
    raise Exception(f"Rate limit: All {max_retries} retries exhausted")

async def evaluate_answer_async(test: TestQuestion, semaphore: asyncio.Semaphore) -> tuple[AnswerEval, str, list]:
    """
    Async evaluation of answer quality with concurrency control.
    """
    async with semaphore:
        loop = asyncio.get_event_loop()
        generated_answer, retrieved_docs = await loop.run_in_executor(
            None, answer_question, test.question
        )
        
        context_str = "\n\n".join(
            [f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in retrieved_docs]
        )
        
        judge_messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator assessing the quality of AI-generated answers. Evaluate the generated answer by comparing it to the reference answer and verifying it against the retrieved context.",
            },
            {
                "role": "user",
                "content": f"""Question: {test.question}

Retrieved Context:
{context_str}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer?
2. Completeness: How thoroughly does it address all aspects of the question?
3. Relevance: How well does it directly answer the specific question asked?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
            },
        ]
        
        async def api_call():
            return await acompletion(
                model=MODEL,
                messages=judge_messages,
                response_format=AnswerEval
            )
        
        judge_response = await call_llm_async_with_retry(api_call)
        answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)
        
        return answer_eval, generated_answer, retrieved_docs

def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress

def evaluate_all_answers():
    """
    Evaluate all answers to tests using batched async execution with rate-limit handling.
    IMPORTANT: This function signature must remain intact for compatibility.
    """
    tests = load_tests()
    total_tests = len(tests)
    
    # Recommended batch size: 5-8 concurrent requests
    # This balances throughput with rate limit safety
    BATCH_SIZE = 5
    MAX_CONCURRENT = 5  # Semaphore limit for concurrent API calls
    
    async def run_async_batch():
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        results = []
        
        for i in range(0, total_tests, BATCH_SIZE):
            batch_tests = tests[i:i + BATCH_SIZE]
            print(f"\nProcessing batch {i//BATCH_SIZE + 1} ({i+1}-{min(i+BATCH_SIZE, total_tests)}/{total_tests})")
            
            tasks = [evaluate_answer_async(test, semaphore) for test in batch_tests]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, (test, result) in enumerate(zip(batch_tests, batch_results)):
                if isinstance(result, Exception):
                    print(f"Error evaluating test {i+j}: {result}")
                    answer_eval = AnswerEval(
                        feedback=f"Evaluation failed: {str(result)}",
                        accuracy=1.0,
                        completeness=1.0,
                        relevance=1.0
                    )
                else:
                    answer_eval = result[0]
                
                progress = (i + j + 1) / total_tests
                results.append((test, answer_eval, progress))
            
            # Small delay between batches to be extra safe with rate limits
            if i + BATCH_SIZE < total_tests:
                await asyncio.sleep(1.0)
        
        return results
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    all_results = loop.run_until_complete(run_async_batch())
    
    for test, result, progress in all_results:
        yield test, result, progress

async def run_cli_evaluation_async(test_number: int):
    """Run evaluation for a specific test (async)."""
    tests = load_tests("tests.jsonl")
    
    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)
    
    test = tests[test_number]
    
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")
    
    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")
    retrieval_result = evaluate_retrieval(test)
    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")
    
    # Answer Evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")
    
    try:
        semaphore = asyncio.Semaphore(1)
        answer_result, generated_answer, retrieved_docs = await evaluate_answer_async(test, semaphore)
    except Exception as e:
        print(f"Error in answer evaluation: {e}")
        answer_result = AnswerEval(
            feedback=f"Evaluation failed: {str(e)}",
            accuracy=1.0,
            completeness=1.0,
            relevance=1.0
        )
        generated_answer = ""
        retrieved_docs = []
    
    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")

def main():
    """CLI to evaluate a specific test by row number."""
    if len(sys.argv) != 2:
        print("Usage: uv run eval.py <test_row_number>")
        sys.exit(1)
    
    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)
    
    asyncio.run(run_cli_evaluation_async(test_number))

if __name__ == "__main__":
    main()