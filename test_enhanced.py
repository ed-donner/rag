#!/usr/bin/env python3
"""Test script for enhanced RAG implementation."""

def test_enhanced_implementation():
    print("Testing Enhanced RAG Implementation")
    print("=" * 60)
    
    try:
        from implementation.answer_enhanced import answer_question, fetch_context
        
        # Test cases
        test_questions = [
            "Who won the IIOTY award in 2023?",
            "How many employees does Insurellm have?",
            "What is Insurellm's vision statement?",
            "Who founded Insurellm and when?",
            "What products does Insurellm offer?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test {i} ---")
            print(f"Question: {question}")
            
            # Test retrieval
            docs = fetch_context(question)
            print(f"Retrieved {len(docs)} documents")
            
            # Check document quality
            if docs:
                print(f"Document types: {[doc.metadata.get('doc_type', 'unknown') for doc in docs[:3]]}")
                print(f"Entity names: {[doc.metadata.get('entity_name', 'unknown') for doc in docs[:3]]}")
            
            # Test full answer
            answer, context_docs = answer_question(question)
            print(f"Answer: {answer[:100]}...")
            print(f"Context docs: {len(context_docs)}")
            
            # Check for specific improvements
            if "IIOTY" in question:
                keywords = ['Maxine', 'Thompson', 'IIOTY']
                found_keywords = []
                for doc in context_docs:
                    content_lower = doc.page_content.lower()
                    for keyword in keywords:
                        if keyword.lower() in content_lower:
                            found_keywords.append(keyword)
                print(f"Keywords found: {found_keywords}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_implementation()
    if success:
        print("\n✅ Enhanced implementation test completed successfully!")
    else:
        print("\n❌ Enhanced implementation test failed!")
