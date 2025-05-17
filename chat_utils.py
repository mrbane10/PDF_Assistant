def rewrite_query(query, conversation_history, client):
    """
    Rewrite ambiguous follow-up queries based on conversation history.
    Optimized for cloud deployment with better error handling.
    """
    # Return original query if conversation history is too short
    if len(conversation_history) < 2:
        return query
        
    try:
        # Get the last few exchanges (up to 3 for context)
        last_messages = conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
        
        # Find the most recent user-assistant exchange
        user_message = ""
        system_response = ""
        
        # Look for the most recent user-assistant pair
        for i in range(len(last_messages) - 1):
            if (last_messages[i].get('role', '').lower() == 'user' and 
                last_messages[i+1].get('role', '').lower() == 'assistant'):
                user_message = last_messages[i].get('content', '')
                system_response = last_messages[i+1].get('content', '')
                break
        
        # If we couldn't find a valid exchange, return original query
        if not user_message or not system_response:
            return query
            
        # Prepare a simplified context for the rewriting model
        # Truncate long messages to avoid token limits
        if len(user_message) > 500:
            user_message = user_message[:500] + "..."
        if len(system_response) > 1000:
            system_response = system_response[:1000] + "..."
        
        # Create the prompt for query rewriting
        prompt = f"""Based on this conversation history, if the current query contains pronouns or references that depend on previous context, rewrite it as a standalone question:

Previous User Question: "{user_message}"
Previous System Answer: "{system_response[:300]}..."
Current Query: "{query}"

If the Current Query needs rewriting to be clear on its own, rewrite it. Otherwise, return it unchanged.
Output ONLY the rewritten or original query with NO explanation."""

        # Use a simpler, smaller model for query rewriting to save resources
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Smaller, faster model
            messages=[
                {"role": "system", "content": "You rewrite ambiguous follow-up questions into clear standalone questions. Output ONLY the rewritten query (or the original if already standalone). DO NOT include explanations or metadata."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        rewritten_query = response.choices[0].message.content.strip()
        
        # If rewritten query is empty, too short, or too different, fall back to original
        if not rewritten_query or len(rewritten_query) < 5 or len(rewritten_query) > len(query) * 3:
            return query
            
        return rewritten_query
        
    except Exception as e:
        print(f"Error rewriting query: {e}")
        # Fall back to original query in case of any error
        return query
